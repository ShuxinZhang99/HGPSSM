import torch.linalg
import numpy as np
from kernel_calculating import *
import torch.nn.functional as F


def systematic_resample(probs):  # 系统重采样
    N = probs.size(0)

    # 累计分布
    cdf = torch.cumsum(probs, dim=0)

    # 对累计分布进行归一化（确保最后一个值为1，避免浮动过小或过大）
    cdf = cdf / cdf[-1]

    # 起点均匀随机
    u0 = torch.rand(1, device=probs.device) / N
    u = u0 + torch.arange(N, device=probs.device) / N

    # 防止最后一个u值超出范围
    u = u.clamp(max=1 - 1e-6)  # 将u值限制在 [0, 1) 范围

    # 二分查找，避免索引超出
    idx = torch.searchsorted(cdf, u)

    # 确保索引没有超出范围
    idx = idx.clamp(min=0, max=N - 1)

    return idx


def natural_to_mean_cov(eta_1, eta_2, jitter=1e-6):
    """
    从自然参数 (eta_1, eta_2) 恢复均值和协方差。
    公式: mu = Sigma @ eta_1,  Sigma = (-2*eta_2)^(-1).
    """
    # 精度矩阵：-2 * eta_2
    precision = -2.0 * eta_2  # [d*M, d*M]
    # precision = ensure_pd_by_eig(precision)  # 保证对称
    precision = 0.5 * (precision + precision.T)

    # 协方差矩阵
    L = torch.linalg.cholesky(precision)
    Sigma = torch.cholesky_inverse(L)

    Sigma = Sigma + jitter * torch.eye(Sigma.shape[0], device=Sigma.device)

    # 均值
    mu = Sigma @ eta_1  # [d*M]
    return mu, Sigma


# batch形式的系数矩阵
def Spatial_effect_matrix_batch(theta_bs, dim_y):
    """
    theta_bs: [..., 3] [Sθ,3]
    返回:     [..., dim_y, dim_y]
    """
    *batch, _ = theta_bs.shape  # theta采样样本的数量
    device = theta_bs.device
    dtype = theta_bs.dtype
    lower = theta_bs[..., 0]
    diag = theta_bs[..., 1]
    upper = theta_bs[..., 2]

    tri = torch.zeros(*batch, dim_y, dim_y, device=device, dtype=dtype)
    idx = torch.arange(dim_y, device=device)

    tri[..., idx, idx] = diag.unsqueeze(-1)  # shape [..., dim_y]
    tri[..., idx[1:], idx[:-1]] = lower.unsqueeze(-1)  # 下对角
    tri[..., idx[:-1], idx[1:]] = upper.unsqueeze(-1)  # 上对角
    return tri


@torch.no_grad()
def _ess_from_logw(logw):
    """用 log-weights 估算 ESS"""
    m = torch.logsumexp(logw, dim=0)
    w = (logw - m).exp()
    w = w / w.sum()
    ess = 1.0 / (w.pow(2).sum() + 1e-12)
    return ess, w


def compute_trace_term_batch(Q, Psi_b, H_b, Sigma_u, Q_chol=None):
    """ -1/2 trace(Q^{-1}(Psi_{t-1}+H_{t-1} Sigma_u H_{t-1}))
    Q:      [dim_x,dim_x]
    Psi_b:  [N,]
    H_b:    [N, M]
    Sigma_u:[M*d, M*d]
    返回:    [N], 每个粒子的 trace(Q^{-1} (Psi + H Σ H^T))
    """
    dim_x = Q.shape[0]
    N, M = H_b.shape
    if Q_chol is None:
        Q_chol = torch.linalg.cholesky(Q)
    Q_inv = torch.cholesky_inverse(Q_chol)
    middle = H_b.new_zeros(N)
    for i in range(dim_x):
        Si = Sigma_u[i * M:(i + 1) * M, i * M:(i + 1) * M]  # [M,M]
        HSigmaH = (H_b @ Si) @ H_b.t()  # [N,N]
        trace_HSigmaH = HSigmaH.diag()  # [N]
        trace_Psi = Psi_b.diag()
        middle = middle + Q_inv[i, i] * (trace_Psi + trace_HSigmaH)
    return -0.5 * middle


def compute_weights_vectorized(
        x_prev_particles,  # [N, dim_x]
        z_t,  # [dim_z]
        Z,  # [M, dim_u]
        mean_u, Sigma_u,  # [M*dim_u], [M*dim_u, M*dim_u]
        K_uu_chol,  # [M, M]
        Q, Q_chol,  # [d,d]
        R, G,       # [dim_z,dim_z]
        r, alpha, beta,  # NB params
        theta_center, theta_cov,  # θ_{t-1} 均值与协方差
        length_scale, variance,  # GP 核参数
        num_y_samples, num_theta_samples,
        mean_fn=None,
        jitter=1e-6,
        dtype=torch.float64,
        obs_model='VGPSSM_NB'
):
    """
    返回:
      x_t_particles:      [N, d]
      log_weights_pre_rs: [N]   （重采样前的权重）
      H_b:                [N, M]
      Psi_b:              [N]
      theta_t_samples:    [Sθ, 3]
      cache: dict         {mu_q, Sigma_q, log_prior, log_prop, m_prior, m_y, Sigma_y}
    """
    device = x_prev_particles.device
    N, d = x_prev_particles.shape
    x_prev_particles_in = x_prev_particles.detach()  # 切断计算图
    x_prev_particles_in = x_prev_particles_in.contiguous()  # 保证内存连续性
    x_prev_particles_in.requires_grad_(False)  # 明确不需要梯度
    dim_z = z_t.numel()
    M = Z.size(0)
    Q_b = Q.unsqueeze(0)

    # ---- GP prior mean ----
    K_xu_b = kernel_xu_batch(x_prev_particles_in, Z, lengthscale=length_scale, variance=variance)  # [N,M]
    H_b = torch.cholesky_solve(K_xu_b.transpose(-1, -2), K_uu_chol).transpose(-1, -2)  # [N,M]
    K_uu_inv = torch.cholesky_inverse(K_uu_chol)
    Kxu_uu_ux = ((K_xu_b @ K_uu_inv) @ K_xu_b.T)  # [N,N]
    K_xx_b = kernel_xu_batch(x_prev_particles_in, x_prev_particles_in,
                             lengthscale=length_scale, variance=variance)  # [N,N]
    Psi_b = K_xx_b - Kxu_uu_ux  # [N,N]

    mean_u_mat = mean_u.contiguous().view(d, M).T
    m_prior = torch.einsum('nm,md->nd', H_b, mean_u_mat)  # [N,d]
    if mean_fn is not None:
        m_prior = m_prior + mean_fn(x_prev_particles_in)

    mvn_p = dist.MultivariateNormal(loc=m_prior, covariance_matrix=Q)
    x_t_particles = mvn_p.rsample()

    # ---- 观测 ----
    if obs_model == 'Normal':
        eps_theta = torch.randn(num_theta_samples, theta_center.size(0), device=device, dtype=dtype)
        theta_samples = theta_center + eps_theta @ torch.linalg.cholesky(theta_cov)
        S_batch = Spatial_effect_matrix_batch(theta_samples, dim_y=d)
        mean_y = torch.einsum('nd,sde->snd', x_t_particles, S_batch)
        y_mvn = dist.MultivariateNormal(loc=mean_y, covariance_matrix=R)
        y_samples = y_mvn.rsample((num_y_samples,))
        logits = torch.einsum('zd,synd->synz', alpha, y_samples) + beta

        logits = logits.clamp(-5.0, 6.0)
        mu  = torch.exp(logits)
        # 防止高mu时似然极小，类似NB方差结构：Var ~ mu + mu^2 / r
        # 这里sigma_scale是一个经验性放缩
        sigma_scale = 0.2 + 0.8 * (mu / (mu.mean() + 1e-6)).clamp(max=5.0)

        # 构造对角协方差矩阵 G_eff
        G_diag = G.diagonal(dim1=-2, dim2=-1)  # [D_z]
        G_eff = torch.diag_embed(G_diag * sigma_scale * 5.0)
        # mu  = F.softplus(logits) + 1e-6
        # probs = F.softmax(logits, dim=2)
        nor = dist.MultivariateNormal(mu, covariance_matrix=G_eff)
        tau = 6.0  # 温度因子
        log_nb = nor.log_prob(z_t.view(1, 1, 1, -1)) / tau
        log_nb = torch.logsumexp(log_nb, dim=(0, 1)) - math.log(num_y_samples) - math.log(num_theta_samples)
    
    if obs_model == 'VGPSSM_NB':
        logits = torch.einsum('zd,nd->nz', alpha, x_t_particles) + beta
        logits = logits.clamp(-5.0, 6.0)
        mu  = torch.exp(logits)

        p = mu / (mu + r)
        nb = dist.NegativeBinomial(total_count=r, probs=p)
        log_nb = nb.log_prob(z_t.view(1,-1)).sum(-1)

    # ---- trace 增广项 ----
    trace_term = compute_trace_term_batch(Q, Psi_b, H_b, Sigma_u, Q_chol=Q_chol)

    # ---- 最终权重 ----
    log_weights_pre_rs = log_nb + trace_term

    return x_t_particles, log_weights_pre_rs, H_b, Psi_b


def particle_filter_vectorized(z_obs, parameter_list, num_particles, num_y_samples, num_theta_samples, mean_fn, dtype,
                               device=None, roughening_std=0.01):
    """
    完整的向量化 PF：
      - 批量生成/更新 N 个粒子
      - 使用“重采样前权重”来计算 pf_x_mean，避免重采样导致的均匀权重问题
    """
    if device is None:
        device = z_obs.device
    # 取参数
    Z = parameter_list['Z'].to(device=device, dtype=dtype)  # [M, dim_u]
    Q = parameter_list['Q'].to(device=device, dtype=dtype)  # [dim_x, dim_x]
    R = parameter_list['R'].to(device=device, dtype=dtype)  # [dim_y, dim_y]
    G = parameter_list['G'].to(device=device, dtype=dtype)
    rho_nb = parameter_list['rho_nb'].to(device=device, dtype=dtype)
    r = F.softplus(rho_nb) + 1e-6
    alpha = parameter_list['alpha'].to(device=device, dtype=dtype)  # [dim_z, dim_y]
    beta = parameter_list['beta'].to(device=device, dtype=dtype)  # [dim_z]
    theta_loc = parameter_list['theta_loc'].to(device=device, dtype=dtype)  # [3]
    theta_scale = parameter_list['theta_scale'].to(device=device, dtype=dtype)  # [3,3]
    length_scale = parameter_list['length_scale'].to(device=device, dtype=dtype)
    variance = parameter_list['variance'].to(device=device, dtype=dtype)

    T, dim_z = z_obs.shape
    M, dim_u = Z.shape
    dim_x = dim_u

    # u 的自然参数 -> 均值协方差
    eta_1 = parameter_list['eta_1'].to(device=device, dtype=dtype)
    eta_2 = parameter_list['eta_2'].to(device=device, dtype=dtype)
    mean_u, Sigma_u = natural_to_mean_cov(eta_1=eta_1, eta_2=eta_2)  # [d*M], [d*M, d*M]

    # K_uu & Q_chol
    K_uu = kernel_uu(Z=Z, lengthscale=length_scale, variance=variance)
    K_uu = 0.5 * (K_uu + K_uu.T) + 1e-6 * torch.eye(K_uu.size(0), device=device, dtype=dtype)
    K_uu_chol = torch.linalg.cholesky(K_uu)
    # K_uu_inv = torch.cholesky_inverse(K_uu_chol)
    Q_chol = torch.linalg.cholesky(Q)

    # 状态容器
    particles = torch.zeros(T + 1, num_particles, dim_x, device=device, dtype=dtype)
    log_weights = torch.zeros(T + 1, num_particles, device=device, dtype=dtype)  # 仅用于保存（重采样后均匀）
    est_log_w = torch.zeros(T + 1, num_particles, device=device, dtype=dtype)  # 重采样前权重的归一化log
    theta_list = torch.zeros(T + 1, 3, device=device, dtype=dtype)
    pf_x_mean = torch.zeros(T, dim_x, device=device, dtype=dtype)
    ess_list = torch.zeros(T, device=device, dtype=dtype)
    logZ_t_list = torch.zeros(T, device=device, dtype=dtype)
    ancestors = torch.zeros(T, num_particles, dtype=torch.long, device=device)  # 保存祖先索引

    # t=0 初始化
    particles[0] = dist.MultivariateNormal(torch.zeros(dim_x, device=device, dtype=dtype),
                                           covariance_matrix=torch.eye(dim_x, device=device, dtype=dtype)).rsample((num_particles,))
    theta_list[0] = dist.MultivariateNormal(theta_loc, theta_scale).rsample()
    log_weights[0] = -math.log(num_particles)  # 均匀
    est_log_w[0] = -math.log(num_particles)

    for t in range(1, T + 1):
        # 计算重采样前的权重 + 传播
        x_prev = particles[t - 1]  # [N, d]
        z_t = z_obs[t - 1].to(device=device, dtype=dtype)  # [dim_z]
        # 返回的是“候选”x_t（不要立刻覆盖成resampled）
        x_t, pre_log_w, H_b, Psi_b = compute_weights_vectorized(
            x_prev_particles=x_prev,
            z_t=z_t,
            Z=Z,
            mean_u=mean_u, Sigma_u=Sigma_u,
            K_uu_chol=K_uu_chol,
            Q=Q, Q_chol=Q_chol,
            R=R, G=G,
            r=r, alpha=alpha, beta=beta,
            theta_center=theta_loc, theta_cov=theta_scale,
            length_scale=length_scale, variance=variance,
            num_y_samples=num_y_samples, num_theta_samples=num_theta_samples,
            mean_fn=mean_fn
        )

        # ---- 规范化重采样前权重、log evidence增量 ----
        logZ_t = torch.logsumexp(pre_log_w, dim=0) - math.log(num_particles)
        logZ_t_list[t - 1] = logZ_t
        # pre_log_w = pre_log_w - pre_log_w.mean()
        est_log_w_t = pre_log_w - torch.logsumexp(pre_log_w, dim=0)  # 归一化 log w
        est_log_w[t] = est_log_w_t
        probs = est_log_w[t].exp()  # [N]

        # ---- 用候选 + 归一化pre-weights 计算 pf_x_mean ----
        pf_x_mean[t - 1] = (probs.unsqueeze(-1) * x_t).sum(dim=0)

        # ---- ESS & 可选重采样 ----
        ess_t = 1.0 / torch.clamp_min((probs ** 2).sum(), 1e-12)
        ess_list[t - 1] = ess_t

        if t % 5 == 0:  # 每5个时间步打印一次，避免太多日志
            # print(f"[t={t}] trace(Sigma_q)/trace(Q)={trace_ratio:.3f}, ||K||={k_norm:.3f}, ess={ess_t.item():.3f}, log_sub={log_sub:.4f}")
            print(f"[t={t}]  ess={ess_t.item():.3f}")

        if ess_t < 0.5 * num_particles:
            # 系统性重采样
            idx = systematic_resample(probs)  # [N] (long)
            ancestors[t - 1] = idx  # PW-R中候选 i 对应父 i，所以 idx 就是父索引
            x_t_resampled = x_t.index_select(0, idx)
            if roughening_std > 0.0:  # 可选粗化
                per_dim_std = x_t_resampled.std(dim=0, unbiased=True) + 1e-6
                noise = roughening_std * per_dim_std * torch.randn_like(x_t_resampled)
                x_t_resampled = x_t_resampled + noise
            particles[t] = x_t_resampled
            log_weights[t] = -math.log(num_particles)  # 重采样后均匀
        else:
            # 不重采样则祖先为恒等映射
            ancestors[t - 1] = torch.arange(num_particles, device=device)
            particles[t] = x_t
            log_weights[t] = est_log_w[t]  # 保留非均匀滤波权重

        # θ 的轨迹（如需，可在此处更新/采样）
        theta_list[t] = dist.MultivariateNormal(theta_loc, theta_scale).rsample()

    return (particles, est_log_w, pf_x_mean,
            K_uu, K_uu_chol, theta_list, ess_list, logZ_t_list, ancestors)
