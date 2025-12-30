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
        R,  # [d,d]
        r, alpha, beta,  # NB params
        theta_center, theta_cov,  # θ_{t-1} 均值与协方差
        length_scale, variance,  # GP 核参数
        num_y_samples, num_theta_samples,
        mean_fn=None,
        jitter=1e-6,
        dtype=torch.float64,
        obs_model='NB',
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

    if obs_model == 'NB':
        # ---- Laplace approx of NB ----
        if torch.is_tensor(r) and r.ndim == 0:
            r_vec = r * torch.ones((dim_z,), device=device, dtype=dtype)
        else:
            r_vec = r

        q_theta = dist.MultivariateNormal(theta_center, theta_cov)
        theta_t_samples = q_theta.rsample((num_theta_samples,))
        S_samples = Spatial_effect_matrix_batch(theta_t_samples, d).to(device=device, dtype=dtype)
        S = S_samples.mean(dim=0)

        # 1) NB 的模式点与曲率
        p_star = (z_t / (r_vec + z_t)).clamp(min=1e-8, max=1 - 1e-8)
        eta_star = torch.log(p_star) - torch.log1p(-p_star)
        W_vec = (r_vec + z_t) * p_star * (1.0 - p_star)
        W_vec = W_vec.clamp(min=1e-12)

        # 2) Precision and posterior
        rhs_term = W_vec * (eta_star - beta)

    elif obs_model == 'Poisson':
        # ---- Laplace approx of Poisson ----
        q_theta = dist.MultivariateNormal(theta_center, theta_cov)
        theta_t_samples = q_theta.rsample((num_theta_samples,))
        S_samples = Spatial_effect_matrix_batch(theta_t_samples, d).to(device=device, dtype=dtype)
        S = S_samples.mean(dim=0)

        # mode: eta_star = log(z + 1e-8)
        eta_star = torch.log(z_t + 1e-8)
        # curvature: W = exp(eta_star) = z_t
        W_vec = (z_t + 1e-3).clamp(min=1e-6)
        W_vec = W_vec.clamp(min=1e-12)
        # 对应的线性项
        rhs_term = W_vec * (eta_star - beta)
    else:
        raise ValueError(f"Unknown obs_model: {obs_model}")
    # 2) build Lambda_y = alpha^T diag(W) alpha  (d x d)
    #    alpha: [dim_z, d] -> alpha_T: [d, dim_z]
    #    Lambda_y = alpha.T @ (W diag) @ alpha
    W_diag = W_vec.view(dim_z)
    alpha_weighted = alpha * W_diag.view(-1, 1)  # [dim_z, d]
    Lambda_y = alpha_weighted.T @ alpha          # [d, d]

    # 3) compute posterior for y given z (Laplace approx)
    #    Prior for y (from x prior propagation) has mean y0 (we had y0 = m_prior @ S.T).
    #    We treat prior covariance of y as R (observation covariance) when forming posterior over y:
    #    Posterior precision: R^{-1} + Lambda_y
    #    Posterior linear term: R^{-1} @ y0 + alpha^T @ ( W * (eta_star - beta) )
    eye_d = torch.eye(d, device=device, dtype=dtype)
    R_inv = torch.linalg.inv(R)  # (d,d)  -- keep stable; R should be PD
    prec_y = R_inv + Lambda_y + jitter * eye_d  # (d,d)

    # stable cholesky / inverse
    try:
        L_prec = torch.linalg.cholesky(0.5 * (prec_y + prec_y.T))
        Sigma_y = torch.cholesky_inverse(L_prec)
    except Exception:
        eigval, eigvec = torch.linalg.eigh(0.5 * (prec_y + prec_y.T))
        eigval = eigval.clamp(min=1e-6, max=1e8)
        Sigma_y = eigvec @ torch.diag_embed(1.0 / eigval) @ eigvec.transpose(-1, -2)

    # compute the linear term b per particle: b_n = R_inv @ y0_n + alpha.T @ ( W * (eta_star - beta) )
    N = m_prior.shape[0]
    # term1: (N,d)  = R_inv @ y0_n  -> use einsum
    term1 = torch.einsum('ij,nj->ni', R_inv, m_prior)  # [N,d]
    # term2: (d,)  = alpha.T @ ( W * (eta_star - beta) )
    term2 = alpha.T @ rhs_term                               # [d]
    # combine -> per-particle
    b = term1 + term2.view(1, -1)                            # [N,d]
    # posterior mean of y for each particle:
    m_y = torch.einsum('ij,nj->ni', Sigma_y, b)        # [N, d]

    # 4) Kalman-like update from y -> x (same as your original flow but using corrected m_y & Sigma_y)
    #    We treat "observation" y ~ S x, and observed mean = m_y with covariance = R_post = R + Sigma_y
    SQST = S @ Q @ S.T
    R_eff = R.unsqueeze(0) + Sigma_y.unsqueeze(0)  # [1,d,d] or broadcast to [N,d,d]
    InnovCov = SQST.unsqueeze(0) + R_eff  # [N,d,d]
    # symmetrize + jitter
    InnovCov = 0.5 * (InnovCov + InnovCov.transpose(-1, -2)) + jitter * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    # use cholesky per-particle (or vectorized if supported)
    L_Innov = torch.linalg.cholesky(InnovCov)  # [N,d,d]
    # delta = m_y - (m_prior @ S.T)  # shape [N,d]
    delta = m_y - (m_prior @ S.T)  # [N,d]
    # compute X = InnovCov^{-1} @ delta^T efficiently via cholesky_solve
    X = torch.cholesky_solve(delta.unsqueeze(-1), L_Innov).squeeze(-1)  # [N,d]

    QS = Q @ S.T  # [d,d]
    mu_q = m_prior + torch.einsum('ij,nj->ni', QS, X)  # [N,d]

    # compute V = InnovCov^{-1} @ (S @ Q)^T via cholesky_solve
    SQ_b = (S @ Q).unsqueeze(0).expand(N, -1, -1)  # [N,d,d]
    V = torch.cholesky_solve(SQ_b.transpose(-1, -2), L_Innov).transpose(-1, -2)  # [N,d,d]

    # Sigma_q = Q - Q S^T Inv(Innov) S Q  (vectorized)
    QS_exp = Q.unsqueeze(0).expand(N, -1, -1)  # [N,d,d]
    Sigma_q = QS_exp - torch.einsum('ij,njk,nkl->nil', Q, S.T.unsqueeze(0).expand_as(V), V)
    Sigma_q = 0.5 * (Sigma_q + Sigma_q.transpose(-1, -2)) + jitter * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)

    # optionally mild inflation for numeric robustness
    Sigma_q = Sigma_q + 0.2 * Q_b

    K = torch.einsum("ij,njk->nik", Q @ S.T, torch.linalg.inv(InnovCov))

    # ---- 最终采样 ----
    mvn_q = dist.MultivariateNormal(loc=mu_q, covariance_matrix=Sigma_q)
    mvn_p = dist.MultivariateNormal(loc=m_prior, covariance_matrix=Q)
    x_t_particles = mvn_q.rsample()

    log_q = mvn_q.log_prob(x_t_particles)
    log_p = mvn_p.log_prob(x_t_particles)

    # ---- 观测 ----
    eps_theta = torch.randn(num_theta_samples, theta_center.size(0), device=device, dtype=dtype)
    theta_samples = theta_center + eps_theta @ torch.linalg.cholesky(theta_cov)
    S_batch = Spatial_effect_matrix_batch(theta_samples, dim_y=d)
    mean_y = torch.einsum('nd,sde->snd', x_t_particles, S_batch)
    y_mvn = dist.MultivariateNormal(loc=mean_y, covariance_matrix=R)
    y_samples = y_mvn.rsample((num_y_samples,))
    logits = torch.einsum('zd,synd->synz', alpha, y_samples) + beta

    logits = logits.clamp(-5.0, 8.5)
    mu  = torch.exp(logits)
    # mu  = F.softplus(logits) + 1e-6
    if obs_model == "NB":
        # 负二项分布 (r, p)
        p = mu / (mu + r)
        nb = dist.NegativeBinomial(total_count=r, probs=p)
        log_lik = nb.log_prob(z_t.view(1, 1, 1, -1)).sum(-1)

    elif obs_model == "Poisson":
        # 泊松分布 (λ = mu)
        pois = dist.Poisson(rate=mu)
        log_lik = pois.log_prob(z_t.view(1, 1, 1, -1)).sum(-1)
    else:
        raise ValueError(f"Unsupported likelihood_type: {obs_model}")
    log_lik = torch.logsumexp(log_lik, dim=(0, 1)) - math.log(num_y_samples) - math.log(num_theta_samples)

    # ---- trace 增广项 ----
    trace_term = compute_trace_term_batch(Q, Psi_b, H_b, Sigma_u, Q_chol=Q_chol)

    # ---- 最终权重 ----
    log_weights_pre_rs = log_lik + log_p - log_q + trace_term

    cache = {'mu_q': mu_q, 'Sigma_q': Sigma_q, 'K': K, 'log_sub': (log_p - log_q)}

    return x_t_particles, log_weights_pre_rs, H_b, Psi_b, cache


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

    # theta_cov = 0.0025 * torch.eye(3, device=device, dtype=dtype)
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
        x_t, pre_log_w, H_b, Psi_b, cache = compute_weights_vectorized(
            x_prev_particles=x_prev,
            z_t=z_t,
            Z=Z,
            mean_u=mean_u, Sigma_u=Sigma_u,
            K_uu_chol=K_uu_chol,
            Q=Q, Q_chol=Q_chol,
            R=R,
            r=r, alpha=alpha, beta=beta,
            theta_center=theta_loc, theta_cov=theta_scale,
            length_scale=length_scale, variance=variance,
            num_y_samples=num_y_samples, num_theta_samples=num_theta_samples,
            mean_fn=mean_fn, obs_model='NB'
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

        # --------- 新增诊断打印 ---------
        Sigma_q = cache['Sigma_q']  # [N,d,d]
        K = cache['K']  # [N,d,d]
        log_sub = cache['log_sub'].mean()
        trace_ratio = (Sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1).mean() / Q.diag().sum()).item()
        k_norm = K.norm(dim=(1, 2)).mean().item()
        if t % 5 == 0:  # 每5个时间步打印一次，避免太多日志
            print(f"[t={t}] trace(Sigma_q)/trace(Q)={trace_ratio:.3f}, ||K||={k_norm:.3f}, ess={ess_t.item():.3f}, log_sub={log_sub:.4f}")
            # print(f"[t={t}]  ess={ess_t.item():.3f}")

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
