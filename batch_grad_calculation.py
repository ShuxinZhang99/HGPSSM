import torch
import math
import pyro.distributions as dist
import torch.nn.functional as F
from kernel_calculating import *
from batch_sampling_xt import natural_to_mean_cov, Spatial_effect_matrix_batch

torch.autograd.set_detect_anomaly(True)

# 为了少重复代码，封装一个H、Psi的批量计算代码
def compute_H_Psi(x_prev_b, Z, K_uu_chol, length_scale, variance):  # x_prev_b: [N,d]
    K_xu_b = kernel_xu_batch(x_prev_b, Z, lengthscale=length_scale, variance=variance)          # [N, M]
    H_b = torch.cholesky_solve(K_xu_b.transpose(-1,-2), K_uu_chol).transpose(-1, -2)                    # [N, M]
    K_xx_b = kernel_xu_batch(x_prev_b, x_prev_b, lengthscale=length_scale, variance=variance)   # [N, N]
    K_uu_inv = torch.cholesky_inverse(K_uu_chol)
    Kxu_uu_ux = ((K_xu_b @ K_uu_inv) @ K_xu_b.T)
    Psi_b = K_xx_b - Kxu_uu_ux

    return H_b, Psi_b  # [N, M], [N,N]

def _sample_theta(theta_loc, theta_scale, t, num_theta_samples):
    # 支持 theta_loc/scale 为 [3] 或 [T+1, 3]两种形状
    # theta_loc   = parameter_list['theta_loc'].to(device=device, dtype=dtype)
    # theta_scale = parameter_list['theta_scale'].to(device=device, dtype=dtype)
    if theta_loc.dim() == 2:                            # [T+1, 3]
        loc   = theta_loc[t]
        scale = torch.clamp(theta_scale[t], min=1e-6)
    else:                                               # [3]
        loc   = theta_loc
        scale = torch.clamp(theta_scale, min=1e-6)
    q_theta   = dist.Normal(loc, scale).to_event(1)
    return q_theta.rsample((num_theta_samples,))        # [num_theta, 3]


def elbo_grad(
        z_obs,                   # [T, dim_z] 只返回长度
        particles, est_log_w,    # [T+1, N, d], [T+1, N] 重采样前的权重
        parameter_list,          # 包含： Z, log_Q_loc, eta_1. eta_2,
        mean_fn=None,
        dtype=torch.float64,     # 高精度计算梯度更稳
        jitter_Q=1e-6,
        jitter_R=1e-6,
        jitter_Kuu=1e-6,
        num_theta_samples=8,
        num_y_samples=8,
        antithetic=True,        # 使用对偶采样降低方差
        obs_model='Poisson',
        scale_factor=1.0,       # 用于 SVI: scale = T / W_actual
        time_idx=None,          # 可选: tensor/list of time indices (1-based) to compute ELBO on
        cache=None              # 可选: dict with precomputed 'H_b_list','Psi_b_list','Kuu','Luu','Kuu_inv_block'
):
    """
    Modified elbo_grad supporting time mini-batch (time_idx), scaling (scale_factor), and caching.
    cache: optional dict to provide precomputed quantities:
    """
    device = particles.device
    T_full = z_obs.size(0)
    dim_z = z_obs.size(1)
    N, d = particles.size(1), particles.size(2)
    dim_y = d

    # 取参数（不要对Q、log_Q_loc以及核函数参数做detach）,同时确保是叶子&可导
    Z = parameter_list['Z'].to(device=device, dtype=dtype)
    # Z.requires_grad_(True)
    M = Z.shape[0]

    log_Q_loc = parameter_list['log_Q_loc'].to(device=device, dtype=dtype)
    log_Q_loc.requires_grad_(True)
    diag_Q = torch.exp(log_Q_loc)
    Q = torch.diag(diag_Q) + jitter_Q * torch.eye(d, device=device, dtype=dtype)
    L_Q = torch.linalg.cholesky(Q)
    Q_inv = torch.cholesky_inverse(L_Q)

    log_R_loc = parameter_list['log_R_loc'].to(device=device, dtype=dtype)
    log_R_loc.requires_grad_(True)
    diag_R = torch.exp(log_R_loc)
    R = torch.diag(diag_R) + jitter_R * torch.eye(dim_y, device=device, dtype=dtype)
    R_chol = torch.linalg.cholesky(R)

    log_G_loc = parameter_list['log_G_loc'].to(device=device, dtype=dtype)
    log_G_loc.requires_grad_(True)
    diag_G = torch.exp(log_G_loc)
    G = torch.diag(diag_G) + jitter_R * torch.eye(dim_z, device=device, dtype=dtype)
    G_chol = torch.linalg.cholesky(G)

    length_scale = parameter_list['length_scale'].to(device=device, dtype=dtype)       # tensor [d]或 标量
    variance = parameter_list['variance'].to(device=device, dtype=dtype)               # 标量
    length_scale.requires_grad_(True)
    variance.requires_grad_(True)

    # 潜在观测的可学习参数 theta
    theta_loc = parameter_list['theta_loc'].to(device=device, dtype=dtype)
    theta_scale = parameter_list['theta_scale'].to(device=device, dtype=dtype)
    # theta_loc.requires_grad_(True)
    # theta_scale.requires_grad_(True)

    # 负二项的可学习原始参数 rho_nb(r = softplus(rho_nb))、alpha、beta
    rho_nb = parameter_list['rho_nb'].to(device=device, dtype=dtype)
    rho_nb.requires_grad_(True)
    r = F.softplus(rho_nb) + 1e-8                                                      # 保证为正

    alpha = parameter_list['alpha'].to(device=device, dtype=dtype)                     # [dim_z, dim_y]
    beta  = parameter_list['beta'].to(device=device, dtype=dtype)                      # [dim_z,]
    alpha.requires_grad_(True)
    beta.requires_grad_(True)

    # 视为常量（不让梯度传播）
    eta_1 = parameter_list['eta_1'].to(device=device, dtype=dtype).detach()
    eta_2 = parameter_list['eta_2'].to(device=device, dtype=dtype).detach()
    mu_u, Sigma_u = natural_to_mean_cov(eta_1, eta_2)                                  # 视作常量 [dM], [dM, dM]

    # ------------------- Kuu / Luu / Kuu_inv_block (可缓存) -------------------
    if cache is not None and 'Kuu' in cache and 'Kuu_inv_block' in cache and 'Luu' in cache:
        Kuu = cache['Kuu'].to(device=device, dtype=dtype)
        Luu = cache['Luu'].to(device=device, dtype=dtype)
        Kuu_inv_block = cache['Kuu_inv_block'].to(device=device, dtype=dtype)
    else:
        Kuu = kernel_uu(Z=Z, lengthscale=length_scale, variance=variance)
        Kuu = 0.5 * (Kuu + Kuu.T) + jitter_Kuu * torch.eye(Kuu.size(0), device=device, dtype=dtype)
        Luu = torch.linalg.cholesky(Kuu)
        Kuu_inv = torch.cholesky_inverse(Luu)
        Kuu_inv_block = torch.block_diag(*([Kuu_inv] * d))
        Kuu_inv_block = Kuu_inv_block.to(device=device, dtype=dtype)                                  # [d*M, d*M]

    Md = Sigma_u.size(0)

    # ------------------- Prior term E_q[log p(u)] -------------------
    term1 = torch.trace(Kuu_inv_block @ Sigma_u)
    mu_u_vec = mu_u.view(-1)
    term2 = mu_u_vec @ (Kuu_inv_block @ mu_u_vec)
    term4 = d * 2.0 * torch.sum(torch.log(torch.diagonal(Luu)))
    E_log_pu = -0.5 * (term1 + term2 + term4 + Md * math.log(2 * math.pi))

    # 准备时间索引（支持 mini-batch）
    # 输入 time_idx 假定为 1-based indices consistent with original loop, values in [1..T_full]
    if time_idx is None:
        time_idx_arr = list(range(1, T_full + 1))
    else:
        # allow torch tensor or list
        if isinstance(time_idx, torch.Tensor):
            time_idx_arr = time_idx.detach().cpu().tolist()
        else:
            time_idx_arr = list(time_idx)

    K_selected = len(time_idx_arr)
    if K_selected == 0:
        raise ValueError("time_idx provided is empty.")

    # If cache has H_b_list and Psi_b_list, use them to avoid recomputation
    use_cache_H = cache is not None and 'H_b_list' in cache and 'Psi_b_list' in cache

    # 第三和第四部分 (B)+(C) trace 增广与转移高斯（使用粒子做期望， 对核 & Q都可导）
    term_trace_sum = Q.new_tensor(0.0)
    term_trans_sum = Q.new_tensor(0.0)
    obs_sum = R.new_tensor(0.0)

    for t in time_idx_arr:
        x_prev = particles[t-1].to(dtype)                                          # [N,d]
        x_t    = particles[t].to(dtype)                                            # [N,d]
        z_t    = z_obs[t-1].to(device=device, dtype=dtype)                         # [dim_z]

        # ---- 规范化权重（数值稳定）----
        w_prev_norm = torch.softmax(est_log_w[t - 1], dim=0)         # [N,]
        w_t_norm = torch.softmax(est_log_w[t], dim=0)                # [N,]

        # H, Psi 计算
        # compute or retrieve H_b, Psi_b
        if use_cache_H:
            H_b = cache['H_b_list'][t - 1].to(device=device, dtype=dtype)  # [N, M]
            Psi_b = cache['Psi_b_list'][t - 1].to(device=device, dtype=dtype)  # [N, N]
        else:
            H_b, Psi_b = compute_H_Psi(x_prev, Z, Luu, length_scale, variance)  # H_b:[N,M], Psi_b:[N,N]

        # trace term: -0.5 * trace(Q_inv @ (Phi + H Sigma H^T)) -----
        # 先计算phi部分
        if Psi_b.dim() == 2 and Psi_b.shape[0] == N and Psi_b.shape[1] == N:
            per_particle_psi = torch.diagonal(Psi_b, dim1=0, dim2=1) if Psi_b.dim() == 3 else Psi_b.diag() if \
            Psi_b.shape[0] == N else torch.sum(Psi_b, dim=(1, 2))
            try:
                E_Psi = (w_prev_norm * per_particle_psi).sum()
            except Exception:
                E_Psi = torch.trace(w_prev_norm * Psi_b)
        else:
            # if Psi_b is [N,] or [N,1]
            E_Psi = (w_prev_norm * Psi_b.view(-1)).sum()

        term_phi = E_Psi * torch.trace(Q_inv)

        # compute HSigmaH term in vectorized manner
        # Sigma_u is (d*M, d*M); we treat it as d blocks of (M x M) along diagonal
        Sigma_blocks = Sigma_u.view(d, M, d, M).permute(0, 2, 1, 3).contiguous()  # shape (d, d, M, M)
        # We only need diagonal blocks Sigma_blocks[i,i] -> (M,M)
        Sigma_diag_blocks = Sigma_blocks[range(d), range(d)]  # (d, M, M)

        # HS = H_b @ Sigma_block -> for all i: HS_i shape (N, M)
        # compute val_per_particle for all i at once -> val shape (d, N)
        # H_b: (N, M); Sigma_diag_blocks: (d, M, M)
        # compute HS_i = H_b @ Sigma_diag_blocks[i]  -> use einsum
        HS = torch.einsum('nm,imj->inj', H_b, Sigma_diag_blocks)  # (i=d, n=N, j=M)
        # val_per_particle = sum_j HS * H_b -> (i, n)
        val_per_particle = (HS * H_b.unsqueeze(0)).sum(dim=-1)  # (d, N)
        # E over particles
        E_val = (w_prev_norm.unsqueeze(0) * val_per_particle).sum(dim=1)  # (d,)
        # multiply by Q_inv diagonal (assumes Q is diagonal here)
        Qinv_diag = torch.diagonal(Q_inv)  # (d,)
        term_hsigh = (Qinv_diag * E_val).sum()

        term_trace_t = -0.5 * (term_phi + term_hsigh)

        # ----- transition Gaussian term: E[ log N(x_t | H mu_u, Q) ]  -----
        mu_u_mat = mu_u.contiguous().view(d, M).T  # (M, d)
        mu = torch.einsum('nm,md->nd', H_b, mu_u_mat)  # (N, d)
        if mean_fn is not None:
            mu = mu + mean_fn(x_prev)

        # batch multivariate normal: broadcast Q to N
        mvn = dist.MultivariateNormal(loc=mu, covariance_matrix=Q)
        logp = mvn.log_prob(x_t)  # (N,)
        term_trans_t = (w_t_norm * logp).sum()

        term_trace_sum = term_trace_sum + term_trace_t
        term_trans_sum = term_trans_sum + term_trans_t

        # ----- observation term: use theta_samples, y_samples as before (vectorized) -----
        # theta sampling (antithetic)
        if antithetic and (num_theta_samples % 2 == 0):
            half = num_theta_samples // 2
            A = torch.randn(half, 3, device=device, dtype=dtype)
            eps_theta = torch.cat([A, -A], dim=0)
        else:
            eps_theta = torch.randn(num_theta_samples, 3, device=device, dtype=dtype)
        theta_samples = theta_loc + eps_theta @ torch.linalg.cholesky(theta_scale).T  # [num_theta, 3]
        S_gamma_t = Spatial_effect_matrix_batch(theta_samples, dim_y=dim_y)  # [num_theta, 3, 3]
        S_gamma_batch = S_gamma_t.unsqueeze(0).expand(N, -1, -1, -1)  # [N, num_theta, 3, 3]

        # sample y using reparameterization, vectorized
        mean_y = torch.matmul(S_gamma_batch, x_t.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # [N, num_theta, dim_y]
        if antithetic and (num_y_samples % 2 == 0):
            half = num_y_samples // 2
            A = torch.randn(half, N, num_theta_samples, dim_y, device=device, dtype=dtype)
            eps_y = torch.cat([A, -A], dim=0)
        else:
            eps_y = torch.randn(num_y_samples, N, num_theta_samples, dim_y, device=device, dtype=dtype)

        y_samples = mean_y.unsqueeze(0) + torch.matmul(eps_y, R_chol.T)  # [num_y, N, num_theta, dim_y]

        logits = torch.einsum('zd, sntd -> sntz', alpha, y_samples) + beta  # [num_y, N, num_theta, dim_z]
        logits = logits.clamp(-5.0, 8.0)
        mu_log = torch.exp(logits)
        # mu_log = F.softplus(logits) + 1e-6
        
        if obs_model == "NB":
            # 负二项分布 (r, p)
            p = mu_log / (mu_log + r)
            nb = dist.NegativeBinomial(total_count=r, probs=p)
            log_lik = nb.log_prob(z_t.view(1, 1, 1, -1)).sum(-1)  # [num_y_samples, N, num_theta_samples]
        elif obs_model == "Poisson":
            # 泊松分布 (λ = mu)
            pois = dist.Poisson(rate=mu_log)
            log_lik = pois.log_prob(z_t.view(1, 1, 1, -1)).sum(-1)
        elif obs_model == 'Normal':
            # 正态分布 （mu, G）
            normal = dist.MultivariateNormal(loc=mu_log, covariance_matrix=G)
            log_lik = normal.log_prob(z_t.view(1, 1, 1, -1))
        elif obs_model == "VGPSSM_NB":
            logits = torch.einsum('zd, nd -> nz', alpha, x_t) + beta  # [N, dim_z]
            logits = logits.clamp(-5.0, 6.0)
            mu_log = torch.exp(logits)
            p      = mu_log / (mu_log + r)
            nb     = dist.NegativeBinomial(total_count=r, probs=p)
            log_lik=nb.log_prob(z_t.view(1,-1)).sum(-1)
        else:
            raise ValueError(f"Unsupported likelihood_type: {obs_model}")

        # 对 y 求均值
        logp_z_mean = torch.logsumexp(log_lik, dim=0) - math.log(num_y_samples)                 # [N, num_theta]
        # 对 θ 求均值
        L_obs_ns = torch.logsumexp(logp_z_mean, dim=-1) - math.log(num_theta_samples)            # [N]
        obs_sum = obs_sum + (w_t_norm * L_obs_ns).sum(0)

    # ------------------- total ELBO and scaling for SVI -------------------
    ELBO_total = E_log_pu + term_trace_sum + term_trans_sum + obs_sum
    # scale to full sequence if using mini-batch
    ELBO_total = scale_factor * ELBO_total
    loss_total = -ELBO_total

    # # 可选：返回一些用于打印的标量
    with torch.no_grad():
        stats = {}
        stats['elbo'] = ELBO_total.detach().cpu().item()
        stats['E_log_pu'] = E_log_pu.detach().cpu().item()
        stats['trace_term'] = term_trace_sum.detach().cpu().item()
        stats['trans_term'] = term_trans_sum.detach().cpu().item()
        stats['obs_term'] = obs_sum.detach().cpu().item()

    return loss_total, stats
