import torch
from kernel_calculating import *

def batched_cholesky_solve(K_xu_flat, K_uu_chol):
    results = []
    for i in range(0, K_xu_flat.size(0)):                                                          # 时间步
        chunk = K_xu_flat[i]                                                                       # [N, M]
        # K_uu_inv = torch.cholesky_inverse(K_uu_chol)
        # out1 = chunk @ K_uu_inv
        out = torch.cholesky_solve(chunk.transpose(-1,-2), K_uu_chol).transpose(-1,-2)             # [N, M]
        results.append(out)
    return torch.stack(results, dim=0)                                                             # [T, N, M]

def estimate_natural_parameters_vectorized(
    Q, K_uu_chol, Z,
    length_scale, variance,
    use_particles,                      # True 推荐
    time_idx, scale_factor,
    particles=None, prelogw=None,       # [T+1,N,d], [T+1,N]
    pf_x_mean=None,                     # 仅在 use_particles=False 时使用
    mean_fn =None,                     # 高斯过程的均值函数
):
    """
       用粒子 + 权重的蒙特卡洛期望来更新自然参数
       对应 GetOptimalqu 的逻辑
    """
    device = K_uu_chol.device
    dtype  = K_uu_chol.dtype
    if use_particles:
        assert particles is not None and prelogw is not None
        T = particles.size(0) - 1                     # 时间步
        N = particles.size(1)                         # 粒子数
        d = particles.size(2)                         # dim_x
        x_prev = particles[:-1].to(dtype)                       # [T,N,d]
        x_curr = particles[1:].to(dtype)                        # [T,N,d]
        # 归一化权重（log-sum-exp，数值稳定）
        logw_prev = prelogw[:-1]                                                         # [T,N]
        logw_t    = prelogw[1:]                                                          # [T,N]
        w_prev    = (logw_prev - torch.logsumexp(logw_prev, dim=1, keepdim=True)).exp().to(dtype)  # [T,N]
        w         = (logw_t - torch.logsumexp(logw_t, dim=1, keepdim=True)).exp().to(dtype)        # [T,N]
    else:
        assert pf_x_mean is not None
        T = pf_x_mean.size(0) - 1
        N = 1
        d = pf_x_mean.size(1)
        x_prev = pf_x_mean[:-1].unsqueeze(1)          # [T,1,d]
        x_curr = pf_x_mean[1:].unsqueeze(1)           # [T,1,d]
        w = torch.ones(T, 1, device=device, dtype=dtype)           # 均匀
        w_prev = torch.ones(T,1, device=device, dtype=dtype)

    if time_idx is None:
        time_idx_arr = list(range(1, T + 1))
    else:
        # allow torch tensor or list
        if isinstance(time_idx, torch.Tensor):
            time_idx_arr = time_idx.detach().cpu().tolist()
        else:
            time_idx_arr = list(time_idx)

    K_selected = len(time_idx_arr)
    if K_selected == 0:
        raise ValueError("time_idx provided is empty.")

    M = Z.size(0); Md = M * d
    Q_chol  = torch.linalg.cholesky(Q)                # [d, d]
    Q_inv   = torch.cholesky_inverse(Q_chol)          # [d, d]

    K_uu_inv  = torch.cholesky_inverse(K_uu_chol)     # [M, M]
    x_prev_flat = x_prev.reshape(T * x_prev.size(1), d)
    K_xu_flat = kernel_xu_batch(x_prev_flat, Z=Z, lengthscale=length_scale, variance=variance)  # [T*N, M]
    K_xu      = K_xu_flat.reshape(T, N, M)
    # H_flat = torch.cholesky_solve(K_xu_flat.transpose(-1, -2), K_uu_chol).transpose(-1, -2)
    H = batched_cholesky_solve(K_xu, K_uu_chol)       # [T, N, M]

    # GP 均值
    if mean_fn is None:
        m = torch.zeros_like(x_curr)  # 默认零均值
    else:
        m = mean_fn(x_prev)  # [T,N,d]

    # Q^{-1} x 和 Q^{-1} H
    eta_1 = torch.zeros(M, d, device=device, dtype=dtype)
    eta_2 = torch.zeros(Md, Md, device=device, dtype=dtype)

    # sum1  = torch.zeros(Md, 1, device=device)
    # sum2  = torch.zeros(Md, Md, device=device)

    for t in time_idx_arr:
        tp = t-1                                               # time_idx_arr是从1开始，所以当前时刻要减一
        Hprev = H[tp]                                           # [N, M]
        Xc    = x_curr[tp] - m[tp]                               # [N, d]
        wt    = w[tp].unsqueeze(-1)                             # [N, 1]
        wprev = w_prev[tp].unsqueeze(-1)                        # [N, 1]

        # η1 累加: H_t^T Xc Q^{-T} / N
        # HX = ((wprev * Hprev).sum(0).unsqueeze(-1)) @ (wt * Xc).sum(0).unsqueeze(0)                               # [M, d]
        # eta_1 += (Q_inv @ HX.T).reshape(M,d)                   # [M, d]
        # ----- η1: sum_n w_t * H_n^T Q^{-1} (x_n - m_n)  ->  [M,d]
        # 先 v_n = Q^{-1} (x_n-m_n)
        V = Xc @ Q_inv.T  # [N,d]（Qinv 对称，.T 不影响）
        # (H^T * wt) @ V :  (M×N) ⊙ (1×N)  @ (N×d) = (M×d)
        HX = (Hprev.T * wt.squeeze(-1)).contiguous() @ V
        eta_1 = eta_1 + HX  # [M,d]
        # eta_1 += (Hprev.T @ (wt * Xc)) @ Q_inv

        # ----- η2: sum_n w_t * (H_n H_n^T)   ->  [M,M], 然后与 Q^{-1} 做 Kronecker
        Bt = (Hprev * wt).T @ Hprev  # [M,M] = Σ w_t h h^T
        # Bt = ((wprev * Hprev).sum(0).unsqueeze(-1)) @ Hprev
        X = torch.einsum("ij,kl->ikjl", Q_inv, Bt).reshape(d * M, d * M)
        eta_2 = eta_2 + X

    # 加常数项并乘 -1/2
    blocks = [K_uu_inv] * d
    kron_term = torch.block_diag(*blocks)                # [d*M, d*M]
    eta_1 = scale_factor * eta_1.T.contiguous().view(-1)                # [d*M,]
    eta_2 = -0.5 * (kron_term + eta_2)
    eta_2 = 0.5 * (eta_2 + eta_2.T)                      # 对称
    eta_2 = scale_factor * (eta_2 - 1e-6 * torch.eye(Md, device=device))  # 确保是负定

    return eta_1, eta_2




