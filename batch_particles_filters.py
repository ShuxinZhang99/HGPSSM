import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from batch_sampling_u import *
# from batch_sampling_lap_xt import *
from batch_sampling_normal_xt import *
from kernel_calculating import *
from batch_grad_calculation import *
import torch
import pyro
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def free_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# 计算关键指标
def compute_metrics(z_obs, z_mean, z_dist, n_samples=1000):
    """
    计算 RMSE, MAPE, CRPS
    z_obs: [T, d_z]
    z_mean: [T, d_z]
    z_dist: torch.distributions (预测分布, e.g. NegativeBinomial)
    """
    eps = 1e-6
    # T, Dz = z_obs.shape

    # ---- RMSE ----
    rmse = torch.sqrt(((z_obs - z_mean) ** 2).mean())

    # ---- MAPE ----
    mask = z_obs != 0
    mape = ((z_obs[mask] - z_mean[mask]).abs() / z_obs[mask]).mean() * 100

    # ---- CRPS (蒙特卡洛近似) ----
    # 从预测分布采样
    samples = z_dist.sample((n_samples,))  # [n_samples, T, d_z]
    obs_expand = z_obs.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, T, d_z]
    obs_expand = obs_expand[:, :, 9]
    samples = samples[:, :, 9]

    term1 = (samples - obs_expand).abs().mean(0)  # [T, d_z]
    term2 = torch.cdist(samples.reshape(n_samples, -1),
                        samples.reshape(n_samples, -1), p=1).mean() / 2.0
    crps = term1.mean() - term2

    return rmse.item(), mape.item() , crps.item()


# 结果可视化
def visualize_results_with_z(pf_x_particles, x_state, y_latent, z_obs, Z, mu_u, L_u, theta_list, theta_loc, theta_cov, length_scale, variance,
                             log_R_loc, log_G_loc, log_Q_loc, alpha, beta, rho_nb, ci=1.96, save_path=None, n_samples=100):
    """
        可视化粒子滤波估计结果 + 不确定性 (均值 ± 置信区间)
        pf_x_particles: [T, N, d] 粒子轨迹
        x_state: [T, d] 真实状态
        y_latent: [T, d_y] 潜在真实变量
        z_obs: [T, d_z] 观测
        """
    pf_x_particles = pf_x_particles[1:]
    T, N, d = pf_x_particles.shape

    # ---- 计算状态均值和置信区间 ----
    pf_x_mean = pf_x_particles.mean(dim=1)  # [T, d]
    pf_x_std = pf_x_particles.std(dim=1)  # [T, d]
    pf_x_low = pf_x_mean - ci * pf_x_std  # 下界 (95% CI)
    pf_x_high = pf_x_mean + ci * pf_x_std  # 上界

    x_true = x_state.detach().cpu()
    z_obs = z_obs[:T].detach().cpu()

    # ---- 推断 y 的分布（传播粒子不确定性） ----
    theta_sample = dist.MultivariateNormal(theta_loc, covariance_matrix=theta_cov).sample()
    S = Spatial_effect_matrix_batch(theta_sample, d)  # [d, d]
    R = torch.diag(torch.exp(log_R_loc))
    Q = torch.diag(torch.exp(log_Q_loc))
    G = torch.diag(torch.exp(log_G_loc)).detach().cpu()

    mean_y = (pf_x_mean @ S.T)  # [T, d]
    y_cov = R  # 假设协方差不变
    y_dist = dist.MultivariateNormal(mean_y, covariance_matrix=y_cov)
    y_mean = y_dist.mean.cpu()
    y_std = y_dist.variance.sqrt().cpu()  # [T, d]

    # ---- 观测 z 的均值和方差 ----
    r = F.softplus(rho_nb).detach().cpu()
    eta = pf_x_mean.detach().cpu() @ alpha.detach().cpu().T + beta.detach().cpu()
    eta = eta.clamp(-5.0, 6.0)
    mu = torch.exp(eta)
    # mu = F.softplus(eta) + 1e-6
    p = mu / (mu + r)
    z_dist = dist.NegativeBinomial(total_count=r, probs=p)
    # z_dist = dist.MultivariateNormal(loc=mu, covariance_matrix=G)
    # z_dist = dist.Poisson(rate=mu)
    z_mean = mu  # NB 的均值 = mu

    alpha_ci = 0.05
    # z_dist: batch NB with batch_shape [T,N]
    num_samples = 100
    # 采样得到 shape [num_samples, T, N]
    samples = z_dist.sample((num_samples,))  # 若内存受限可逐批采样
    z_low = samples.quantile(alpha_ci / 2, dim=0)  # shape [T,N]
    z_high = samples.quantile(1 - alpha_ci / 2, dim=0)

    # ---- 计算评估指标 ----
    rmse, mape, _ = compute_metrics(z_obs, z_mean, z_dist, n_samples=n_samples)
    print(f'[ALL], RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')
    for i in range(z_obs.shape[1]):
        rmse, mape, _ = compute_metrics(z_obs[:, i], z_mean[:, i], z_dist, n_samples=n_samples)
        print(f"[Station {i}], RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")  # CRPS: {crps:.4f}

    # ---- 画图 ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 状态 x
    axes[0].plot(x_true[:, 2], label='x_true_2')
    axes[0].plot(pf_x_mean[:, 2].detach().cpu(), '--', label='x_est_2')
    axes[0].fill_between(range(T), pf_x_low[:, 2].detach().cpu(), pf_x_high[:, 2].detach().cpu(),
                         color='gray', alpha=0.3, label='95% CI')
    axes[0].set_ylabel('State x')
    axes[0].legend()

    # 潜在变量 y
    axes[1].plot(y_latent[:, 1].detach().cpu(), label='y_true_1')
    axes[1].plot(y_mean[:, 1].detach().cpu(), '--', label='y_est_1')
    axes[1].fill_between(range(T), (y_mean[:, 1].detach().cpu() - ci * y_std[:, 1].detach().cpu()),
                         (y_mean[:, 1].detach().cpu() + ci * y_std[:, 1].detach().cpu()),
                         color='blue', alpha=0.2, label='95% CI')
    axes[1].set_ylabel('Latent y')
    axes[1].legend()

    # 观测 z
    axes[2].plot(z_obs[:, 1], label='z_obs_5')
    axes[2].plot(z_mean[:, 1].detach().cpu(), '--', label='z_est_5')
    axes[2].fill_between(range(T), z_low[:, 1].detach().cpu(), z_high[:, 1].detach().cpu(),
                         color='orange', alpha=0.3, label='95% CI')
    axes[2].set_ylabel('Observation z')
    axes[2].set_xlabel('Time step')
    axes[2].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

        # === 保存推断参数 ===
        save_dict = {
            'y_mean': y_mean.detach().cpu().numpy(),
            'y_cov' : y_cov.detach().cpu().numpy(),
            'mu'    : mu.detach().cpu().numpy(),
            'p'     : p.detach().cpu().numpy(),
            'r'     : r.detach().cpu().numpy(),
            'z_mean': z_mean.detach().cpu().numpy(),
            'z_low' : z_low.detach().cpu().numpy(),
            'z_high': z_high.detach().cpu().numpy(),
            'Z': Z.detach().cpu().numpy(),
            'mu_u': mu_u.detach().cpu().numpy(),
            'L_u': L_u.detach().cpu().numpy(),
            'pf_x_mean': pf_x_mean.detach().cpu().numpy(),
            'pf_x_low': pf_x_low.detach().cpu().numpy(),
            'pf_x_high': pf_x_high.detach().cpu().numpy(),
            'alpha' : alpha.detach().cpu().numpy(),
            'beta'  : beta.detach().cpu().numpy(),
            'theta_loc': theta_loc.detach().cpu().numpy(),
            'theta_cov': theta_cov.detach().cpu().numpy(),
            'Q': Q.detach().cpu().numpy(),
            'R': R.detach().cpu().numpy(),
            'G': G.detach().cpu().numpy(),
            'length_scale': length_scale.detach().cpu().numpy(),
            'variance': variance.detach().cpu().numpy()
        }

        np.savez(save_path.replace('.png', '_params.npz'), **save_dict)
        print(f"[Saved] model parameters saved to: {save_path.replace('.png', '_params.npz')}")
    plt.show()


def mean_fn(x_prev):
    # x_prev: [,dim_x]
    # print(x_prev.shape)
    # return x_prev @ A.T + B
    return x_prev


# 使用最小二乘从x_state估计A
def estimate_A_from_states(x_hist):
    # x_hist: (T, dim_x)
    T, dim_x = x_hist.shape
    if T < 2:
        return 0.9 * torch.eye(dim_x, device=x_hist.device, dtype=x_hist.dtype)
    X_t = x_hist[:-1]
    X_t1 = x_hist[1:]
    # 求解x_t @ A = X_t1的最小二乘
    A = torch.linalg.lstsq(X_t, X_t1).solution  # (dim_x, dim_x)
    return A


def generate_periodic_torch(T, period=34, amplitude=1.5, phase0=0.0, noise_std=0.1,
                            device='cpu', dtype=torch.float32, return_phase=False):
    """
    生成周期性序列，状态为相位 phi_t，满足 phi_t = (phi_{t-1} + omega) % 1，
    输出 x_t = amplitude * abs(sin(2*pi*phi_t + phase0)) + noise（保证是正半周期的 sin）。

    Args:
        T (int): 序列长度
        period (int): 周期长度（以时间步计）。默认 50。
        amplitude: 振幅 A
        phase0 (float): 初相位（弧度）
        noise_std (float): 高斯噪声标准差（添加到 x_t 上）
        device, dtype: torch tensor 类型
        return_phase (bool): 是否同时返回内部相位序列 phi

    """
    T = int(T)
    period = float(period)
    omega = 1.0 / period  # 每步相位增量，保证 period 步后回到原位

    phi = torch.empty(T, device=device, dtype=dtype)
    x = torch.empty(T, device=device, dtype=dtype)

    # 初始相位随机或固定为 0
    phi[0] = (phase0 / (2 * math.pi)) % 1.0  # 将 phase0(弧度) 转为 [0,1) 的相位
    x[0] = amplitude * torch.abs(torch.sin(2 * math.pi * phi[0] + phase0))  # 使用绝对值，确保是正的
    if noise_std > 0:
        x[0] = x[0] + noise_std * torch.randn(1, device=device, dtype=dtype)

    for t in range(1, T):
        # 纯依赖以前相位的映射 -> phi_t = f(phi_{t-1})
        phi[t] = (phi[t - 1] + omega) % 1.0
        x[t] = amplitude * torch.abs(torch.sin(2 * math.pi * phi[t] + phase0))  # 使用绝对值，确保是正的
        if noise_std > 0:
            x[t] = x[t] + noise_std * torch.randn(1, device=device, dtype=dtype)

    if return_phase:
        return x, phi
    return x


def compute_H_Psi(x_prev_b, Z, K_uu_chol, length_scale, variance):  # x_prev_b: [N,d]
    K_xu_b = kernel_xu_batch(x_prev_b, Z, lengthscale=length_scale, variance=variance)          # [N, M]
    H_b = torch.cholesky_solve(K_xu_b.transpose(-1,-2), K_uu_chol).transpose(-1, -2)                    # [N, M]
    K_xx_b = kernel_xu_batch(x_prev_b, x_prev_b, lengthscale=length_scale, variance=variance)   # [N, N]
    K_uu_inv = torch.cholesky_inverse(K_uu_chol)
    Kxu_uu_ux = ((K_xu_b @ K_uu_inv) @ K_xu_b.T)
    Psi_b = K_xx_b - Kxu_uu_ux
    Psi_b_diag = torch.diag(Psi_b)

    return H_b, Psi_b_diag  # [N, M], [N,N]


def build_cache(particles, inducing_points, length_scale, variance):
    """
    particles: [T, N, Dx]
    inducing_points: [M, Dx]
    """

    T, N, Dx = particles.shape

    cache = {}

    # ---- 1. compute K_uu once ----
    K_uu = kernel_uu(Z=inducing_points, lengthscale=length_scale, variance=variance)
    L_uu = torch.linalg.cholesky(K_uu)
    cache["K_uu"] = K_uu
    cache["L_uu"] = L_uu

    # ---- 2. precompute GP-related terms for each t ----
    cache["Psi"] = []
    cache["H_diag"] = []

    for t in range(T):
        psi_t, H_diag_t = compute_H_Psi(
            particles[t],          # [N, Dx]
            inducing_points,       # [M, Dx]
            L_uu,
            length_scale,
            variance
        )
        cache["Psi"].append(psi_t)       # [N, M]
        cache["H_diag"].append(H_diag_t) # [N]

    return cache


def particle_filter(z_obs, dim_x, x_state, y_latent, num_inducing=24, num_particle=50, num_iterations=30,
                    num_y_samples=14, num_theta_samples=6, dtype=torch.float64, device=None):
    rmse_hist, elbo_hist = [], []
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' if torch.cuda.is_available() else

    print(device)

    z_obs = z_obs.to(device)
    time_step, dim_z = z_obs.shape
    # 引入inducing_points
    Z_init = init_inducing_points_from_history(X_hist=x_state, T=time_step, dim_x=dim_x, use_kmeans=False,
                                               M=num_inducing, device=device).to(dtype)
    # Z = torch.nn.Parameter(torch.randn(num_inducing, dim_x, device=device, dtype=dtype))
    Z = torch.nn.Parameter(Z_init.clone())
    # Z = Z_init

    length_scale = torch.nn.Parameter(torch.tensor(1.5, device=device, dtype=dtype))
    variance = torch.nn.Parameter(torch.tensor(1.5, device=device, dtype=dtype))

    log_Q_loc = torch.nn.Parameter(-0.9 * torch.ones(dim_x, device=device, dtype=dtype))
    log_R_loc = torch.nn.Parameter(-0.9 * torch.ones(dim_x, device=device, dtype=dtype))
    log_G_loc = torch.nn.Parameter(-0.9 * torch.ones(dim_z, device=device, dtype=dtype))

    mean_theta = torch.nn.Parameter(torch.tensor([0.2, 0.6, 0.2], device=device, dtype=dtype))
    sigma_theta = 0.01 * torch.eye(3, device=device, dtype=dtype)

    rho_nb = torch.nn.Parameter(torch.log(torch.expm1(torch.ones(dim_z, device=device, dtype=dtype) * 1.0)))
    alpha = torch.nn.Parameter(0.1 * torch.ones((dim_z, dim_x), device=device, dtype=dtype))
    beta = torch.nn.Parameter(0.2 * torch.ones(dim_z, device=device, dtype=dtype))

    # --- init beta from observed z magnitude (so mu starts at correct scale) ---
    with torch.no_grad():
        mean_z_per_dim = z_obs.to(device=device, dtype=dtype).mean(dim=0)
        mean_z_per_dim = torch.clamp(mean_z_per_dim, min=1e-6)
        beta_init = torch.log(mean_z_per_dim + 1e-6)
        beta.data.copy_(beta_init)
        print("Initialized beta (first 5):", beta.data[:min(5, beta.numel())])
    # -------------------------------------------------------------------------

    # 初始化自然参数
    A = 0.9 * torch.eye(dim_x, device=device, dtype=dtype)
    # 初始化u的均值
    mu_u = Z @ A.T
    # 初始化u的协方差
    jitter = 1e-5
    K_uu = kernel_uu(Z, length_scale, variance)
    K_uu = K_uu + jitter * torch.eye(K_uu.size(0), device=device, dtype=dtype)
    I_dx = torch.eye(dim_x, device=device, dtype=dtype)
    Sigma_u = torch.kron(I_dx, K_uu)
    Sigma_u = Sigma_u + jitter * torch.eye(Sigma_u.size(0), device=device, dtype=dtype)  # (dim_x*M, dim_x*M)
    # 由(mu_u, Sigma_u)转化为自然参数
    mu_vec = mu_u.T.contiguous().view(-1)  # (dim_x*M,)
    Sigma_chol = torch.linalg.cholesky(Sigma_u)  # 下三角 (dim_x*M, dim_x*M)
    Precision = torch.cholesky_inverse(Sigma_chol)  # (dim_x*M, dim_x*M)

    eta_1 = Precision @ mu_vec  # (dim_x*M, )
    eta_2 = -0.5 * Precision  # (dim_x*M, dim_x*M)
    eta_2 = 0.5 * (eta_2 + eta_2.T)

    # ---- optimizer ----
    opt_all = torch.optim.Adam(
        [length_scale, variance, log_Q_loc, log_R_loc, log_G_loc, rho_nb, alpha, beta, Z],
        lr=1e-2
    )

    # ---- 打包 parameter_list（注意：传引用，不要 .detach()）----
    parameter_list = {
        'Z': Z,
        'eta_1': eta_1, 'eta_2': eta_2,
        # 下面这些会在每个迭代前用当前值刷新:
        # 'mean_x_0': mean_x_init,
        'rho_nb': rho_nb, 'alpha': alpha, 'beta': beta,
        'length_scale': length_scale, 'variance': variance,
        'theta_loc': mean_theta, 'theta_scale': sigma_theta,
        'log_Q_loc': log_Q_loc, 'log_R_loc': log_R_loc, 'log_G_loc': log_G_loc,
        'Q': None, 'R': None
    }

    # ---------- 新增： SV SVI window 参数 ----------
    W = 17 # 窗口长度（可调，建议 200-1000）
    overlap = 0.0  # 窗口重叠比（0~1），例如 0.2 表示 20% overlap
    use_random_windows = False # 是否每次随机抽窗；False -> 按序列滑动
    # ------------------------------------------------

    num_windows = max(1, int(np.ceil((time_step - W) / (W * (1 - overlap)) + 1)))      # 一共有多少个窗口
    window_stride = int(round(W * (1 - overlap)))                                      # 滑动步长

    last_iter_cache = {}
    for it in range(num_iterations):
        start_time = time.time()
        # ---- 用当前参数构造 Q, R 并刷新 parameter_list ----
        Q = torch.diag(torch.exp(log_Q_loc))
        R = torch.diag(torch.exp(log_R_loc))
        G = torch.diag(torch.exp(log_G_loc))
        eta_1 = parameter_list['eta_1']
        eta_2 = parameter_list['eta_2']
        parameter_list['Q'] = Q
        parameter_list['R'] = R
        parameter_list['G'] = G

        freeze_dyn = (it < 20)  # 前20轮冻结动态/核/q(u)
        for p in [parameter_list['length_scale'], parameter_list['variance'],
                  parameter_list['log_Q_loc']]:
            p.requires_grad_(not freeze_dyn)

        # ---- 选择窗口索引 t0,t1 ----
        if use_random_windows:
            # 随机选择窗口起点，确保有足够长 W 或用最后窗口补齐
            max_start = max(0, time_step - W)
            t0 = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            # 顺序滑动窗口（按迭代号选择窗口）
            idx = it % num_windows
            t0 = min(idx * window_stride, max(0, time_step - W))
        t1 = min(t0 + W, time_step)  # 右开区间末端 index

        # 提取窗口观测
        z_obs_window = z_obs[t0:t1].contiguous()  # shape (W, dim_z)
        W_actual = z_obs_window.shape[0]

        # 计算 scaling 因子使得梯度对全序列无偏：scale = T / W_actual
        time_idx = torch.arange(t0 + 1, t1 + 1, device=device)  # 1-based
        scale_factor = float(time_step) / float(W_actual)

        # ---- 自然参数 eta_1, eta_2 的 Polyak 更新（q(u)）----
        with torch.no_grad():
            particles, est_log_w, pf_x_mean, K_uu, K_uu_chol, theta_list, ess, logZ_t_list, ancestors = \
                particle_filter_vectorized(
                    z_obs=z_obs,
                    parameter_list=parameter_list,
                    num_particles=num_particle,
                    num_y_samples=num_y_samples,
                    num_theta_samples=num_theta_samples,
                    device=device,
                    dtype=dtype,
                    mean_fn=None,
                )

            eta_1_new, eta_2_new = estimate_natural_parameters_vectorized(
                Q=Q, K_uu_chol=K_uu_chol, Z=Z, length_scale=length_scale, variance=variance, use_particles=True,
                time_idx=time_idx, scale_factor=scale_factor, particles=particles, prelogw=est_log_w, pf_x_mean=pf_x_mean, mean_fn=None
            )
            rho = 0.6 if it < 60 else 0.2 * (num_iterations - it) / num_iterations
            eta_1 = (1 - rho) * eta_1 + rho * eta_1_new
            eta_2 = (1 - rho) * eta_2 + rho * eta_2_new
            eta_2 = 0.5 * (eta_2 + eta_2.T)

            parameter_list['eta_1'] = eta_1
            parameter_list['eta_2'] = eta_2
            mu_u, L_u = natural_to_mean_cov(eta_1, eta_2)

            cach = build_cache(particles, inducing_points=Z, length_scale=parameter_list['length_scale'], variance=parameter_list['variance'])

        # ---- 参数更新 (ELBO 反传) ----
        opt = opt_all
        opt.zero_grad(set_to_none=True)
        loss, stats = elbo_grad(
            z_obs=z_obs,
            particles=particles, est_log_w=est_log_w,
            parameter_list=parameter_list,
            mean_fn=None, dtype=dtype,
            num_theta_samples=num_theta_samples,
            num_y_samples=num_y_samples,
            obs_model='VGPSSM_NB',
            time_idx=time_idx,
            scale_factor=scale_factor,  # 新增参数：在 elbo 内乘以 scale
            cache=cach
        )

        # 正则化
        # reg_Q = 1e-3 * (torch.exp(log_Q_loc).mean() - 0.5) ** 2
        # reg_ker = 1e-3 * (variance - 1.0) ** 2 + 1e-3 * (torch.log(length_scale) - math.log(2.0)) ** 2
        # reg_alpha = 1e-3 * (alpha ** 2).mean()
        # reg_beta  = 1e-4 * (beta ** 2).mean()
        # smooth_lambda = 1e-4
        # diffs = pf_x_mean[1:] - pf_x_mean[:-1]
        # smooth_reg = smooth_lambda * (diffs ** 2).mean()
        loss = loss # + smooth_reg + reg_alpha +reg_beta + reg_Q  # + (0.0 if it < 10 else reg_ker) + reg_alpha

        loss.backward()
        for name, p in [
            ('length_scale', length_scale), ('variance', variance),  ('Z', Z),
            ('log_Q_loc', log_Q_loc), ('log_R_loc', log_R_loc), ('log_G_loc', log_G_loc),
            ('rho_nb', rho_nb), ('alpha', alpha), ('beta', beta)
        ]:
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                has_nan = torch.isnan(p.grad).any().item()
                has_inf = torch.isinf(p.grad).any().item()
                print(f"{name}: grad_norm={grad_norm:.3e}, nan={has_nan}, inf={has_inf}")
            else:
                print(f"{name}: grad is None")
        torch.nn.utils.clip_grad_norm_(
            [length_scale, variance, log_Q_loc, log_R_loc, log_G_loc, rho_nb, alpha, Z, beta],  # Z, alpha, beta
            max_norm=20.0
        )
        opt.step()

        if it == num_iterations - 1:  # 最后一轮
            last_iter_cache['particles'] = particles
            last_iter_cache['pf_x_mean'] = pf_x_mean if pf_x_mean is not None else None
            last_iter_cache['est_log_w'] = est_log_w
            last_iter_cache['K_uu_chol'] = K_uu_chol
            last_iter_cache['K_uu'] = K_uu

        # 显式释放窗口中间变量的 GPU 内存并清理缓存
        del particles, est_log_w, pf_x_mean, K_uu_chol, K_uu, eta_1, eta_2, eta_1_new, eta_2_new
        torch.cuda.empty_cache()

        free_gpu_memory()

        end_time = time.time()

        # ---- RMSE ----
        # rmse = np.sqrt(np.mean((pf_x_mean_window_cpu - x_state[:].detach().cpu().numpy()) ** 2))
        # ---- ELBO ----
        # 记录/调试
        elbo = stats['elbo']
        E_log_pu = stats['E_log_pu']
        trace_term = stats['trace_term']
        obs_term = stats['obs_term']
        trans_term = stats['trans_term']
        print(f"[Iter {it}] ELBO={elbo:.2f}, Trace={trace_term:.2f}, Obs={obs_term:.2f}, E_log_pu={E_log_pu:.2f}, trans_term={trans_term:.2f}")  # RMSE={rmse:.3f},
        print('Running time: %d seconds' %(end_time - start_time))
        elbo_hist.append(elbo)

    # ---- 绘图 ----
    visualize_results_with_z(
        # pf_x_mean=pf_x_mean,
        pf_x_particles=last_iter_cache['particles'], x_state=x_state, z_obs=z_obs, y_latent=y_latent,
        Z=parameter_list['Z'], mu_u=mu_u, L_u=L_u, length_scale=length_scale, variance=variance, theta_list=theta_list,
        theta_loc=parameter_list['theta_loc'],
        theta_cov=parameter_list['theta_scale'], log_R_loc=parameter_list['log_R_loc'], log_G_loc=parameter_list['log_G_loc'],
        log_Q_loc=parameter_list['log_Q_loc'], alpha=parameter_list['alpha'], beta=parameter_list['beta'],
        rho_nb=parameter_list['rho_nb'], save_path='/root/autodl-tmp/HGPSSM_1/result/NN_30min_VGPSSM_result.png')


    rmse_hist = np.asarray(rmse_hist)
    elbo_hist = np.asarray(elbo_hist)

    iters = np.arange(1, num_iterations + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(iters, rmse_hist, '-o')
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("RMSE over iterations")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iters, elbo_hist, '-o')
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (proxy)")
    plt.title("ELBO proxy over iterations")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'Z': Z,
        'particle': particles,
        'est_log_w': est_log_w,
        'lengthscale': length_scale,
        'variance': variance,
        'r': rho_nb,
        'log_Q_loc': log_Q_loc,
        'log_R_loc': log_R_loc,
        'alpha': alpha,
        'beta': beta,
        "theta_sample": theta_list,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "mu_u": mu_u,
        "Sigma_u": Sigma_u,
        "pf_x_mean": pf_x_mean,
        "pyro_params": dict(pyro.get_param_store())
    }


# ------------------------
if __name__ == "__main__":
    # data = np.load('data/large_NB_synthetic_data.npz')
    nn_data = pd.read_csv('/root/autodl-tmp/HGPSSM_1/data/200501-0531-nanning-30min.csv', header=None)
    # zs = data['z_t']
    # xs = data['x_t']
    # ys = data['y_t']
    # theta = data['theta_t']
    # T, dim_z = zs.shape
    # _, dim_x = xs.shape
    # dim_u = dim_x
    T, dim_z = nn_data.shape
    dim_x = 24                                                           
    dim_u = dim_x

    xs = []
    for d in range(dim_x):
        # amplitude = torch.rand(1)
        x_t = generate_periodic_torch(T=T, period=34, amplitude=2.0)
        xs.append(x_t)

    xs = torch.stack(xs).transpose(1, 0)

    M = 340

    nn_data = np.array(nn_data)
    z_obs = torch.tensor(nn_data)
    x_his = xs
    y_latent = torch.randn(T, dim_x)
    # z_obs    = torch.tensor(zs)
    # x_his    = torch.tensor(xs)
    # y_latent = torch.tensor(ys)
    out = particle_filter(z_obs=z_obs, dim_x=dim_x, x_state=x_his, y_latent=y_latent, num_inducing=M,
                          num_particle=300, num_iterations=150, device=None)
    print("Finished. Keys:", out.keys())
    print("length_scale:", out['lengthscale'], "variance", out['variance'], "r:", out['r'])
