import torch
import torch.distributions as dist
import torch.nn.functional as F
import math
import numpy as np
from kernel_calculating import kernel_xx, kernel_xu, kernel_uu
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Define parameters
T = 480  # Number of time steps
N = 300 # Number of stations
D = 32  # Number of latent states


def generate_periodic_torch(T, period=24, amplitude=1.5, phase0=0.0, noise_std=0.01,
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


xs = []
for d in range(D):
    # amplitude = torch.rand(1)
    x_t = generate_periodic_torch(T=T, amplitude=1.5)
    xs.append(x_t)

xs = torch.stack(xs).transpose(1, 0)


def sample_f_t_given_history(x_prev, X_hist, f_hist, D, length_scale, variance, mean_fn, jitter=1e-6):
    """
    p(f_t | f_{1:t-1}, x_{0:t-1}) = N( m(x_{t-1}) + K_* K^{-1} (f_hist - m(X_hist)), K_xx - K_* K^{-1} K_*^T )
    :param x_prev: [D],即x_{t-1}
    :param X_hist: [t, D] x_0, x_1, ..., x_t-1 (t=1时可以为None)
    :param f_hist: [t-1, D] f_1, f_2, ..., f_t-1 (t=1时为None)
    :return: f_t, mean, conv
    """
    device = x_prev.device
    x_prev = x_prev.reshape(-1)  # [1, D]
    if mean_fn is None:
        mean_fn = lambda X: torch.zeros(X.size(), device=device, dtype=X.dtype)
    m_star = mean_fn(x_prev)  # [D]
    I_d = torch.eye(D, device=device, dtype=x_prev.dtype)

    # 先验，无历史
    if X_hist is None or f_hist is None:
        K_xx = kernel_xx(x_prev, lengthscale=length_scale, variance=variance)  # [1, 1]
        K_xx_batch = torch.kron(I_d, K_xx)  # [D, D]
        K_xx_batch = K_xx_batch + jitter * torch.eye(D, device=device, dtype=x_prev.dtype)
        # 轻微裁剪，防负特征值
        min_eig = torch.linalg.eigvalsh(K_xx).min().item()
        if min_eig < 1e-12:
            K_xx_batch = K_xx_batch + (1e-12 - min_eig) * torch.eye(D, dtype=K_xx.dtype, device=K_xx.device)
        return dist.MultivariateNormal(m_star, K_xx_batch).sample(), m_star, K_xx_batch

    K_hh = kernel_uu(X_hist, lengthscale=length_scale, variance=variance)  # [t-1, t-1]
    K_hh_batch = torch.kron(I_d, K_hh)  # [D*(t-1), D*(t-1)]
    K_hh_batch = K_hh_batch + jitter * torch.eye(K_hh_batch.shape[0], device=device, dtype=x_prev.dtype)
    K_star = kernel_xu(x_prev, X_hist, lengthscale=length_scale, variance=variance)  # [1, t-1]
    K_star_batch = torch.kron(I_d, K_star)  # [D, D*(t-1)]
    K_xx = kernel_xx(x_prev, lengthscale=length_scale, variance=variance)  # [1, 1]
    K_xx_batch = torch.kron(I_d, K_xx)  # [D, D]
    K_xx_batch = K_xx_batch + jitter * torch.eye(K_xx_batch.shape[0], device=device, dtype=x_prev.dtype)

    m_hist = mean_fn(X_hist)  # [t-1, D]
    resid = (f_hist - m_hist).T.reshape(-1)  # [D*(t-1)]

    L = torch.linalg.cholesky(K_hh_batch)
    Khh_inv_res = torch.cholesky_solve(resid.unsqueeze(-1), L)  # [D*(t-1),]
    mean = m_star + (K_star_batch @ Khh_inv_res).squeeze(-1)  # [D]

    V = torch.cholesky_solve(K_star_batch.T, L)  # [D*(t-1), D]
    cov = K_xx_batch - K_star_batch @ V  # [D, D]
    cov = 0.5 * (cov + cov.T)  # 对称化
    eigmin = torch.linalg.eigvalsh(cov).min().clamp(max=0).item()
    if eigmin < 1e-10:
        cov = cov + (1e-6 - eigmin) * torch.eye(D, device=device, dtype=x_prev.dtype)

    return dist.MultivariateNormal(mean, cov).sample(), mean, cov


length_scale = torch.tensor(2.0)
variance = torch.tensor(1.0)

log_Q_loc = -1.6 * torch.ones(D)
log_R_loc = -1.6 * torch.ones(D)
# log_G_loc = -3 * torch.ones(N)
Q = torch.diag(torch.exp(log_Q_loc))
R = torch.diag(torch.exp(log_R_loc))
# G = torch.diag(torch.exp(log_G_loc))
rho_nb = torch.log(torch.expm1(torch.ones(N) * 8.0))
r = torch.nn.functional.softplus(rho_nb)
alpha = 0.8 * torch.ones((N, D))
beta = 0.6 * torch.ones(N)

# 变量列表
# xs = []
thetas = []
ys = []
zs = []
# 初始状态
x_init = dist.MultivariateNormal(torch.zeros(D), covariance_matrix=0.01 * torch.eye(D)).sample()
mean_theta = torch.tensor([0.1, 0.8, 0.1])
sigma_theta = 0.01 * torch.eye(3)
theta_prev = dist.MultivariateNormal(mean_theta, covariance_matrix=sigma_theta).sample()

# xs.append(x_init)
thetas.append(theta_prev)


def Spatial_effect_matrix(theta_t):
    # 构建空间效应矩阵，返回一个D_y * D_x的矩阵
    tri_matrix = torch.zeros((D, D))
    diagonal = theta_t[1]  # 主对角线
    lower_diagonal = theta_t[0]  # 下对角线
    upper_diagonal = theta_t[2]  # 上对角线
    # 填充三对角矩阵
    tri_matrix.diagonal().copy_(diagonal)
    tri_matrix.diagonal(offset=-1).copy_(lower_diagonal)
    tri_matrix.diagonal(offset=1).copy_(upper_diagonal)

    return tri_matrix


# GP历史状态缓存
X_hist = None
F_hist = None

# 均值函数
# mean_fn = lambda X: torch.zeros(X.size(), device=X.device, dtype=X.dtype)
mean_fn = lambda X: X.clone()

for t in range(1, T + 1):
    # compute kernel matrix between x_{t-1} and itself
    # k_xx = kernel_xx(xs[t-1], lengthscale=length_scale, variance=variance)
    # f_t = dist.MultivariateNormal(torch.zeros(D), k_xx + 1e-6 * torch.eye(D)).sample()
    # f_t, _, _ = sample_f_t_given_history(x_prev=xs[t-1], X_hist=X_hist, f_hist=F_hist, D=D, length_scale=length_scale,
    #                                      variance=variance, mean_fn=mean_fn)
    # # 更新 GP 历史，以便下一步用到 x_{0:t-2}, f_{1:t-1}
    # X_hist = xs[t - 1].unsqueeze(0) if X_hist is None else torch.cat([X_hist, xs[t - 1].unsqueeze(0)], dim=0)
    # F_hist = f_t.unsqueeze(0) if F_hist is None else torch.cat([F_hist, f_t.unsqueeze(0)], dim=0)
    # # # 计算 x_t | f_t ~ N(f_t, lambda_x)
    # x_t = dist.MultivariateNormal(f_t, covariance_matrix=Q).sample()
    # xs.append(x_t)

    # ---- 参数theta演化 ----
    # 简单马尔可夫演化：θ_t | θ_{t-1} ~ N(θ_{t-1}, Σθ)
    theta_t = dist.MultivariateNormal(mean_theta, covariance_matrix=sigma_theta).sample()
    thetas.append(theta_t)
    # Step 3: Generate observation variable y_t
    S_gamma_t = Spatial_effect_matrix(theta_t)  # Time-varying S(gamma_T)
    mean_y_t = (xs[t - 1] @ S_gamma_t).squeeze(0)  # use tridiagonal matrix S
    y_t = dist.MultivariateNormal(mean_y_t, covariance_matrix=R).sample()
    ys.append(y_t)

    # Step 4: Generate Negative observations z_t
    logits = alpha @ y_t + beta
    mu_prob = F.softplus(logits) + 1e-6
    p_probe = mu_prob / (mu_prob + r)
    # probs = torch.sigmoid(logits)
    # mean = alpha @ y_t + beta
    # NB分布 (参数化为 total_count=r, probs=p)
    z_t = dist.NegativeBinomial(total_count=r, probs=p_probe).sample()
    # z_t = dist.MultivariateNormal(mean, covariance_matrix=G).sample()
    zs.append(z_t)
    theta_prev = theta_t

# x_sample = torch.stack(xs)  # shape: [T+1, D]
x_sample = xs
theta_sample = torch.stack(thetas)  # shape: [T+1, 3]
y_sample = torch.stack(ys)  # shape: [T, D]
z_obs = torch.stack(zs)  # shape: [T, N]

# z_t = np.mean(sampled_z, axis=0)
# savemat("sample_x.mat", {"x": np.array(x_sample)})
# savemat('sample_theta.mat', {"theta": np.array(theta_sample)})
# savemat("sample_y.mat", {"y": np.array(y_sample)})
# savemat("sample_z.mat", {"z": np.array(z_obs)})

# z_t: (time_steps, observation_dimension); y_t: (time_steps, latent_dimension)
plt.figure(figsize=(12, 6))
# plt.plot(range(T), x_sample[:, 2], label=f'Observed (station {5})', linestyle='dotted')
plt.plot(range(T), z_obs[:, 2], label=f'Observed (station {5})', linestyle='dotted')
plt.show()

x_sample = x_sample.detach().cpu().numpy()
theta_sample = theta_sample.detach().cpu().numpy()
y_sample = y_sample.detach().cpu().numpy()
z_sample = z_obs.detach().cpu().numpy()
length_scale = np.array(length_scale)
variance = np.array(variance)
log_Q_loc = np.array(log_Q_loc)
log_R_loc = np.array(log_R_loc)
mean_theta = mean_theta.detach().cpu().numpy()
sigma_theta = sigma_theta.detach().cpu().numpy()
rho_nb = np.array(rho_nb)
alpha = alpha.detach().cpu().numpy()
beta = beta.detach().cpu().numpy()

# Save parameters and generated data
np.savez("data/large_NB_synthetic_data.npz", x_t=x_sample, theta_t=theta_sample, y_t=y_sample,
         z_t=z_obs,
         length_scale=length_scale, variance=variance, mean_theta=mean_theta, sigma_theta=sigma_theta, Q=log_Q_loc,
         R=log_R_loc, rho_nb=rho_nb, beta=beta, alpha=alpha)
