import torch.linalg
import torch
import pyro
import pyro.distributions as dist
import math
# import numpy as np
# from scipy.spatial.distance import cdist


def init_inducing_points_from_history(X_hist, T, dim_x, M=100, use_kmeans=False, noise_scale=0.1, device='cpu'):
    """
    返回 Z: (M, D_in), D_in = D_x 或 D_x + D_u
    三选一初始化：
      - 从历史中随机采样 + 高斯扰动（默认）
      - 从历史 kmeans 质心（如果 use_kmeans=True）
      - 纯高斯随机 (如果 X_hist is None)
    """
    if X_hist is not None:
        XY = X_hist
    else:
        XY = torch.randn(T,dim_x)

    N = XY.shape[0]
    D_in = XY.shape[1]

    if use_kmeans:
        # 简单 kmeans (如果sklearn不可用，可用随机采样退回)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=M, n_init=5)
            km.fit(XY.cpu().numpy())
            Z = torch.tensor(km.cluster_centers_, dtype=XY.dtype, device=device)
        except Exception:
            idx = torch.randperm(N)[:M]
            Z = XY[idx].clone().to(device)
    else:
        # 随机从历史中挑 M 个并加噪声（Matlab 的做法）
        idx = torch.randperm(N)[:M]
        Z_hist = XY[idx].clone().to(device)
        # noise = noise_scale * torch.randn_like(Z_hist)
        Z = Z_hist
        # 如果样本数 < M，可补齐若干随机高斯
        if Z.shape[0] < M:
            extra = torch.randn(M - Z.shape[0], D_in, device=device) * XY.std(dim=0).mean()
            Z = torch.cat([Z, extra], dim=0)

    return Z  # shape (M, D_in)

def init_parameters(dim_x, dim_u, dim_z, num_inducing, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化inducing_inputs: Z
    init_Z = torch.randn(num_inducing, dim_u, device=device)
    pyro.param("Z", init_Z)  # inducing inputs will be optimized by SVI
    # log diagonal Q/R
    pyro.param("log_Q_loc", -0.7 * torch.ones(dim_x, device=device))
    pyro.param("log_R_loc", -0.7 * torch.ones(dim_x, device=device))
    pyro.param('r', 2 * torch.ones(dim_z, device=device))  # 初值 ~ r≈2
    # 关于初始p(\theta_0)的参数
    # pyro.param('mean_theta', torch.tensor([0.1, 0.8, 0.1], device=device))
    # pyro.param('sigma_theta', 0.0001 * torch.eye(3, device=device), constraint=dist.constraints.positive_definite)
    # inducing_points的自然参数
    eta_1_init = torch.zeros(num_inducing * dim_u, device=device)
    eta_2_init = -0.5 * torch.eye(num_inducing * dim_u, device=device)

    return {
        "eta_1": eta_1_init,
        "eta_2": eta_2_init
    }

def periodic_kernel(X1, X2, lengthscale, variance, period):
    """
    Periodic kernel function (RBF + periodic part)
    X1: (n1, D)
    X2: (n2, D)
    lengthscale: float
    variance: float
    period: float, period of the sine function
    """
    x1 = X1.unsqueeze(1)  # (n1, 1, D)
    x2 = X2.unsqueeze(0)  # (1, n2, D)
    dist = torch.sqrt(((x1 - x2) ** 2).sum(-1))  # Euclidean distance

    # Periodic kernel
    kernel = torch.exp(-2.0 * (torch.sin(math.pi * dist / period) ** 2) / lengthscale**2)
    return variance * kernel


def rbf_kernel_matrix(X1, X2, lengthscale, variance):
    # 支持每个维度不同的 lengthscale
    # X1: (n1,D), X2: (n2,D)
    x1 = X1.unsqueeze(1)  # (n1,1,D)
    x2 = X2.unsqueeze(0)  # (1,n2,D)

    # x1 = x1 / lengthscale
    # x2 = x2 / lengthscale

    # x1 = X1_scaled.unsqueeze(1)   # (n1,1,D)
    # x2 = X2_scaled.unsqueeze(0)   # (1,n2,D)

    sq = ((x1 - x2) ** 2).sum(-1)  # (n1,n2)
    sq = sq / lengthscale
    return variance * torch.exp(-0.5 * sq)


# def matern_kernel_matrix(X1, X2, lengthscale, variance, nu=1.5):
#     """
#     Matern Kernel with ARD (different lengthscale per dimension).
#     X1: (n1, D)
#     X2: (n2, D)
#     lengthscale: (D,) tensor or float
#     variance: float or scalar tensor
#     nu: 1.5 or 2.5
#     """
#     # 确保 lengthscale 是 tensor，支持 broadcast
#     if not torch.is_tensor(lengthscale):
#         lengthscale = torch.tensor(lengthscale, device=X1.device, dtype=X1.dtype)
#
#     # 提前做缩放，避免在 view 上直接除法
#     ls = lengthscale.clone()
#     X1_scaled = (X1 / ls)  # (n1,D)
#     X2_scaled = (X2 / ls)  # (n2,D)
#
#     x1 = X1_scaled.unsqueeze(1)    # (n1,1,D)
#     x2 = X2_scaled.unsqueeze(0)    # (1,n2,D)
#
#     dist = torch.sqrt(((x1 - x2) ** 2).sum(-1) + 1e-12)  # (n1,n2)
#
#     if nu == 1.5:
#         sqrt3 = torch.sqrt(torch.tensor(3.0, device=X1.device, dtype=X1.dtype))
#         scaled_dist = sqrt3 * dist
#         K = (1.0 + scaled_dist) * torch.exp(-scaled_dist)
#
#     elif nu == 2.5:
#         sqrt5 = torch.sqrt(torch.tensor(5.0, device=X1.device, dtype=X1.dtype))
#         scaled_dist = sqrt5 * dist
#         K = (1.0 + scaled_dist + (scaled_dist ** 2) / 3.0) * torch.exp(-scaled_dist)
#
#     else:
#         raise ValueError("Only nu=1.5 or nu=2.5 are supported")
#
#     return variance * K


def matern_kernel_matrix(X1, X2, lengthscale, variance, nu=1.5):
    """
    Matern Kernel with ARD (different lengthscale per dimension).
    X1: (n1, D)
    X2: (n2, D)
    lengthscale: (D,) tensor or float
    variance: float or scalar tensor
    nu: 1.5 or 2.5
    """
    x1 = X1.unsqueeze(1)    # (n1,1,D)
    x2 = X2.unsqueeze(0)    # (1,n2,D)

    dist = torch.sqrt(((x1 - x2) ** 2).sum(-1) + 1e-12)  # (n1,n2)

    if nu == 1.5:
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=X1.device, dtype=X1.dtype))
        scaled_dist = sqrt3 * dist / lengthscale
        K = (1.0 + scaled_dist) * torch.exp(-scaled_dist)

    elif nu == 2.5:
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=X1.device, dtype=X1.dtype))
        scaled_dist = sqrt5 * dist / lengthscale
        K = (1.0 + scaled_dist + (scaled_dist ** 2) / 3.0) * torch.exp(-scaled_dist)

    else:
        raise ValueError("Only nu=1.5 or nu=2.5 are supported")

    return variance * K

def kernel_uu(Z, lengthscale, variance, jitter=1e-6):
    M, dim_u = Z.shape
    K_Z = matern_kernel_matrix(Z, Z, lengthscale, variance)  #  shape: [M, M]
    # K_Z = 0.5 * (K_Z + K_Z.T)
    # 这样每个输出维度共享相同的核相关性，但不同输出维度独立。
    # K_block = torch.kron(torch.eye(dim_u, device=Z.device), K_Z)
    K_uu = K_Z + jitter * torch.eye(M, device=Z.device)
    return K_uu  # shape: [M, M]

def kernel_xx(x_t, lengthscale, variance):
    # x_t :shape: [dim_x,]
    K_xx = matern_kernel_matrix(x_t.unsqueeze(0), x_t.unsqueeze(0), lengthscale, variance)  #, period=12 [1, 1]
    # K_block = torch.kron(torch.eye(dim_x, device=x_t.device), K_xx)
    return K_xx  # [1,]

def kernel_xu(x_t, Z, lengthscale, variance):
    """
    Compute K_{x_t, u} with shape [1*dim_u, M*dim_u] using Kronecker product.
    """
    M, dim_u = Z.shape
    # 核矩阵 K_xz: [1, M]
    K_xu = matern_kernel_matrix(x_t.unsqueeze(0), Z, lengthscale, variance)  #, period=12 [1, M]
    # Kronecker 积: ( [1, M] ⊗ I_dim_u ) -> [dim_u, M*dim_u]
    # K_xu = torch.kron(torch.eye(dim_u, device=Z.device), K_xz)
    # 最后 reshape 成 [1*dim_u, M*dim_u]
    return K_xu


def kernel_xu_batch(x_batch, Z, lengthscale, variance, chunk=20000):
    TN = x_batch.shape[0]
    M = Z.shape[0]

    K_xu = torch.empty(TN, M, device=x_batch.device, dtype=x_batch.dtype)

    for start in range(0, TN, chunk):
        end = min(start + chunk, TN)
        K_xu[start:end] = matern_kernel_matrix(
            x_batch[start:end], Z, lengthscale, variance
        )
        torch.cuda.empty_cache()  # 防止碎片继续膨胀

    # M, dim_u = Z.shape
    # # 核矩阵 K_xz: [1, M]
    # K_xu = matern_kernel_matrix(x_batch, Z, lengthscale, variance)  #, period=12 [1, M]
    # Kronecker 积: ( [1, M] ⊗ I_dim_u ) -> [dim_u, M*dim_u]
    # K_xu = torch.kron(torch.eye(dim_u, device=Z.device), K_xz)
    # 最后 reshape 成 [1*dim_u, M*dim_u]
    return K_xu

def kernel_xx_batch(x_batch, lengthscale, variance):
    """
    x_batch: [N, dim_x]
    返回:    [N, dim_x, dim_x]
    """
    return torch.vmap(lambda x: kernel_xx(x, lengthscale, variance))(x_batch)


def cholesky_solve_apply(L, B):
    # given L = cholesky(A), solve A^{-1} B via cholesky_solve
    # B can be vector or matrix (last dim is features)
    # torch.cholesky_solve expects RHS with shape (..., n, k) and L lower-triangular
    return torch.cholesky_solve(B.unsqueeze(-1), L).squeeze(-1) if B.dim()==1 else torch.cholesky_solve(B, L)

def gp_predictive_mean(x_prev, Z, L_uu, u_vec, lengthscale, variance):
    K_xu = kernel_xu(x_prev, Z, lengthscale=lengthscale, variance=variance)  # shape: [1*dim_u, M*dim_u]
    # f_mean = (K_xu @ (K_uu_inv @ u_vec)).view(-1)  # shape: [dim_u, ]; dim_u = dim_x
    Ku_inv_u = cholesky_solve_apply(L_uu, u_vec)
    # predictive mean f_mean = K_xu @ Ku_inv_u  (shapes must align)
    f_mean = K_xu @ Ku_inv_u
    return f_mean, K_xu

def ensure_pd_by_eig(mat, min_eig=1e-6):
    # 对称化 + 特征值截断，返回 PD 矩阵（在 CPU/GPU 都可用）
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals_clamped = torch.clamp(eigvals, min=min_eig)
    return (eigvecs * eigvals_clamped) @ eigvecs.T