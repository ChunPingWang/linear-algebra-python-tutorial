"""SVD：由 A^T A 的特徵分解建構、低秩近似、PCA。

對應 Strang 第 7 章與第 10.3 節，以及 Riley 8.18 的條件數討論。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "svd_from_eigen",
    "svd_pieces",
    "low_rank_approx",
    "frobenius_error",
    "pca",
    "condition_number",
    "polar_decomposition",
]


def svd_from_eigen(A, tol: float = 1e-12):
    """從 A^T A = V Σ² V^T 出發，親手把 SVD 組出來。

    流程：
    1. A^T A 對稱半正定 → eigh 得到 V 與 λ ≥ 0
    2. σ = √λ（由大到小）
    3. σ > 0 的部分用 u = Av/σ 得到左奇異向量
    4. 其餘的 u 用正交補空間補滿（full_matrices 版）
    回傳 (U, s, Vt)，使 A = U diag(s) V^T。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    lam, V = np.linalg.eigh(A.T @ A)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]
    s = np.sqrt(lam)
    U = np.zeros((m, min(m, n)))
    k = 0
    for j in range(len(s)):
        if k >= U.shape[1]:
            break
        if s[j] > tol * (s[0] if s[0] > 0 else 1.0):
            U[:, k] = A @ V[:, j] / s[j]
            k += 1
    if k < U.shape[1]:                        # 補滿左側正交基底
        Q, _ = np.linalg.qr(np.hstack([U[:, :k], np.eye(m)]))
        U[:, k:] = Q[:, k:U.shape[1]]
    r = min(m, n)
    # 回傳精簡型：U 是 m×r、s 有 r 個、Vt 是 r×n（m < n 時也正確）
    return U, s[:r], V[:, :r].T


def svd_pieces(A):
    """把 A 拆成 σ_k u_k v_k^T 的秩一積木清單（能量由大到小）。"""
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return [(float(s[k]), np.outer(U[:, k], Vt[k, :])) for k in range(len(s))]


def low_rank_approx(A, k: int):
    """Eckart–Young：保留前 k 個奇異值得到最佳秩 k 近似。"""
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(min(k, len(s)))
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def frobenius_error(A, B) -> float:
    """‖A - B‖_F，衡量近似好壞。"""
    return float(np.linalg.norm(np.asarray(A, dtype=float) - np.asarray(B, dtype=float), "fro"))


def pca(X, n_components: int = 2, center: bool = True):
    """主成分分析 = 置中資料矩陣的 SVD（Strang 7.3）。

    回傳 dict：components（主軸，列向量）、scores（投影座標）、
    explained_variance、explained_ratio、singular_values、mean。
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2 / max(n_samples - 1, 1)
    k = int(min(n_components, len(s)))
    return {
        "components": Vt[:k, :],
        "scores": U[:, :k] * s[:k],
        "singular_values": s,
        "explained_variance": var[:k],
        "explained_ratio": (var / var.sum())[:k] if var.sum() > 0 else var[:k],
        "mean": mean,
        "covariance": Xc.T @ Xc / max(n_samples - 1, 1),
    }


def condition_number(A) -> float:
    """條件數 σ_max / σ_min：解 Ax = b 時誤差可能被放大的倍數。"""
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else np.inf


def polar_decomposition(A):
    """極分解 A = QS：Q 正交（旋轉/反射）、S 對稱半正定（拉伸）。"""
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A)
    Q = U @ Vt
    S = Vt.T @ np.diag(s) @ Vt
    return Q, S
