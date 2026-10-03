"""特徵值與特徵向量：冪法、反冪法、QR 演算法、Jacobi、對角化、矩陣指數。

對應 Strang 第 6 章、附錄 5（Jordan 形式）、附錄 8（Markov 矩陣），
以及 Riley 8.13–8.17、9（法模態）、27（數值方法）。
"""

from __future__ import annotations

import numpy as np

from .orthogonal import householder_qr

__all__ = [
    "characteristic_polynomial",
    "eigen_2x2",
    "power_iteration",
    "inverse_power_iteration",
    "rayleigh_quotient",
    "qr_algorithm",
    "jacobi_eigen",
    "diagonalize",
    "matrix_power_by_eigen",
    "matrix_exp_by_eigen",
    "matrix_exp_series",
    "is_symmetric",
    "is_positive_definite",
    "generalized_symmetric_eig",
    "spectral_decomposition",
    "markov_steady_state",
]


# --------------------------------------------------------------------------
# 定義層面
# --------------------------------------------------------------------------
def characteristic_polynomial(A):
    """用 sympy 求特徵多項式 det(A - λI) 的符號式與根（適合小矩陣手算對照）。"""
    import sympy as sp

    M = sp.Matrix(np.asarray(A, dtype=float).tolist())
    lam = sp.symbols("lamda")
    poly = sp.factor(sp.expand((M - lam * sp.eye(M.shape[0])).det()))
    roots = sp.solve(sp.Eq(poly, 0), lam)
    return poly, roots


def eigen_2x2(A) -> tuple[np.ndarray, np.ndarray]:
    """2x2 的封閉解：λ = (跡 ± √(跡² - 4·det)) / 2。"""
    A = np.asarray(A, dtype=float)
    t, d = np.trace(A), A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = t * t - 4 * d
    sq = np.sqrt(complex(disc))
    lams = np.array([(t + sq) / 2, (t - sq) / 2])
    vecs = []
    for lam in lams:
        B = A.astype(complex) - lam * np.eye(2)
        v = np.array([B[0, 1], -B[0, 0]]) if abs(B[0, 1]) > 1e-12 else np.array([B[1, 1], -B[1, 0]])
        if np.linalg.norm(v) < 1e-12:
            v = np.array([1.0, 0.0])
        vecs.append(v / np.linalg.norm(v))
    return lams, np.array(vecs).T


def rayleigh_quotient(A, x) -> float:
    """Rayleigh 商 x^T A x / x^T x：對稱矩陣特徵值的變分刻畫。"""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    return float(x @ A @ x / (x @ x))


# --------------------------------------------------------------------------
# 迭代法
# --------------------------------------------------------------------------
def power_iteration(A, iters: int = 200, tol: float = 1e-12, x0=None):
    """冪法：反覆做 x ← Ax/‖Ax‖，收斂到最大 |λ| 的特徵向量。

    回傳 (λ, v, 收斂歷程)。這正是 Google PageRank 與 Markov 鏈的算法核心。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, dtype=float).ravel()
    x = x / np.linalg.norm(x)
    history = []
    lam = rayleigh_quotient(A, x)
    for _ in range(iters):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-300:
            return 0.0, x, history
        x_new = y / ny
        lam_new = rayleigh_quotient(A, x_new)
        history.append(lam_new)
        if abs(lam_new - lam) < tol:
            x, lam = x_new, lam_new
            break
        x, lam = x_new, lam_new
    return lam, x, history


def inverse_power_iteration(A, shift: float = 0.0, iters: int = 200, tol: float = 1e-12):
    """反冪法（帶位移）：對 (A - σI)^{-1} 做冪法，收斂到最接近 σ 的特徵值。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    B = A - shift * np.eye(n)
    x = np.ones(n) / np.sqrt(n)
    lam = shift
    for _ in range(iters):
        y = np.linalg.solve(B, x)
        x_new = y / np.linalg.norm(y)
        lam_new = rayleigh_quotient(A, x_new)
        if abs(lam_new - lam) < tol:
            return lam_new, x_new
        x, lam = x_new, lam_new
    return lam, x


def qr_algorithm(A, iters: int = 500, tol: float = 1e-12):
    """未加位移的 QR 演算法：A_{k+1} = R_k Q_k，對角線收斂到特徵值。

    每一步 A_k = Q_k R_k 是相似變換 A_{k+1} = Q_k^T A_k Q_k，特徵值不變。
    回傳 (特徵值陣列, 累積的 Q)。
    """
    A_k = np.array(A, dtype=float, copy=True)
    n = A_k.shape[0]
    Q_total = np.eye(n)
    for _ in range(iters):
        Q, R = householder_qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q
        off = np.sum(np.abs(np.tril(A_k, -1)))
        if off < tol:
            break
    return np.diag(A_k).copy(), Q_total


def jacobi_eigen(A, iters: int = 100, tol: float = 1e-12):
    """對稱矩陣的 Jacobi 旋轉法：用一連串 2x2 旋轉把非對角元素消成 0。

    回傳 (特徵值, 特徵向量矩陣 V)，且 A = V diag(λ) V^T。
    """
    A_k = np.array(A, dtype=float, copy=True)
    n = A_k.shape[0]
    if not np.allclose(A_k, A_k.T):
        raise ValueError("Jacobi 法只適用於對稱矩陣")
    V = np.eye(n)
    for _ in range(iters):
        off = np.abs(A_k - np.diag(np.diag(A_k)))
        i, j = np.unravel_index(np.argmax(off), off.shape)
        if off[i, j] < tol:
            break
        if A_k[i, i] == A_k[j, j]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * A_k[i, j], A_k[i, i] - A_k[j, j])
        c, s = np.cos(theta), np.sin(theta)
        G = np.eye(n)
        G[i, i], G[j, j], G[i, j], G[j, i] = c, c, -s, s
        A_k = G.T @ A_k @ G
        V = V @ G
    idx = np.argsort(np.diag(A_k))[::-1]
    return np.diag(A_k)[idx].copy(), V[:, idx]


# --------------------------------------------------------------------------
# 對角化與函數
# --------------------------------------------------------------------------
def is_symmetric(A, tol: float = 1e-10) -> bool:
    A = np.asarray(A)
    return bool(A.shape[0] == A.shape[1] and np.allclose(A, A.T, atol=tol))


def is_positive_definite(A, tol: float = 1e-12) -> bool:
    """對稱 + 所有特徵值 > 0（等價於所有主子式 > 0、存在 A = R^T R）。"""
    if not is_symmetric(A, 1e-8):
        return False
    return bool(np.all(np.linalg.eigvalsh(np.asarray(A, dtype=float)) > tol))


def diagonalize(A):
    """A = X Λ X^{-1}；若特徵向量不足（不可對角化）則回傳 None。"""
    A = np.asarray(A, dtype=float)
    lam, X = np.linalg.eig(A)
    if np.linalg.matrix_rank(X, tol=1e-8) < A.shape[0]:
        return None
    return lam, X, np.linalg.inv(X)


def spectral_decomposition(A):
    """對稱矩陣的譜分解 A = Σ λ_k q_k q_k^T（秩一投影的加權和）。"""
    A = np.asarray(A, dtype=float)
    lam, Q = np.linalg.eigh(A)
    pieces = [(lam[k], np.outer(Q[:, k], Q[:, k])) for k in range(len(lam))]
    return lam, Q, pieces


def matrix_power_by_eigen(A, k: int):
    """A^k = X Λ^k X^{-1}：特徵值取 k 次方即可。"""
    out = diagonalize(A)
    if out is None:
        return np.linalg.matrix_power(np.asarray(A, dtype=float), k)
    lam, X, Xinv = out
    return np.real_if_close(X @ np.diag(lam ** k) @ Xinv)


def matrix_exp_by_eigen(A, t: float = 1.0):
    """e^{At} = X e^{Λt} X^{-1}：解 du/dt = Au 的關鍵（Strang 6.5）。"""
    out = diagonalize(A)
    if out is None:
        return matrix_exp_series(A, t)
    lam, X, Xinv = out
    return np.real_if_close(X @ np.diag(np.exp(lam * t)) @ Xinv)


def matrix_exp_series(A, t: float = 1.0, terms: int = 60):
    """用冪級數 e^{At} = Σ (At)^k / k! 計算（不需要可對角化）。"""
    A = np.asarray(A, dtype=float) * t
    n = A.shape[0]
    S, term = np.eye(n), np.eye(n)
    for k in range(1, terms):
        term = term @ A / k
        S = S + term
    return S


def generalized_symmetric_eig(K, M):
    """廣義特徵值問題 K x = λ M x（法模態；Riley 第 9 章）。

    做法：M = R^T R（Cholesky）→ 轉成對稱標準問題
    (R^{-T} K R^{-1}) y = λ y，再把 x = R^{-1} y 還原。
    """
    from .elimination import cholesky

    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    R = cholesky(M)
    Rinv = np.linalg.inv(R)
    S = Rinv.T @ K @ Rinv
    S = (S + S.T) / 2
    lam, Y = np.linalg.eigh(S)
    X = Rinv @ Y
    # 以 x^T M x = 1 正規化（法模態的標準歸一）
    for k in range(X.shape[1]):
        X[:, k] /= np.sqrt(X[:, k] @ M @ X[:, k])
    return lam, X


def markov_steady_state(P, iters: int = 10000, tol: float = 1e-14):
    """Markov 矩陣（各欄和為 1）的穩態：λ = 1 的特徵向量（Strang 附錄 8）。"""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    x = np.ones(n) / n
    for _ in range(iters):
        x_new = P @ x
        if np.max(np.abs(x_new - x)) < tol:
            break
        x = x_new
    return x / x.sum()
