"""正交性：投影、Gram–Schmidt、Householder QR、最小平方、偽逆。

對應 Strang 第 4 章（含 4.5 偽逆）與 Riley 8.1（內積空間）、31.6（最小平方）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "project_onto_vector",
    "projection_matrix",
    "project_onto_subspace",
    "gram_schmidt",
    "qr_gram_schmidt",
    "householder_qr",
    "least_squares_normal",
    "least_squares_qr",
    "fit_polynomial",
    "pseudoinverse",
    "min_norm_solution",
    "gram_matrix",
    "is_orthonormal",
]


def project_onto_vector(a, b):
    """把 b 投影到 a 的方向：p = (a·b / a·a) a。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return (a @ b) / (a @ a) * a


def projection_matrix(A) -> np.ndarray:
    """投影到 C(A) 的矩陣 P = A (A^T A)^{-1} A^T（A 需各欄獨立）。

    P 的兩個特徵性質：P^2 = P（投影兩次等於一次）、P^T = P。
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A @ np.linalg.solve(A.T @ A, A.T)


def project_onto_subspace(A, b):
    """b 在 C(A) 上的投影 p 與誤差 e = b - p（e ⟂ C(A)）。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    xhat = least_squares_normal(A, b)
    p = A @ xhat
    return p, b - p


def gram_matrix(A) -> np.ndarray:
    """Gram 矩陣 A^T A：元素是各欄的兩兩內積，對稱半正定。"""
    A = np.asarray(A, dtype=float)
    return A.T @ A


def gram_schmidt(A, normalize: bool = True) -> np.ndarray:
    """古典 Gram–Schmidt（用修正版 MGS 以提高數值穩定度）。

    依序把每一欄減去它在先前所有欄上的投影，剩下的部分就與它們正交。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):                       # 修正版：邊減邊更新
            v -= (Q[:, i] @ v) * Q[:, i]
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            raise ValueError(f"第 {j} 欄與前面的欄相依，Gram–Schmidt 得到零向量")
        Q[:, j] = v / nrm if normalize else v
    return Q


def qr_gram_schmidt(A) -> tuple[np.ndarray, np.ndarray]:
    """A = QR，Q 的欄正交單位化，R = Q^T A 為上三角。"""
    A = np.asarray(A, dtype=float)
    Q = gram_schmidt(A)
    R = Q.T @ A
    R[np.tril_indices_from(R, -1)] = 0.0
    return Q, R


def householder_qr(A) -> tuple[np.ndarray, np.ndarray]:
    """Householder 反射法的完整 QR（數值上優於 Gram–Schmidt）。

    每一步用反射矩陣 H = I - 2vv^T/(v^T v) 把一欄的下方元素打成 0。
    """
    R = np.array(A, dtype=float, copy=True)
    m, n = R.shape
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        nx = np.linalg.norm(x)
        if nx < 1e-15:
            continue
        v = x.copy()
        v[0] += np.sign(x[0] if x[0] != 0 else 1.0) * nx   # 選符號避免相減誤差
        vn = np.linalg.norm(v)
        if vn < 1e-15:
            continue
        v = v / vn
        R[k:, :] -= 2.0 * np.outer(v, v @ R[k:, :])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    R[np.abs(R) < 1e-14] = 0.0
    return Q, R


def least_squares_normal(A, b) -> np.ndarray:
    """正規方程 A^T A x = A^T b（各欄獨立時唯一解）。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    return np.linalg.solve(A.T @ A, A.T @ b)


def least_squares_qr(A, b) -> np.ndarray:
    """用 QR 解最小平方：Rx = Q^T b，避免形成條件數平方的 A^T A。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    Q, R = householder_qr(A)
    n = A.shape[1]
    from .elimination import back_substitution
    return back_substitution(R[:n, :n], (Q.T @ b)[:n])


def fit_polynomial(x, y, degree: int) -> np.ndarray:
    """最小平方多項式擬合，回傳係數（由常數項往高次排列）。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    A = np.vander(x, degree + 1, increasing=True)      # Vandermonde 矩陣
    return least_squares_normal(A, y)


def pseudoinverse(A, tol: float = 1e-12) -> np.ndarray:
    """由 SVD 定義的偽逆 A^+ = V Σ^+ U^T（Strang 4.5）。

    A^+ 把 C(A) 上的向量送回 row space，並把 N(A^T) 送到 0。
    """
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.array([1.0 / si if si > tol * max(s.max(initial=0.0), 1e-300) else 0.0
                      for si in s])
    return Vt.T @ np.diag(s_inv) @ U.T


def min_norm_solution(A, b) -> np.ndarray:
    """欠定系統中長度最短的解 x+ = A^+ b（落在 row space 內）。"""
    return pseudoinverse(A) @ np.asarray(b, dtype=float).ravel()


def is_orthonormal(Q, tol: float = 1e-10) -> bool:
    """檢查 Q^T Q = I。"""
    Q = np.asarray(Q, dtype=float)
    return bool(np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol))
