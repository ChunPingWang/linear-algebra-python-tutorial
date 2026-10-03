"""消去法家族：Ax = b、A = LU、PA = LU、Gauss–Jordan 反矩陣。

對應 Strang《Introduction to Linear Algebra》第 2 章與 Riley《Mathematical
Methods》第 8.18 / 27.3 節。所有函式都只用基本陣列運算實作，便於對照課文。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "elimination_matrix",
    "lu_no_pivot",
    "plu",
    "forward_substitution",
    "back_substitution",
    "solve",
    "inverse_gauss_jordan",
    "ldu",
    "cholesky",
    "difference_matrix",
    "second_difference_matrix",
]


# --------------------------------------------------------------------------
# 基本列運算矩陣
# --------------------------------------------------------------------------
def elimination_matrix(n: int, i: int, j: int, mult: float) -> np.ndarray:
    """E_ij：把「第 i 列減去 mult 倍的第 j 列」寫成矩陣。

    E = I - mult * e_i e_j^T，其反矩陣只要把 mult 變號。
    """
    E = np.eye(n)
    E[i, j] -= mult
    return E


def difference_matrix(n: int) -> np.ndarray:
    """一階後向差分矩陣 D（n x n），(Dx)_k = x_k - x_{k-1}，對應微分算子。"""
    D = np.eye(n)
    idx = np.arange(1, n)
    D[idx, idx - 1] = -1.0
    return D


def second_difference_matrix(n: int) -> np.ndarray:
    """二階中央差分矩陣 K = tridiag(-1, 2, -1)，離散化 -d^2/dx^2。

    K 是對稱正定矩陣，貫穿第 2、6、17、18 章。
    """
    K = 2.0 * np.eye(n)
    idx = np.arange(1, n)
    K[idx, idx - 1] = -1.0
    K[idx - 1, idx] = -1.0
    return K


# --------------------------------------------------------------------------
# LU 分解
# --------------------------------------------------------------------------
def lu_no_pivot(A) -> tuple[np.ndarray, np.ndarray]:
    """不換列的 A = LU（Doolittle）。主元為 0 會拋出 ValueError。"""
    U = np.array(A, dtype=float, copy=True)
    n = U.shape[0]
    if U.shape[0] != U.shape[1]:
        raise ValueError("lu_no_pivot 只接受方陣")
    L = np.eye(n)
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            raise ValueError(f"第 {k} 個主元為 0，必須換列（請用 plu）")
        for i in range(k + 1, n):
            mult = U[i, k] / U[k, k]
            L[i, k] = mult
            U[i, k:] -= mult * U[k, k:]
        U[k + 1:, k] = 0.0
    return L, U


def plu(A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """部分軸選取的 PA = LU，回傳 (P, L, U)。

    每一步挑選該行中絕對值最大的元素當主元，這是數值穩定的標準做法
    （Riley 27.3、Strang 附錄 4）。
    """
    U = np.array(A, dtype=float, copy=True)
    n = U.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    for k in range(n):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[p, k]) < 1e-14:
            continue                      # 整欄都是 0：奇異矩陣，跳過
        if p != k:                        # 換列：U、P 全換，L 只換已完成的部分
            U[[k, p], :] = U[[p, k], :]
            P[[k, p], :] = P[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]
        for i in range(k + 1, n):
            mult = U[i, k] / U[k, k]
            L[i, k] = mult
            U[i, k:] -= mult * U[k, k:]
        U[k + 1:, k] = 0.0
    return P, L, U


def ldu(A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """對稱矩陣常用的 A = L D L^T 形式；一般矩陣則得到 A = L D U。"""
    L, U = lu_no_pivot(A)
    d = np.diag(U).copy()
    D = np.diag(d)
    Unorm = U / d[:, None]
    return L, D, Unorm


def cholesky(A) -> np.ndarray:
    """對稱正定矩陣的 A = R^T R（R 上三角），不正定時拋出 ValueError。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if not np.allclose(A, A.T):
        raise ValueError("Cholesky 需要對稱矩陣")
    R = np.zeros((n, n))
    for i in range(n):
        s = A[i, i] - R[:i, i] @ R[:i, i]
        if s <= 0:
            raise ValueError("矩陣不是正定的")
        R[i, i] = np.sqrt(s)
        for j in range(i + 1, n):
            R[i, j] = (A[i, j] - R[:i, i] @ R[:i, j]) / R[i, i]
    return R


# --------------------------------------------------------------------------
# 回代
# --------------------------------------------------------------------------
def forward_substitution(L, b) -> np.ndarray:
    """解下三角系統 Lc = b（往下做）。"""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = len(b)
    c = np.zeros(n)
    for i in range(n):
        c[i] = (b[i] - L[i, :i] @ c[:i]) / L[i, i]
    return c


def back_substitution(U, c) -> np.ndarray:
    """解上三角系統 Ux = c（由最後一個未知數往上回代）。"""
    U = np.asarray(U, dtype=float)
    c = np.asarray(c, dtype=float).ravel()
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise ValueError("U 的主元為 0：方陣奇異，無法唯一回代")
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve(A, b) -> np.ndarray:
    """用 PA = LU + 兩次回代解方陣系統 Ax = b。

    流程：PA = LU → Ax = b 變成 LUx = Pb → 先解 Lc = Pb，再解 Ux = c。
    """
    P, L, U = plu(A)
    b = np.asarray(b, dtype=float).ravel()
    c = forward_substitution(L, P @ b)
    return back_substitution(U, c)


def inverse_gauss_jordan(A) -> np.ndarray:
    """Gauss–Jordan：對 [A I] 做列運算直到變成 [I A^{-1}]。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    M = np.hstack([A.astype(float), np.eye(n)])
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < 1e-12:
            raise ValueError("矩陣不可逆（主元為 0）")
        if p != k:
            M[[k, p], :] = M[[p, k], :]
        M[k, :] /= M[k, k]                       # 主元化成 1
        for i in range(n):                        # 上下都清成 0
            if i != k and M[i, k] != 0.0:
                M[i, :] -= M[i, k] * M[k, :]
    return M[:, n:]
