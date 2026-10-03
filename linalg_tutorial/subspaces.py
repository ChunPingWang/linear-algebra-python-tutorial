"""四個基本子空間：rref、A = CR、零空間、Ax = b 的完整解。

對應 Strang 第 1.4 與第 3 章（C(A)、N(A)、C(A^T)、N(A^T)）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rref",
    "pivot_columns",
    "rank",
    "rank_svd",
    "cr_factor",
    "nullspace",
    "column_space",
    "row_space",
    "left_nullspace",
    "particular_solution",
    "complete_solution",
    "is_independent",
    "coordinates_in_basis",
    "four_subspace_report",
]

TOL = 1e-10


def rref(A, tol: float = TOL) -> tuple[np.ndarray, list[int]]:
    """化簡列階梯形式 R0（reduced row echelon form）與主元欄位置。

    規則：每個主元化成 1，主元所在欄的其他元素全部清成 0。
    回傳 (R0, pivots)；R0 的非零列數 = rank。
    """
    R = np.array(A, dtype=float, copy=True)
    m, n = R.shape
    # 主元門檻必須隨矩陣的尺度調整：絕對門檻對元素很小（或很大）的矩陣會誤判
    scale = float(np.max(np.abs(R))) if R.size else 0.0
    thresh = tol * max(m, n) * max(scale, 1.0)
    pivots: list[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        p = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[p, col]) <= thresh:
            R[row:, col] = 0.0          # 整欄（在剩下的列中）都是 0 → 自由欄
            continue
        if p != row:
            R[[row, p], :] = R[[p, row], :]
        R[row, :] /= R[row, col]
        for i in range(m):
            if i != row and abs(R[i, col]) > tol:
                R[i, :] -= R[i, col] * R[row, :]
        pivots.append(col)
        row += 1
    R[np.abs(R) <= thresh] = 0.0
    return R, pivots


def pivot_columns(A, tol: float = TOL) -> list[int]:
    """A 的主元欄（= 由左到右第一組獨立欄）的索引。"""
    return rref(A, tol)[1]


def rank(A, tol: float = TOL) -> int:
    """秩 r = 獨立欄數 = 獨立列數（用消去法/rref 計算）。

    注意：消去法**不是**可靠的數值秩判定器。對接近退化的矩陣，
    消去過程的捨入誤差會讓「幾乎為零」的主元看起來不為零。
    數值上要判斷秩，請用 :func:`rank_svd`（看奇異值的大小）。
    這正是 Strang 第 7 章強調「奇異值優於主元」的理由。
    """
    return len(pivot_columns(A, tol))


def rank_svd(A, tol: float | None = None) -> int:
    """用奇異值判斷秩（數值上可靠的做法）。

    預設門檻是 ``max(m, n) * σ_max * eps``，與 ``numpy.linalg.matrix_rank`` 相同。
    """
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if tol is None:
        tol = max(A.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
    return int(np.sum(s > tol))


def cr_factor(A, tol: float = TOL) -> tuple[np.ndarray, np.ndarray]:
    """Strang 的 A = CR：C 是 A 的 r 個獨立欄，R 是 rref 去掉零列。

    C 的欄是 column space 的基底；R 的列是 row space 的基底。
    """
    A = np.asarray(A, dtype=float)
    R0, piv = rref(A, tol)
    C = A[:, piv]
    R = R0[:len(piv), :]
    return C, R


def nullspace(A, tol: float = TOL) -> np.ndarray:
    """零空間 N(A) 的一組基底（每個自由變數給一個特殊解）。

    R0 = [I F]（主元欄重排後），對應的零空間基底就是 [-F; I] 重排回去。
    回傳 n x (n-r) 矩陣，欄為基底向量；r = n 時回傳 n x 0。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    R0, piv = rref(A, tol)
    free = [c for c in range(n) if c not in piv]
    N = np.zeros((n, len(free)))
    for k, f in enumerate(free):
        N[f, k] = 1.0
        for i, p in enumerate(piv):
            N[p, k] = -R0[i, f]
    return N


def column_space(A, tol: float = TOL) -> np.ndarray:
    """C(A) 的基底：A 的主元欄本身（維度 r）。"""
    A = np.asarray(A, dtype=float)
    return A[:, pivot_columns(A, tol)]


def row_space(A, tol: float = TOL) -> np.ndarray:
    """C(A^T) 的基底：rref 的非零列（維度 r），以欄向量形式回傳。"""
    R0, piv = rref(A, tol)
    return R0[:len(piv), :].T


def left_nullspace(A, tol: float = TOL) -> np.ndarray:
    """N(A^T) 的基底（維度 m - r）：滿足 y^T A = 0 的 y。"""
    return nullspace(np.asarray(A, dtype=float).T, tol)


def particular_solution(A, b, tol: float = TOL):
    """Ax = b 的一個特解（自由變數全設 0）；無解時回傳 None。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    M = np.hstack([A, b])
    R0, piv_aug = rref(M, tol)
    n = A.shape[1]
    if n in piv_aug:                      # 增廣欄成為主元欄 → 出現 0 = 1
        return None
    x = np.zeros(n)
    for i, p in enumerate(piv_aug):
        x[p] = R0[i, n]
    return x


def complete_solution(A, b, tol: float = TOL) -> dict:
    """Ax = b 的完整解 x = x_p + （零空間的任意組合）。

    回傳 dict：consistent / particular / nullspace_basis / rank /
    n_free / description。
    """
    A = np.asarray(A, dtype=float)
    xp = particular_solution(A, b, tol)
    N = nullspace(A, tol)
    r = rank(A, tol)
    m, n = A.shape
    if xp is None:
        desc = "無解：b 不在 C(A) 內（消去後出現 0 = 非零）"
    elif N.shape[1] == 0:
        desc = "恰有一組解：N(A) = {0}，各欄獨立"
    else:
        desc = f"無窮多解：特解 + N(A) 的 {N.shape[1]} 維任意組合"
    return {
        "consistent": xp is not None,
        "particular": xp,
        "nullspace_basis": N,
        "rank": r,
        "n_free": n - r,
        "shape": (m, n),
        "description": desc,
    }


def is_independent(vectors, tol: float = TOL) -> bool:
    """給一組欄向量（矩陣），判斷是否線性獨立。"""
    A = np.asarray(vectors, dtype=float)
    return rank(A, tol) == A.shape[1]


def coordinates_in_basis(basis, v, tol: float = TOL):
    """把向量 v 寫成基底（矩陣 basis 的欄）的座標；不在 span 內回傳 None。"""
    return particular_solution(np.asarray(basis, dtype=float), v, tol)


def four_subspace_report(A, tol: float = TOL) -> dict:
    """一次算出四個基本子空間的基底與維度（Strang 3.5 的大圖）。"""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    r = rank(A, tol)
    return {
        "shape": (m, n),
        "rank": r,
        "column_space": column_space(A, tol),      # 在 R^m，維度 r
        "nullspace": nullspace(A, tol),            # 在 R^n，維度 n - r
        "row_space": row_space(A, tol),            # 在 R^n，維度 r
        "left_nullspace": left_nullspace(A, tol),  # 在 R^m，維度 m - r
        "dims": {"C(A)": r, "N(A)": n - r, "C(A^T)": r, "N(A^T)": m - r},
    }
