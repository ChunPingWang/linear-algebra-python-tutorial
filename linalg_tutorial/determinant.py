"""行列式：餘因子展開、LU 乘主元、Cramer 法則、面積與體積。

對應 Strang 第 5 章與 Riley 8.9。
"""

from __future__ import annotations

import numpy as np

from .elimination import plu

__all__ = [
    "det_cofactor",
    "det_lu",
    "minor",
    "cofactor_matrix",
    "inverse_by_cofactors",
    "cramer",
    "volume",
    "big_formula_terms",
]


def minor(A, i: int, j: int) -> np.ndarray:
    """去掉第 i 列與第 j 欄後的子矩陣 M_ij。"""
    A = np.asarray(A, dtype=float)
    return np.delete(np.delete(A, i, axis=0), j, axis=1)


def det_cofactor(A) -> float:
    """沿第一列做餘因子展開（遞迴定義，O(n!)，只適合小矩陣）。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("行列式只對方陣有定義")
    if n == 1:
        return float(A[0, 0])
    if n == 2:
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    total = 0.0
    for j in range(n):
        if A[0, j] == 0.0:
            continue
        total += (-1.0) ** j * A[0, j] * det_cofactor(minor(A, 0, j))
    return float(total)


def det_lu(A) -> float:
    """實務做法：PA = LU ⇒ det A = ±(U 的主元相乘)，符號由換列次數決定。"""
    A = np.asarray(A, dtype=float)
    P, L, U = plu(A)
    # P 是列置換矩陣；其行列式為 (-1)^(對換次數)
    perm = np.argmax(P, axis=1)
    sign, seen = 1.0, np.zeros(len(perm), dtype=bool)
    for i in range(len(perm)):            # 依循環分解計算置換符號
        if seen[i]:
            continue
        length, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return float(sign * np.prod(np.diag(U)))


def cofactor_matrix(A) -> np.ndarray:
    """餘因子矩陣 C，C_ij = (-1)^{i+j} det M_ij。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    return np.array([[(-1.0) ** (i + j) * det_cofactor(minor(A, i, j))
                      for j in range(n)] for i in range(n)])


def inverse_by_cofactors(A) -> np.ndarray:
    """A^{-1} = C^T / det A（餘因子公式，解釋 A^{-1} 為何都是分式）。"""
    d = det_lu(A)
    if abs(d) < 1e-14:
        raise ValueError("det A = 0，矩陣不可逆")
    return cofactor_matrix(A).T / d


def cramer(A, b) -> np.ndarray:
    """Cramer 法則：x_j = det(B_j)/det(A)，B_j 是把第 j 欄換成 b。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    d = det_lu(A)
    if abs(d) < 1e-14:
        raise ValueError("det A = 0，Cramer 法則不適用")
    n = A.shape[1]
    x = np.empty(n)
    for j in range(n):
        Bj = A.copy()
        Bj[:, j] = b
        x[j] = det_lu(Bj) / d
    return x


def volume(vectors) -> float:
    """以欄向量為邊的平行多面體體積 = |det|（Strang 5.3）。"""
    return abs(det_lu(np.asarray(vectors, dtype=float)))


def big_formula_terms(A) -> list[tuple[tuple[int, ...], float, float]]:
    """「大公式」det A = Σ_σ sign(σ) a_{1σ(1)}...a_{nσ(n)} 的每一項。

    回傳 [(置換, 符號, 乘積), ...]，共 n! 項，用來看清行列式的組合本質。
    """
    from itertools import permutations

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    out = []
    for perm in permutations(range(n)):
        inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        sign = (-1.0) ** inversions
        prod = float(np.prod([A[i, perm[i]] for i in range(n)]))
        out.append((perm, sign, prod))
    return out
