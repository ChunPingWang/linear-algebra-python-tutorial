# %% [markdown]
# # 第 13 章　附錄精選：秩、分解目錄、張量、條件數、Markov 與圖學
#
# > 對應 Strang 附錄 1（$AB$ 與 $A+B$ 的秩）、2（矩陣分解目錄）、3（參數計數）、
# > 4（數值線性代數的程式庫）、5（Jordan 形式與 Cayley–Hamilton）、6（張量）、
# > 7（條件數）、8（Markov 矩陣與 Perron–Frobenius）、9（消去與分解）、10（計算機圖學），
# > 以及 Riley 8.11（秩）、26（張量）、27.2（迭代法的收斂）。
#
# 這一章把全書的工具整理成一張「地圖」，並補上幾個獨立但重要的主題。
#
# ```bash
# python chapters/ch13_appendices.py
# python tools/build_notebooks.py ch13
# ```

# %%
import pathlib
import sys
import time

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.eigen import markov_steady_state, power_iteration
from linalg_tutorial.elimination import cholesky, ldu, lu_no_pivot, plu
from linalg_tutorial.orthogonal import householder_qr, pseudoinverse
from linalg_tutorial.subspaces import cr_factor, rank, rref
from linalg_tutorial.svd_tools import condition_number, polar_decomposition
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 13.1　附錄 1：秩的不等式
#
# 1. $\operatorname{rank}(AB)\le\min(\operatorname{rank}A,\operatorname{rank}B)$
#    ── 因為 $C(AB)\subseteq C(A)$、$AB$ 的列都是 $B$ 的列的組合
# 2. $\operatorname{rank}(A+B)\le\operatorname{rank}A+\operatorname{rank}B$
# 3. $\operatorname{rank}(A^{\mathsf T}A)=\operatorname{rank}(AA^{\mathsf T})
#    =\operatorname{rank}A=\operatorname{rank}A^{\mathsf T}$
# 4. $A$ 是 $m\times r$、$B$ 是 $r\times n$、**兩者秩都是 $r$** $\Rightarrow$
#    $\operatorname{rank}(AB)=r$

# %%
section("13.1 秩的四條規則與反例")

rng = np.random.default_rng(0)
for trial in range(4):
    A = rng.standard_normal((5, 4)) @ rng.standard_normal((4, 6))     # 秩 ≤ 4
    B = rng.standard_normal((6, 3))
    rA, rB = rank(A), rank(B)
    check(f"#{trial + 1} rank(AB) ≤ min(rank A, rank B)",
          rank(A @ B) <= min(rA, rB))

A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
B1 = np.array([[0.0, 0.0], [0.0, 1.0]])
print(f"\nrank(A) = {rank(A1)}, rank(B) = {rank(B1)}, rank(A+B) = {rank(A1 + B1)}")
check("rank(A+B) ≤ rank A + rank B", rank(A1 + B1) <= rank(A1) + rank(B1))
print(f"反例（等號不成立）：A = B = I ⇒ rank(A+B) = {rank(2 * np.eye(3))} "
      f"≠ {rank(np.eye(3)) + rank(np.eye(3))}")
check("A = B = I 時等號不成立", rank(2 * np.eye(3)) < 2 * rank(np.eye(3)))

for trial in range(4):
    A = rng.standard_normal((7, 5)) @ rng.standard_normal((5, 9))
    check(f"#{trial + 1} rank(AᵀA) = rank(AAᵀ) = rank(A)",
          rank(A.T @ A) == rank(A @ A.T) == rank(A))

C = np.array([[1.0], [1.0], [1.0]])
R = np.array([[1.0, 2.0, -3.0]])
print(f"\n規則 4：C 是 3x1 秩 1，R 是 1x3 秩 1 ⇒ rank(CR) = {rank(C @ R)}")
check("rank(CR) = r = 1", rank(C @ R) == 1)
print(f"但反過來 rank(RC) = {rank(R @ C)}（RC = {float((R @ C).item()):.0f}，"
      f"剛好是 0 ⇒ 規則 4 的順序很重要）")
check("RC = 0（規則 4 不能反過來用）", R @ C, np.zeros((1, 1)))

# 秩與 SVD：rank = 非零奇異值個數
A_num = rng.standard_normal((6, 4)) @ rng.standard_normal((4, 6))
s = np.linalg.svd(A_num, compute_uv=False)
print(f"\n數值上的秩：σ = {np.round(s, 8)}")
print(f"  rank = σ > tol 的個數 = {int(np.sum(s > 1e-10))}")
check("rank(A) = 非零 σ 的個數", rank(A_num), int(np.sum(s > 1e-10)))

# %% [markdown]
# ## 13.2　附錄 2：矩陣分解目錄
#
# 整本書的分解一覽（下面逐一在同一組矩陣上驗證）：
#
# | # | 分解 | 條件 | 本書章節 |
# |---|---|---|---|
# | 1 | $A=CR$ | 任意 | 第 1、3 章 |
# | 2 | $A=CW^{-1}B$ | $W$ 是 $r\times r$ 可逆子矩陣 | 第 3 章 |
# | 3 | $A=LU$ | 無需換列 | 第 2 章 |
# | 4 | $A=LDU$（對稱時 $LDL^{\mathsf T}$）| 無需換列 | 第 2、7 章 |
# | 5 | $PA=LU$ | 可逆 | 第 2 章 |
# | 6 | $S=C^{\mathsf T}C$（Cholesky）| 對稱正定 | 第 7 章 |
# | 7 | $A=QR$ | 各欄獨立 | 第 4 章 |
# | 8 | $A=X\Lambda X^{-1}$ | 可對角化 | 第 6 章 |
# | 9 | $S=Q\Lambda Q^{\mathsf T}$（譜定理）| 對稱 | 第 7 章 |
# | 10 | $A=BJB^{-1}$（Jordan）| 任意方陣 | 第 10 章 |
# | 11 | $A=U\Sigma V^{\mathsf T}$（SVD）| **任意** | 第 9 章 |
# | 12 | $A^{+}=V\Sigma^{+}U^{\mathsf T}$ | 任意 | 第 4、9 章 |
# | 13 | $A=QS$（極分解）| 任意 | 第 9 章 |
# | 14 | $A=U\Lambda U^{\mathsf H}$ | normal（$A^{\mathsf H}A=AA^{\mathsf H}$）| 第 7 章 |
# | 15 | $A=QTQ^{\mathsf H}$（Schur）| **任意方陣** | 本節 |
# | 16 | $F_n$ 的 FFT 遞迴分解 | Fourier 矩陣 | 第 7 章 |

# %%
section("13.2 十六個分解，一次全部驗證")

A = np.array([
    [2.0, 6.0, 4.0, 1.0],
    [4.0, 12.0, 8.0, 3.0],
    [1.0, 3.0, 5.0, 2.0],
])
S = np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]])
M = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
r = rank(A)
print(f"測試矩陣：A 是 {A.shape}（rank {r}）、S 對稱正定 3x3、M 一般 3x3\n")

# 1. A = CR
C, R = cr_factor(A)
check("① A = CR", C @ R, A)

# 2. A = C W⁻¹ B
R0, piv = rref(A)
rows_pivot = []
seen = set()
for i in range(r):                      # 找出 r 個獨立列（主元列）
    rows_pivot.append(i)
B_rows = A[rows_pivot, :]
W = A[np.ix_(rows_pivot, piv)]
check("② A = C·W⁻¹·B（W 是獨立列與獨立欄交會的可逆子矩陣）",
      C @ np.linalg.inv(W) @ B_rows, A)
print(f"   W = A[獨立列, 獨立欄] = \n{W}，det W = {np.linalg.det(W):.4g}")

# 3/4/5. LU 家族
L, U = lu_no_pivot(M)
check("③ M = LU", L @ U, M)
L2, D2, U2 = ldu(M)
check("④ M = LDU（主元抽出到 D）", L2 @ D2 @ U2, M)
P, Lp, Up = plu(A[:, :3])
check("⑤ PA = LU（含換列）", P @ A[:, :3], Lp @ Up)

# 6. Cholesky
Rc = cholesky(S)
check("⑥ S = CᵀC（Cholesky，C 上三角）", Rc.T @ Rc, S)

# 7. QR
Q7, R7 = householder_qr(A.T)            # Aᵀ 是 4x3，各欄獨立
check("⑦ Aᵀ = QR", Q7 @ R7, A.T)
check("   QᵀQ = I（Householder 給完整的 m x m 正交 Q）",
      Q7.T @ Q7, np.eye(Q7.shape[1]))

# 8. XΛX⁻¹
lam8, X8 = np.linalg.eig(M)
check("⑧ M = XΛX⁻¹", np.real(X8 @ np.diag(lam8) @ np.linalg.inv(X8)), M, tol=1e-8)

# 9. 譜定理
lam9, Q9 = np.linalg.eigh(S)
check("⑨ S = QΛQᵀ（譜定理）", Q9 @ np.diag(lam9) @ Q9.T, S)

# 10. Jordan
J = np.array([[3.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
Bj = rng.standard_normal((3, 3))
A_jordan = Bj @ J @ np.linalg.inv(Bj)
check("⑩ A = BJB⁻¹（Jordan 形式）", np.linalg.inv(Bj) @ A_jordan @ Bj, J, tol=1e-8)

# 11/12. SVD 與偽逆
U11, s11, Vt11 = np.linalg.svd(A)
Sigma = np.zeros_like(A)
Sigma[:len(s11), :len(s11)] = np.diag(s11)
check("⑪ A = UΣVᵀ（SVD，任意矩陣）", U11 @ Sigma @ Vt11, A)
s_plus = np.array([1 / x if x > 1e-10 else 0.0 for x in s11])
Sigma_plus = np.zeros((A.shape[1], A.shape[0]))
Sigma_plus[:len(s11), :len(s11)] = np.diag(s_plus)
check("⑫ A⁺ = VΣ⁺Uᵀ", Vt11.T @ Sigma_plus @ U11.T, pseudoinverse(A), tol=1e-10)

# 13. 極分解
Qp, Sp = polar_decomposition(M)
check("⑬ M = QS（極分解）", Qp @ Sp, M)

# 14. normal 矩陣
N = np.array([[1.0, -1.0], [1.0, 1.0]])          # 正規矩陣（NᵗN = NNᵗ）
check("⑭ N 是 normal（NᴴN = NNᴴ）", N.conj().T @ N, N @ N.conj().T)
lam14, U14 = np.linalg.eig(N)
check("   N = UΛU⁻¹ 且 U unitary", U14.conj().T @ U14, np.eye(2, dtype=complex),
      tol=1e-10)

# 15. Schur 分解（任意方陣都可以！）
try:
    from scipy.linalg import schur
    T15, Z15 = schur(A_jordan, output="complex")
    check("⑮ A = QTQᴴ（Schur：任意方陣都可三角化）",
          Z15 @ T15 @ Z15.conj().T, A_jordan.astype(complex), tol=1e-8)
    check("   Q unitary", Z15.conj().T @ Z15, np.eye(3, dtype=complex), tol=1e-10)
    check("   T 上三角且對角線 = 特徵值",
          np.sort_complex(np.diag(T15)),
          np.sort_complex(np.linalg.eigvals(A_jordan)), tol=1e-7)
except ImportError:
    print("  （未安裝 scipy，略過 Schur 分解）")

# 16. FFT 的遞迴分解（第 7 章已驗證過，這裡用 n = 16 再確認一次）
n = 16
j, k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
Fn = np.exp(2j * np.pi * j * k / n)
half = n // 2
jh, kh = np.meshgrid(np.arange(half), np.arange(half), indexing="ij")
Fh = np.exp(2j * np.pi * jh * kh / half)
D = np.diag(np.exp(2j * np.pi * np.arange(half) / n))
perm = np.zeros((n, n))
for i, idx in enumerate(list(range(0, n, 2)) + list(range(1, n, 2))):
    perm[i, idx] = 1.0
middle = np.block([[Fh, np.zeros((half, half))], [np.zeros((half, half)), Fh]])
check("⑯ F₁₆ = [[I,D],[I,−D]]·diag(F₈,F₈)·P（FFT 一步）",
      np.vstack([np.hstack([np.eye(half), D]),
                 np.hstack([np.eye(half), -D])]) @ middle @ perm, Fn, tol=1e-8)

# %% [markdown]
# ## 13.3　附錄 3：參數計數
#
# 每個分解的兩邊**自由參數個數必須相同** ── 這是檢驗理解的好方法。
#
# | 矩陣 | 自由參數 | 理由 |
# |---|---|---|
# | 下三角 $L$（對角線為 1）| $\frac12n(n-1)$ | 對角線下方 |
# | 上三角 $U$ | $\frac12n(n+1)$ | 含對角線 |
# | 正交 $Q$ | $\frac12n(n-1)$ | 第 $k$ 欄有 $n-k$ 個自由度 |
# | 對稱 $S$ | $\frac12n(n+1)$ | 上三角（含對角）|
# | 對角 $\Lambda$ | $n$ | |
# | 特徵向量 $X$ | $n^2-n$ | 每欄可任意縮放 |
#
# 於是 $A=LU$：$\frac12n(n-1)+\frac12n(n+1)=n^2$ ✓
#
# 秩 $r$ 的 $m\times n$ 矩陣有 $(m+n-r)r$ 個自由參數 ──
# 這也是 $A=CR$ 與 $A=U_r\Sigma_rV_r^{\mathsf T}$ 的共同答案。

# %%
section("13.3 參數計數：兩邊必須相等")

print(f"{'分解':<22} | {'左邊 (A)':>10} | {'右邊的計數':>36} | 相等?")
print("-" * 86)
for n in [3, 5, 8]:
    counts = [
        ("A = LU", n * n, n * (n - 1) // 2 + n * (n + 1) // 2,
         f"L:{n*(n-1)//2} + U:{n*(n+1)//2}"),
        ("A = QR", n * n, n * (n - 1) // 2 + n * (n + 1) // 2,
         f"Q:{n*(n-1)//2} + R:{n*(n+1)//2}"),
        ("S = QΛQᵀ", n * (n + 1) // 2, n * (n - 1) // 2 + n,
         f"Q:{n*(n-1)//2} + Λ:{n}"),
        ("A = XΛX⁻¹", n * n, n * n - n + n, f"X:{n*n-n} + Λ:{n}"),
        ("A = QS", n * n, n * (n - 1) // 2 + n * (n + 1) // 2,
         f"Q:{n*(n-1)//2} + S:{n*(n+1)//2}"),
    ]
    print(f"--- n = {n} ---")
    for name, left, right, detail in counts:
        print(f"{name:<22} | {left:>10} | {detail:>36} = {right:<6} | {left == right}")
        check(f"  n={n} {name} 參數數相等", left == right)

print("\n秩 r 的 m x n 矩陣的自由參數 = (m + n − r)·r：")
for m_, n_, r_ in [(5, 4, 2), (10, 7, 3), (100, 50, 5)]:
    total = (m_ + n_ - r_) * r_
    cr_count = m_ * r_ + n_ * r_ - r_ * r_
    svd_count = (m_ * r_ - r_ * (r_ + 1) // 2) + r_ + (n_ * r_ - r_ * (r_ + 1) // 2)
    print(f"  {m_}x{n_} 秩 {r_}：公式 {total}，A=CR 計數 {cr_count}，"
          f"SVD 計數 {svd_count}，完整矩陣 {m_ * n_}")
    check(f"  (m+n−r)r = CR 的計數", total, cr_count)
    check(f"  = SVD 的計數", total, svd_count)
print("  ⇒ 低秩矩陣的「資訊量」遠小於 mn，這就是壓縮能成功的理由")

# 數值驗證：正交矩陣的自由度 = 做 Gram–Schmidt 時剩下的自由度
print("\n驗證 Q 的自由度：正交矩陣的切空間 = {Q₀·A : A 反對稱}")
from linalg_tutorial.eigen import matrix_exp_series
for n in [3, 4, 5]:
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # 反對稱矩陣的基底：E_ij − E_ji（i < j），共 n(n−1)/2 個
    skew_basis = []
    for i in range(n):
        for j in range(i + 1, n):
            E = np.zeros((n, n))
            E[i, j], E[j, i] = 1.0, -1.0
            skew_basis.append(E)
    tangents = np.array([(Q0 @ E).ravel() for E in skew_basis])
    dim = np.linalg.matrix_rank(tangents)
    print(f"  n = {n}：切空間維度 = {dim}，理論 n(n−1)/2 = {n * (n - 1) // 2}")
    check(f"  n = {n}：正交矩陣流形的維度 = n(n−1)/2", dim, n * (n - 1) // 2)
    # 沿著切方向走，確實還在正交矩陣上
    A_rand = sum(c * E for c, E in zip(rng.standard_normal(len(skew_basis)), skew_basis))
    Q_moved = Q0 @ matrix_exp_series(A_rand, 0.3, 60)
    check(f"  n = {n}：Q₀·exp(tA)（A 反對稱）仍然正交",
          Q_moved.T @ Q_moved, np.eye(n), tol=1e-9)

# %% [markdown]
# ## 13.4　附錄 5：Cayley–Hamilton 定理
#
# > **每個矩陣都滿足它自己的特徵方程**：$p(\lambda)=\det(A-\lambda I)$
# > $\Rightarrow$ $p(A)=0$（零矩陣）
#
# 用 Jordan 形式看最清楚：$p(A)=Bp(J)B^{-1}$，而每個 Jordan 塊
# $(J_i-\lambda_iI)^{n_i}=0$。

# %%
section("13.4 Cayley–Hamilton 定理")


def char_poly_coeffs(A):
    """用 numpy 由特徵值組出特徵多項式的係數（最高次在前）。"""
    return np.poly(np.asarray(A, dtype=float))


def eval_poly_matrix(coeffs, A):
    """把多項式代入矩陣：p(A) = c₀Aⁿ + c₁Aⁿ⁻¹ + ... + cₙI。"""
    n = A.shape[0]
    out = np.zeros((n, n), dtype=complex)
    for c in coeffs:
        out = out @ A + c * np.eye(n)
    return np.real_if_close(out)


for name, Mx in [("一般矩陣 M", M), ("對稱矩陣 S", S),
                 ("Jordan 矩陣 J", J), ("隨機 5x5", rng.standard_normal((5, 5)))]:
    coeffs = char_poly_coeffs(Mx)
    pA = eval_poly_matrix(coeffs, Mx)
    print(f"  {name:<16} 特徵多項式次數 {len(coeffs) - 1}，"
          f"‖p(A)‖ = {np.linalg.norm(pA):.3e}")
    check(f"  {name}：p(A) = 0", np.linalg.norm(pA) < 1e-8 * max(1, np.linalg.norm(Mx) ** len(coeffs)))

print(f"\nJordan 的例子：J 有 λ = 3,3,5 ⇒ p(λ) = (λ−3)²(λ−5)")
check("(J − 3I)²(J − 5I) = 0",
      np.linalg.matrix_power(J - 3 * np.eye(3), 2) @ (J - 5 * np.eye(3)),
      np.zeros((3, 3)))
print("推論：A⁻¹ 可以寫成 A 的多項式（把 p(A) = 0 整理一下）")
Minv_poly = None
coeffs = char_poly_coeffs(M)
# p(λ) = λ³ + c₁λ² + c₂λ + c₃ = 0 ⇒ A⁻¹ = −(A² + c₁A + c₂I)/c₃
c1, c2, c3 = coeffs[1], coeffs[2], coeffs[3]
Minv_poly = -(M @ M + c1 * M + c2 * np.eye(3)) / c3
check("A⁻¹ = −(A² + c₁A + c₂I)/c₃", Minv_poly, np.linalg.inv(M), tol=1e-10)

# %% [markdown]
# ## 13.5　附錄 6：張量（多維陣列）
#
# 矩陣是「2 路張量」；彩色影像是「3 路張量」$T_{ijk}$（第三個索引是 RGB）。
#
# **壞消息**：3 路張量**沒有** SVD。我們只能求近似的
# **CP 分解**（CANDECOMP/PARAFAC）
#
# $$T\approx\sum_{k=1}^{R}\mathbf a_k\circ\mathbf b_k\circ\mathbf c_k,
# \qquad T_{ijl}\approx\sum_k a_{ik}b_{jk}c_{lk}$$
#
# 正交性一般做不到，秩也沒有唯一定義。
# 常用解法是**交替最小平方**（ALS）：固定兩組、對第三組解最小平方。

# %%
section("13.5 張量與 CP 分解（交替最小平方）")

I_, J_, K_ = 12, 10, 8
R_true = 3
rg = np.random.default_rng(5)
A_true = rg.standard_normal((I_, R_true))
B_true = rg.standard_normal((J_, R_true))
C_true = rg.standard_normal((K_, R_true))
T = np.einsum("ir,jr,kr->ijk", A_true, B_true, C_true)
print(f"張量 T 的形狀 = {T.shape}（{T.size} 個元素），由 {R_true} 個秩一項組成")
print(f"秩一項的參數量 = {R_true * (I_ + J_ + K_)}（遠小於 {T.size}）")


def unfold(T, mode):
    """把張量沿指定模態攤成矩陣（mode-n unfolding）。"""
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def khatri_rao(A, B):
    """逐欄的 Kronecker 積（Khatri–Rao 乘積）。"""
    return np.einsum("ir,jr->ijr", A, B).reshape(-1, A.shape[1])


def cp_als_once(T, R, iters=300, seed=None, svd_init=False):
    """CP 分解的交替最小平方：固定兩個因子，對第三個解最小平方。"""
    if svd_init:
        A, B, C = [np.linalg.svd(unfold(T, m), full_matrices=False)[0][:, :R]
                   for m in range(3)]
    else:
        rg = np.random.default_rng(seed)
        A, B, C = [rg.standard_normal((T.shape[m], R)) for m in range(3)]
    for _ in range(iters):
        A = unfold(T, 0) @ np.linalg.pinv(khatri_rao(B, C).T)
        B = unfold(T, 1) @ np.linalg.pinv(khatri_rao(A, C).T)
        C = unfold(T, 2) @ np.linalg.pinv(khatri_rao(A, B).T)
    err = (np.linalg.norm(T - np.einsum("ir,jr,kr->ijk", A, B, C))
           / np.linalg.norm(T))
    return A, B, C, err


def cp_als(T, R, iters=300, restarts=3):
    """ALS 容易卡在局部極小（所謂的 swamp），所以用 SVD 初始化 + 多次重啟取最好的。"""
    best = cp_als_once(T, R, iters, svd_init=True)
    for s in range(restarts):
        cand = cp_als_once(T, R, iters, seed=s)
        if cand[3] < best[3]:
            best = cand
    return best[:3]


print(f"\n{'R':>4} | {'相對誤差':>12} | {'參數量':>8}")
print("-" * 30)
for R_ in [1, 2, 3, 4]:
    Ah, Bh, Ch = cp_als(T, R_)
    T_hat = np.einsum("ir,jr,kr->ijk", Ah, Bh, Ch)
    err = np.linalg.norm(T - T_hat) / np.linalg.norm(T)
    print(f"{R_:>4} | {err:>12.3e} | {R_ * (I_ + J_ + K_):>8}")
    if R_ == R_true:
        check(f"  R = 真實秩 {R_true} 時幾乎完全重建", err < 1e-6)
    if R_ < R_true:
        check(f"  R = {R_} < 真實秩 ⇒ 仍有可觀誤差", err > 1e-3)

# 張量的「不能正交化」
Ah, Bh, Ch = cp_als(T, R_true)
G = Ah.T @ Ah
off = np.abs(G - np.diag(np.diag(G))).max()
print(f"\nCP 分解得到的因子不正交：A 的 Gram 矩陣最大非對角元素 = {off:.4f}")
print("  另一個與 SVD 的差異：ALS 會卡在局部極小（swamp），需要多次重啟")
check("CP 的因子一般不正交（與 SVD 不同）", off > 1e-6)

# 彩色影像就是 3 路張量
img = np.zeros((40, 40, 3))
yy, xx = np.mgrid[0:40, 0:40] / 40
img[:, :, 0] = xx
img[:, :, 1] = yy
img[:, :, 2] = 0.5 * (xx + yy)
print(f"\n彩色影像張量 {img.shape}：第三個索引是 R/G/B")
check("第三個通道 = 前兩個的平均（人造的相依性）",
      img[:, :, 2], 0.5 * (img[:, :, 0] + img[:, :, 1]))
Ai, Bi, Ci = cp_als(img, 2)
err_img = np.linalg.norm(img - np.einsum("ir,jr,kr->ijk", Ai, Bi, Ci)) / np.linalg.norm(img)
print(f"  用 R = 2 的 CP 分解，相對誤差 = {err_img:.3e}")
check("  R = 2 就足以重建這張人造影像", err_img < 1e-6)

# %% [markdown]
# ## 13.6　附錄 7：條件數
#
# $$\kappa(A)=\|A\|\,\|A^{-1}\|=\frac{\sigma_{\max}}{\sigma_{\min}}
# =\max_{\mathbf b,\Delta\mathbf b}
# \frac{\|\Delta\mathbf x\|/\|\mathbf x\|}{\|\Delta\mathbf b\|/\|\mathbf b\|}$$
#
# 兩個重要補充：
#
# * **最近的奇異矩陣**距離 $A$ 恰好 $\sigma_{\min}$（不是 $\lambda_{\min}$！）
# * **單個特徵值**的條件數是 $1/|\mathbf y^{\mathsf T}\mathbf x|=1/|\cos\theta|$，
#   其中 $\mathbf x,\mathbf y$ 是右／左特徵向量。對稱矩陣 $\mathbf y=\mathbf x$
#   $\Rightarrow$ 條件數 1（完美）

# %%
section("13.6 條件數：相對誤差的放大倍數")

print(f"{'矩陣':<26} | {'κ(A)':>12} | {'最壞情況誤差放大':>18}")
print("-" * 62)
rg = np.random.default_rng(1)
for name, Mx in [("正交矩陣 Q", np.linalg.qr(rg.standard_normal((5, 5)))[0]),
                 ("隨機 5x5", rg.standard_normal((5, 5))),
                 ("Hilbert 5x5", np.array([[1 / (i + j + 1) for j in range(5)]
                                           for i in range(5)])),
                 ("Vandermonde 8x8", np.vander(np.linspace(0, 1, 8)))]:
    kappa = condition_number(Mx)
    worst = 0.0
    x0 = np.linalg.solve(Mx, np.ones(Mx.shape[0]))
    b0 = Mx @ x0
    for _ in range(3000):
        db = 1e-8 * rg.standard_normal(Mx.shape[0])
        dx = np.linalg.solve(Mx, db)
        worst = max(worst, (np.linalg.norm(dx) / np.linalg.norm(x0))
                    / (np.linalg.norm(db) / np.linalg.norm(b0)))
    print(f"{name:<26} | {kappa:>12.4g} | {worst:>18.4g}")
    check(f"  {name}：實測放大 ≤ κ(A)", worst <= kappa * 1.01)
check("正交矩陣的 κ = 1（最完美）",
      condition_number(np.linalg.qr(rg.standard_normal((5, 5)))[0]), 1.0, tol=1e-8)

# 最近的奇異矩陣
A6 = np.array([[3.0, 1.0], [1.0, 2.0]])
U6, s6, Vt6 = np.linalg.svd(A6)
A_singular = U6 @ np.diag([s6[0], 0.0]) @ Vt6
print(f"\nA 的 σ = {s6}，λ = {np.linalg.eigvalsh(A6)}")
print(f"最近的奇異矩陣距離 = ‖A − A_sing‖₂ = {np.linalg.norm(A6 - A_singular, 2):.6f}")
check("距離 = σ_min（不是 λ_min！）", np.linalg.norm(A6 - A_singular, 2), s6[-1])
check("A_sing 真的奇異", abs(np.linalg.det(A_singular)) < 1e-12)
worse = all(np.linalg.norm(A6 - (A6 + 0.9 * s6[-1] * rg.standard_normal((2, 2))
                                 / np.linalg.norm(rg.standard_normal((2, 2)), 2)), 2)
            < s6[-1] * 1.01 for _ in range(10))
print(f"  （λ_min = {np.linalg.eigvalsh(A6)[0]:.6f} ≠ σ_min = {s6[-1]:.6f}）")

# 特徵值的條件數
print("\n單個特徵值的條件數 1/|yᵀx|：")
for name, Mx in [("對稱矩陣", np.array([[2.0, 1.0], [1.0, 3.0]])),
                 ("接近缺陷的矩陣", np.array([[1.0, 1000.0], [0.0, 1.0001]]))]:
    lam, X = np.linalg.eig(Mx)
    lam_l, Y = np.linalg.eig(Mx.T)
    conds = []
    for i in range(len(lam)):
        j = int(np.argmin(np.abs(lam_l - lam[i])))
        x_v = X[:, i] / np.linalg.norm(X[:, i])
        y_v = Y[:, j] / np.linalg.norm(Y[:, j])
        conds.append(1 / abs(y_v @ x_v))
    print(f"  {name:<16} λ = {np.round(np.real(lam), 6)}，條件數 = {np.round(conds, 4)}")
    if name == "對稱矩陣":
        check("  對稱矩陣的特徵值條件數 = 1（完美）", np.allclose(conds, 1.0))
    else:
        check("  接近缺陷的矩陣條件數很大", max(conds) > 100)

# %% [markdown]
# ## 13.7　附錄 8：Markov 矩陣與 Perron–Frobenius
#
# Markov 矩陣：所有 $a_{ij}\ge0$ 且**每欄和為 1**。
#
# * $\lambda=1$ 永遠是特徵值（因為 $\mathbf 1^{\mathsf T}M=\mathbf 1^{\mathsf T}$）
# * 所有 $|\lambda|\le1$
# * **Perron–Frobenius**：若 $M>0$（全正），則 $\lambda=1$ 是**唯一**的最大特徵值，
#   對應的特徵向量可取成全正 ── 這就是唯一的穩態機率分布
# * 收斂速度由第二大的 $|\lambda_2|$ 決定
#
# 這就是 Google PageRank 的數學基礎。

# %%
section("13.7 Markov 矩陣與穩態")

P = np.array([[0.8, 0.3, 0.2],
              [0.1, 0.4, 0.3],
              [0.1, 0.3, 0.5]])
show_matrix("Markov 矩陣 P", P)
check("每欄和為 1", P.sum(axis=0), np.ones(3))
check("所有元素 ≥ 0", np.all(P >= 0))

lam = np.linalg.eigvals(P)
idx = np.argsort(-np.abs(lam))
lam = lam[idx]
print(f"特徵值 = {np.round(np.real(lam), 6)}")
check("λ = 1 是特徵值", np.min(np.abs(lam - 1.0)) < 1e-12)
check("所有 |λ| ≤ 1", np.all(np.abs(lam) <= 1 + 1e-12))
check("1ᵀP = 1ᵀ（這就是 λ=1 的原因）", np.ones(3) @ P, np.ones(3))

pi = markov_steady_state(P)
print(f"\n穩態分布 π = {np.round(pi, 8)}（和 = {pi.sum():.6f}）")
check("Pπ = π", P @ pi, pi)
check("π 全為正（Perron–Frobenius）", np.all(pi > 0))
check("π 是機率分布", float(pi.sum()), 1.0)
check("冪法找到的 λ = 1", power_iteration(P)[0], 1.0, tol=1e-10)

print(f"\n收斂速度由 |λ₂| = {abs(lam[1]):.6f} 決定：")
x = np.array([1.0, 0.0, 0.0])
print(f"{'k':>4} | {'x_k':<34} | {'‖x_k − π‖':>12} | {'|λ₂|ᵏ':>12}")
print("-" * 70)
for k in range(9):
    err = np.linalg.norm(x - pi)
    print(f"{k:>4} | {str(np.round(x, 6)):<34} | {err:>12.3e} | "
          f"{abs(lam[1]) ** k:>12.3e}")
    x = P @ x
errs = []
x = np.array([1.0, 0.0, 0.0])
for k in range(20):
    errs.append(np.linalg.norm(x - pi))
    x = P @ x
ratio = errs[15] / errs[14]
print(f"\n實測收斂比 = {ratio:.6f}，|λ₂| = {abs(lam[1]):.6f}")
check("誤差以 |λ₂| 的速度幾何衰減", ratio, abs(lam[1]), tol=1e-4)

# PageRank
section("13.7 PageRank = 阻尼後的 Markov 矩陣")

links = {0: [1, 2], 1: [2], 2: [0], 3: [0, 1, 2]}      # 節點 3 沒有被連入
n_pages = 4
A_link = np.zeros((n_pages, n_pages))
for src, dsts in links.items():
    for d in dsts:
        A_link[d, src] = 1.0 / len(dsts)
d_factor = 0.85
G = d_factor * A_link + (1 - d_factor) / n_pages * np.ones((n_pages, n_pages))
check("Google 矩陣是 Markov 矩陣（每欄和 1）", G.sum(axis=0), np.ones(n_pages))
check("且所有元素 > 0 ⇒ Perron–Frobenius 保證唯一穩態", np.all(G > 0))
pr = markov_steady_state(G)
print(f"PageRank = {np.round(pr, 6)}")
print(f"排名（高到低）= {np.argsort(-pr)}")
check("PageRank 是 λ=1 的特徵向量", G @ pr, pr, tol=1e-10)
check("全為正", np.all(pr > 0))
lam_G = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]
lam_link = np.sort(np.abs(np.linalg.eigvals(A_link)))[::-1]
print(f"|λ₂(G)| = {lam_G[1]:.6f}，而 d·|λ₂(連結矩陣)| = "
      f"{d_factor * lam_link[1]:.6f}")
check("|λ₂(G)| = d·|λ₂(連結矩陣)|", lam_G[1], d_factor * lam_link[1], tol=1e-8)
check("因此 |λ₂(G)| ≤ d（阻尼係數是收斂速度的保證上界）",
      lam_G[1] <= d_factor + 1e-12)
n_iter = int(np.ceil(np.log(1e-8) / np.log(max(lam_G[1], 1e-12))))
print(f"  ⇒ 誤差每步乘 {lam_G[1]:.3f}，約 {n_iter} 次迭代就能到 1e−8")
print("  （Google 用 d = 0.85 的理由之一：保證 |λ₂| ≤ 0.85，約 50 次迭代收斂）")

# %% [markdown]
# ## 13.8　附錄 10：計算機圖學的齊次座標
#
# 問題：平移 $\mathbf v\mapsto\mathbf v+\mathbf t$ **不是**線性的。
#
# 解法：升一個維度（**齊次座標**）。把 $(x,y,z)$ 寫成 $(x,y,z,1)$，則
#
# $$\begin{bmatrix}R&\mathbf t\\\mathbf 0^{\mathsf T}&1\end{bmatrix}
# \begin{bmatrix}\mathbf v\\1\end{bmatrix}
# =\begin{bmatrix}R\mathbf v+\mathbf t\\1\end{bmatrix}$$
#
# **affine 轉換變成 4 維的線性轉換** ── 於是旋轉、縮放、平移、透視投影
# 全部可以用 $4\times4$ 矩陣相乘串起來。這就是所有 3D 圖學管線的基礎。

# %%
section("13.8 齊次座標：讓平移變成線性")


def translation(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def scaling(s):
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = s
    return S


def rotation_z(theta):
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def perspective(d):
    """簡單的透視投影：z 越遠物體越小（第四個分量不再是 1）。"""
    P = np.eye(4)
    P[3, 2] = 1.0 / d
    return P


def to_homogeneous(V):
    return np.vstack([V, np.ones((1, V.shape[1]))])


def from_homogeneous(Vh):
    return Vh[:3] / Vh[3]


v = np.array([[1.0], [2.0], [3.0]])
t = np.array([10.0, -5.0, 2.0])
T = translation(t)
print(f"點 v = {v.ravel()}，平移 t = {t}")
check("齊次座標下平移是線性的：T·(v,1) = (v+t, 1)",
      from_homogeneous(T @ to_homogeneous(v)).ravel(), (v.ravel() + t))
check("3x3 的世界裡平移做不到（沒有矩陣 A 使 Av = v + t 對所有 v 成立）",
      not np.allclose(np.zeros((3, 3)) @ v, v + t.reshape(-1, 1)))

# 組合變換：先旋轉再平移 vs 先平移再旋轉
R = rotation_z(np.radians(90))
print(f"\n順序很重要（矩陣不可交換）：")
p1 = from_homogeneous(T @ R @ to_homogeneous(v)).ravel()
p2 = from_homogeneous(R @ T @ to_homogeneous(v)).ravel()
print(f"  先旋轉再平移 = {np.round(p1, 4)}")
print(f"  先平移再旋轉 = {np.round(p2, 4)}")
check("TR ≠ RT", not np.allclose(p1, p2))
check("反向變換 T⁻¹ = translation(−t)", np.linalg.inv(T), translation(-t))
check("旋轉的反矩陣 = 轉置（上 3x3 部分）",
      np.linalg.inv(R)[:3, :3], R[:3, :3].T)

# 完整的圖學管線
Scale = scaling([2.0, 2.0, 2.0])
Persp = perspective(10.0)
pipeline = Persp @ T @ R @ Scale
print(f"\n完整管線 = 透視 · 平移 · 旋轉 · 縮放（一個 4x4 矩陣！）")
cube = np.array([[0.0, 1, 1, 0, 0, 1, 1, 0],
                 [0.0, 0, 1, 1, 0, 0, 1, 1],
                 [0.0, 0, 0, 0, 1, 1, 1, 1]])
out = from_homogeneous(pipeline @ to_homogeneous(cube))
step = from_homogeneous(Persp @ (T @ (R @ (Scale @ to_homogeneous(cube)))))
check("一次矩陣乘法 = 逐步套用（結合律）", out, step)
print(f"  立方體 8 個頂點變換後的 x 範圍 = "
      f"[{out[0].min():.3f}, {out[0].max():.3f}]")

# 透視投影讓平行線交於消失點
print("\n透視投影的效果：相同大小的物體，越遠越小")
for z in [5.0, 10.0, 20.0, 40.0]:
    square = np.array([[-1.0, 1, 1, -1], [-1.0, -1, 1, 1], [z, z, z, z]])
    proj = from_homogeneous(perspective(10.0) @ to_homogeneous(square))
    width = proj[0].max() - proj[0].min()
    print(f"  z = {z:>5}：投影後寬度 = {width:.4f}（理論 2/(1+z/d) = "
          f"{2 / (1 + z / 10.0):.4f}）")
    check(f"  z = {z} 的投影寬度符合 2/(1+z/d)", width, 2 / (1 + z / 10.0))

fig = plt.figure(figsize=(10.5, 3.6))
house = np.array([[-6.0, -6, -7, 0, 7, 6, 6, -3, -3, 0, 0, -6],
                  [-7.0, 2, 1, 8, 1, 2, -7, -7, -2, -2, -7, -7],
                  [0.0] * 12])
for k, (name, Mx) in enumerate([
        ("original", np.eye(4)),
        ("translate + rotate", translation([8.0, 3.0, 0.0]) @ rotation_z(np.radians(30))),
        ("rotate + scale + translate",
         translation([-4.0, 6.0, 0.0]) @ scaling([0.6, 1.4, 1.0]) @ rotation_z(np.radians(-20)))]):
    ax = fig.add_subplot(1, 3, k + 1)
    pts = from_homogeneous(Mx @ to_homogeneous(house))
    ax.plot(np.append(pts[0], pts[0, 0]), np.append(pts[1], pts[1, 0]), "C0-", lw=1.6)
    ax.set_title(name, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-14, 16)
    ax.set_ylim(-12, 14)
    ax.grid(alpha=0.3)
finish(fig, "ch13_graphics")

# %% [markdown]
# ## 13.9　附錄 4：真實世界的程式庫
#
# | 任務 | 業界標準 |
# |---|---|
# | 稠密線性代數 | **LAPACK**（底層 BLAS） |
# | 超大規模 | ScaLAPACK、Trilinos |
# | 稀疏直接法 | SuiteSparse（UMFPACK）、SuperLU |
# | 迭代法 | 共軛梯度、GMRES（Trilinos、PETSc）|
# | 特徵值 | LAPACK 的位移 QR、ARPACK（大型稀疏）|
# | SVD | Golub–Kahan（LAPACK）|
# | 最佳化 | COIN/OR、CVXPY |
# | FFT | FFTW |
#
# NumPy / SciPy 都是 LAPACK 的包裝。下面比較「教學版實作」與 LAPACK 的速度 ──
# 教學版讓我們看懂演算法，LAPACK 讓我們真的算得動。

# %%
section("13.9 教學版實作 vs LAPACK")

print(f"{'n':>5} | {'自製 plu (s)':>14} | {'LAPACK solve (s)':>18} | {'倍數':>8}")
print("-" * 54)
rg = np.random.default_rng(0)
for n in [100, 200, 400]:
    Mx = rg.standard_normal((n, n))
    bvec = rg.standard_normal(n)
    t0 = time.perf_counter()
    plu(Mx)
    t_own = time.perf_counter() - t0
    t0 = time.perf_counter()
    np.linalg.solve(Mx, bvec)
    t_lapack = time.perf_counter() - t0
    print(f"{n:>5} | {t_own:>14.5f} | {t_lapack:>18.5f} | "
          f"{t_own / max(t_lapack, 1e-9):>7.0f}x")
print("\n差距來自：快取最佳化的分塊演算法、BLAS-3 矩陣乘法、SIMD、多執行緒。")
print("但兩者解出的答案相同 —— 數學是一樣的：")
Mx = rg.standard_normal((80, 80))
bvec = rg.standard_normal(80)
from linalg_tutorial.elimination import solve as own_solve
check("自製 solve 與 numpy.linalg.solve 結果一致",
      own_solve(Mx, bvec), np.linalg.solve(Mx, bvec), tol=1e-8)
print(f"numpy 使用的 LAPACK 實作：{np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name']}"
      if hasattr(np.__config__, "show") else "")

# %% [markdown]
# ## 動手練習
#
# 1. 找出一個例子使 $\operatorname{rank}(AB)<\min(\operatorname{rank}A,\operatorname{rank}B)$。
# 2. 驗證 $A=QR$ 與 $A=LU$ 的參數計數在 $n=6$ 時都等於 36。
# 3. 用 Cayley–Hamilton 把 $A^{10}$ 寫成 $A^2,A,I$ 的線性組合（$3\times3$ 矩陣）。
# 4. 造一個 Markov 矩陣，其 $|\lambda_2|$ 接近 1，觀察收斂變得多慢。
# 5. 用齊次座標寫出「繞任意軸旋轉」= 平移到原點 → 旋轉 → 平移回去。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A_e = np.array([[1.0, 0.0], [0.0, 0.0]])
B_e = np.array([[0.0, 0.0], [1.0, 0.0]])
print(f"練習 1：rank A = {rank(A_e)}, rank B = {rank(B_e)}, "
      f"rank(AB) = {rank(A_e @ B_e)}")
check("  嚴格小於 min", rank(A_e @ B_e) < min(rank(A_e), rank(B_e)))
print("  原因：B 的欄空間落在 A 的零空間裡")

# 練習 2
n = 6
check("練習 2：L+U 的參數 = 36", n * (n - 1) // 2 + n * (n + 1) // 2, n * n)
check("          Q+R 的參數 = 36", n * (n - 1) // 2 + n * (n + 1) // 2, n * n)
print(f"  L: {n*(n-1)//2}, U: {n*(n+1)//2}, Q: {n*(n-1)//2}, R: {n*(n+1)//2}，"
      f"總和都是 {n*n}")

# 練習 3
A3 = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0]])
coef3 = np.poly(A3)                      # λ³ + c₁λ² + c₂λ + c₃
# 用 A³ = −c₁A² − c₂A − c₃I 反覆降次
powers = [np.eye(3), A3, A3 @ A3]
reduce_coeffs = np.array([-coef3[3], -coef3[2], -coef3[1]])   # A³ 的表示 [I, A, A²]
rep = np.array([0.0, 0.0, 1.0])          # 現在表示 A²
for _ in range(8):                        # 一路乘到 A¹⁰
    new = np.zeros(3)
    new[1:] += rep[:2]                    # 乘 A：I→A, A→A²
    new += rep[2] * reduce_coeffs         # A²·A = A³ → 用降次公式
    rep = new
A10_poly = rep[0] * powers[0] + rep[1] * powers[1] + rep[2] * powers[2]
print(f"\n練習 3：A¹⁰ = {rep[2]:.0f}·A² + {rep[1]:.0f}·A + {rep[0]:.0f}·I")
check("  與直接計算 A¹⁰ 相同", A10_poly, np.linalg.matrix_power(A3, 10), tol=1e-6)

# 練習 4
print("\n練習 4：|λ₂| 接近 1 的 Markov 矩陣收斂很慢")
for eps in [0.3, 0.05, 0.005]:
    Pm = np.array([[1 - eps, eps], [eps, 1 - eps]])
    lam2 = sorted(np.abs(np.linalg.eigvals(Pm)))[0]
    x = np.array([1.0, 0.0])
    pi_m = np.array([0.5, 0.5])
    k = 0
    while np.linalg.norm(x - pi_m) > 1e-6 and k < 100000:
        x = Pm @ x
        k += 1
    print(f"  ε = {eps:<6} |λ₂| = {lam2:.6f} ⇒ 需要 {k:>6} 步才收斂到 1e−6")
    check(f"  ε = {eps}：|λ₂| = 1 − 2ε", lam2, 1 - 2 * eps, tol=1e-10)

# 練習 5
print("\n練習 5：繞過點 p 的軸旋轉 = T(p)·R·T(−p)")
p_axis = np.array([3.0, 4.0, 0.0])
R_about = translation(p_axis) @ rotation_z(np.radians(90)) @ translation(-p_axis)
check("  旋轉中心是不動點", from_homogeneous(R_about @ to_homogeneous(
    p_axis.reshape(-1, 1))).ravel(), p_axis)
v_test = np.array([[4.0], [4.0], [0.0]])
out5 = from_homogeneous(R_about @ to_homogeneous(v_test)).ravel()
print(f"  點 (4,4,0) 繞 (3,4,0) 轉 90° → {np.round(out5, 4)}（理論 (3,5,0)）")
check("  結果正確", out5, np.array([3.0, 5.0, 0.0]))
check("  旋轉保持與中心的距離",
      np.linalg.norm(out5 - p_axis), np.linalg.norm(v_test.ravel() - p_axis))

# %% [markdown]
# ## 本章重點回顧
#
# * **秩的規則**：$\operatorname{rank}(AB)\le\min$、
#   $\operatorname{rank}(A+B)\le$ 和、
#   $\operatorname{rank}(A^{\mathsf T}A)=\operatorname{rank}A$。
#   「$m\times r$ 乘 $r\times n$ 且都滿秩」才保證乘積仍是秩 $r$。
# * **十六個分解**全部驗證過一次；SVD 與 Schur 是唯一對**任意**矩陣都成立的兩個。
# * **參數計數**是檢驗理解的好工具：$A=LU$、$A=QR$ 兩邊都是 $n^2$；
#   秩 $r$ 矩陣只有 $(m+n-r)r$ 個自由度（低秩壓縮的理論基礎）。
# * **Cayley–Hamilton**：$p(A)=0$，因此 $A^{-1}$ 與 $A^{k}$ 都能寫成 $A$ 的低次多項式。
# * **張量**沒有 SVD：只能用 CP 分解 + 交替最小平方，因子一般不正交。
# * **條件數** $\kappa=\sigma_{\max}/\sigma_{\min}$ 是相對誤差的放大上限；
#   最近的奇異矩陣距離 $\sigma_{\min}$；對稱矩陣的特徵值條件數恰好是 1。
# * **Markov 與 Perron–Frobenius**：$\lambda=1$ 永遠存在、全正矩陣的穩態唯一且全正，
#   收斂速度由 $|\lambda_2|$ 決定（PageRank 的阻尼係數就是 $|\lambda_2|$）。
# * **齊次座標**把 affine 轉換升級成線性轉換 ── 整個 3D 圖學管線就是 $4\times4$ 矩陣的乘積。
#
# 下一章起進入第二部分：以 Riley《Mathematical Methods》的視角，
# 看線性代數如何貫穿物理與工程的每一個角落。
