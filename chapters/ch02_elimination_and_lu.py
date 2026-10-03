# %% [markdown]
# # 第 2 章　解 $A\mathbf x=\mathbf b$：消去法、反矩陣與 $A=LU$
#
# > 對應 Strang 第 2 章（2.1 消去與回代、2.2 消去矩陣與反矩陣、2.3 $A=LU$ 與計算成本、
# > 2.4 置換與轉置、2.5 導數與有限差分矩陣），以及 Riley 8.10（反矩陣）、8.18、27.3。
#
# 核心流程只有三步：
#
# $$A\mathbf x=\mathbf b
# \ \xrightarrow{\ \text{消去}\ }\ U\mathbf x=\mathbf c
# \ \xrightarrow{\ \text{回代}\ }\ \mathbf x$$
#
# 把每一步寫成矩陣，就得到線性代數裡最實用的分解
#
# $$PA = LU$$
#
# **執行方式**
#
# ```bash
# python chapters/ch02_elimination_and_lu.py
# python tools/build_notebooks.py ch02
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

from linalg_tutorial.elimination import (
    back_substitution,
    cholesky,
    difference_matrix,
    elimination_matrix,
    forward_substitution,
    inverse_gauss_jordan,
    ldu,
    lu_no_pivot,
    plu,
    second_difference_matrix,
    solve,
)
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 2.1　消去與回代
#
# 消去法對方程做兩種**可逆**操作：把某式乘上非零數、把某式減去另一式的倍數。
# 解集合完全不變，但矩陣慢慢變成上三角 $U$。
#
# 消去乘數 $\ell_{ij}=\dfrac{a_{ij}}{\text{主元}}$，目標是在 $(i,j)$ 位置製造 0。
#
# $n\times n$ 的三種結局，全部寫在 $U$ 的對角線上：
#
# | $U$ 的對角線 | 意義 |
# |---|---|
# | $n$ 個非零主元 | 恰一組解，$A$ 可逆 |
# | 出現 0 主元但換列可救 | 仍可逆，需要 $P$ |
# | 換列也救不了 | 欄相依，$A$ 奇異（無解或無窮多解）|

# %%
section("2.1 從 A 到 U：逐欄消去（含右側 b）")

A = np.array([
    [2.0, 3.0, 4.0],
    [4.0, 11.0, 14.0],
    [2.0, 8.0, 17.0],
])
b = np.array([19.0, 55.0, 50.0])


def eliminate_verbose(A, b):
    """把 [A b] 做前向消去，一步一步印出來（Strang 2.1 的手算流程）。"""
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    n = A.shape[0]
    mults = {}
    for k in range(n):
        print(f"\n  --- 第 {k + 1} 欄，主元 = {M[k, k]:.4g} ---")
        for i in range(k + 1, n):
            if M[i, k] == 0:
                continue
            ell = M[i, k] / M[k, k]
            mults[(i, k)] = ell
            M[i, :] -= ell * M[k, :]
            print(f"  列{i + 1} ← 列{i + 1} − ({ell:.4g})·列{k + 1}   得 {M[i, :]}")
    return M[:, :n], M[:, n], mults


U, c, mults = eliminate_verbose(A, b)
show_matrix("U", U)
show_matrix("c", c)
print(f"\n主元 = {np.diag(U)}（三個都不是 0 → 恰有一組解）")

x = back_substitution(U, c)
print(f"\n回代：x3 = {x[2]:.4g} → x2 = {x[1]:.4g} → x1 = {x[0]:.4g}")
check("Ax = b", A @ x, b)
check("與 numpy.linalg.solve 一致", x, np.linalg.solve(A, b))
check("自製 solve()（PA=LU + 兩次回代）一致", solve(A, b), x)

# %% [markdown]
# ### 列觀點與欄觀點的幾何
#
# * **列觀點**：每一個方程是一條直線（或平面），解 = 交點。平行 → 無解；重合 → 一條解線。
# * **欄觀點**：把 $A$ 的欄組合出 $\mathbf b$。欄獨立 → 一定成功且方式唯一。
#
# 維度一高，列觀點（超平面相交）就畫不出來了，欄觀點仍然清楚──這是 Strang 一再強調的。

# %%
fig, _ = new_axes("", figsize=(10.2, 3.6), equal=False)
fig.clf()
cases = [
    (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([2.0, 5.0]), "no solution (parallel)"),
    (np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([2.0, 0.0]), "one solution"),
    (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([2.0, 4.0]), "a line of solutions"),
]
xs = np.linspace(-1, 3, 50)
for k, (M, rhs, title) in enumerate(cases):
    ax = fig.add_subplot(1, 3, k + 1)
    for i, color in zip(range(2), ["C0", "C3"]):
        a1, a2 = M[i]
        if a2 != 0:
            ax.plot(xs, (rhs[i] - a1 * xs) / a2, color=color, lw=2,
                    label=f"{a1:g}x+{a2:g}y={rhs[i]:g}")
        else:
            ax.axvline(rhs[i] / a1, color=color, lw=2)
    ax.set_title(f"{title}\nrank={np.linalg.matrix_rank(M)}", fontsize=9)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
finish(fig, "ch02_row_picture")

# %% [markdown]
# ## 2.2　消去矩陣與反矩陣
#
# 「第 $i$ 列減去 $\ell$ 倍第 $j$ 列」這個動作就是左乘
#
# $$E_{ij} = I - \ell\,\mathbf e_i\mathbf e_j^{\mathsf T},
# \qquad E_{ij}^{-1} = I + \ell\,\mathbf e_i\mathbf e_j^{\mathsf T}$$
#
# 整個消去是 $EA=U$，其中 $E=E_{32}E_{31}E_{21}$。
# 關鍵觀察：$E$ 本身很亂（會出現 $\ell_{32}\ell_{21}-\ell_{31}$ 這種項），
# 但它的**反矩陣按相反順序相乘**卻完美：
#
# $$L=E^{-1}=E_{21}^{-1}E_{31}^{-1}E_{32}^{-1}
# =\begin{bmatrix}1&&\\ \ell_{21}&1&\\ \ell_{31}&\ell_{32}&1\end{bmatrix}$$
#
# 每個乘數 $\ell_{ij}$ 都乖乖坐在自己的位置上。這就是 $A=LU$。

# %%
section("2.2 消去矩陣 E 與它的反矩陣")

n = 3
E21 = elimination_matrix(n, 1, 0, mults[(1, 0)])
E31 = elimination_matrix(n, 2, 0, mults[(2, 0)])
E32 = elimination_matrix(n, 2, 1, mults[(2, 1)])
E = E32 @ E31 @ E21

show_matrix("E21", E21)
show_matrix("E = E32·E31·E21", E)
check("E·A = U", E @ A, U)

L_from_inverse = np.linalg.inv(E)
show_matrix("L = E⁻¹（乘數各就各位）", L_from_inverse)
check("反矩陣按相反順序：E21⁻¹E31⁻¹E32⁻¹ = L",
      np.linalg.inv(E21) @ np.linalg.inv(E31) @ np.linalg.inv(E32), L_from_inverse)
print("注意 E 的左下角是 ℓ32·ℓ21 − ℓ31 的混合項，而 L 乾淨：這就是我們用 L 而不用 E 的理由")

# %% [markdown]
# ### 反矩陣的七個要點（Strang 2.2）
#
# 1. $A^{-1}$ 存在 $\iff$ 消去（允許換列）得到 $n$ 個非零主元
# 2. 反矩陣唯一；左反矩陣 = 右反矩陣（結合律證明）
# 3. $A$ 可逆時 $A\mathbf x=\mathbf b$ 的唯一解是 $\mathbf x=A^{-1}\mathbf b$
# 4. 若存在 $\mathbf x\ne 0$ 使 $A\mathbf x=\mathbf 0$，則 $A$ 不可逆
# 5. 方陣可逆 $\iff$ 各欄獨立
# 6. $2\times2$：$\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}
#    =\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$
# 7. 三角矩陣可逆 $\iff$ 對角線無 0；**反矩陣順序相反**：$(AB)^{-1}=B^{-1}A^{-1}$

# %%
section("2.2 Gauss–Jordan 求反矩陣與反矩陣法則")

D = difference_matrix(4)          # 差分矩陣（下三角）
show_matrix("A = 差分矩陣", D)
Dinv = inverse_gauss_jordan(D)
show_matrix("A⁻¹ = 累加（求和）矩陣", Dinv)
check("A·A⁻¹ = I", D @ Dinv, np.eye(4))
check("與 numpy.linalg.inv 一致", Dinv, np.linalg.inv(D))
print("  差分的反運算是累加 —— 這是微積分基本定理的線性代數版本")

M1 = np.array([[1.0, 2.0], [3.0, 7.0]])
M2 = np.array([[2.0, 0.0], [1.0, 3.0]])
check("(AB)⁻¹ = B⁻¹A⁻¹（順序相反）",
      np.linalg.inv(M1 @ M2), np.linalg.inv(M2) @ np.linalg.inv(M1))
check("(Aᵀ)⁻¹ = (A⁻¹)ᵀ", np.linalg.inv(M1.T), np.linalg.inv(M1).T)
singular = np.array([[1.0, 3.0], [2.0, 6.0]])
check("奇異矩陣：存在 x≠0 使 Ax=0", singular @ np.array([3.0, -1.0]), np.zeros(2))

# %% [markdown]
# ## 2.3　$A=LU$ 與計算成本
#
# $$A = LU \qquad
# L=\text{（單位對角的下三角，裝乘數）},\quad
# U=\text{（上三角，裝主元）}$$
#
# **第二種證明（欄 × 列）** 特別漂亮：每一步消去都是從 $A$ 中減掉一個秩一矩陣
# 「$L$ 的一欄 × $U$ 的一列」：
#
# $$A=\boldsymbol\ell_1\mathbf u_1+\boldsymbol\ell_2\mathbf u_2+\dots+\boldsymbol\ell_n\mathbf u_n=LU$$
#
# **成本**：$A\to U$ 約 $\tfrac13 n^3$ 次乘法；每個新的右側 $\mathbf b$ 只要 $n^2$。
# 所以要解很多個右側時，先存下 $LU$ 再回代，遠比重算或求 $A^{-1}$ 便宜。

# %%
section("2.3 A = LU 與三種讀法")

L, U2 = lu_no_pivot(A)
show_matrix("L", L)
show_matrix("U", U2)
check("A = L·U", L @ U2, A)
check("U 與前面手算的 U 相同", U2, U)

print("\n欄×列觀點：A = Σ (L 的第 k 欄)(U 的第 k 列)")
acc = np.zeros_like(A)
for k in range(3):
    piece = np.outer(L[:, k], U2[k, :])
    acc += piece
    print(f"  第 {k + 1} 個秩一矩陣，累積誤差 ‖A − Σ‖ = {np.linalg.norm(A - acc):.3g}")
check("秩一矩陣加總 = A", acc, A)

Lp, Dp, Up = ldu(A)
show_matrix("D（主元）", Dp)
check("A = L·D·U'（把主元抽出來）", Lp @ Dp @ Up, A)

# %% [markdown]
# ### 成本實測：$\tfrac13 n^3$ 的 $n^3$ 律
#
# 矩陣尺寸加倍，消去時間大約變成 8 倍（$2^3$）。下面用自己寫的 `plu` 實測。

# %%
section("2.3 消去成本的 n³ 律")

print(f"{'n':>6} | {'自製 plu 時間(s)':>16} | {'相對前一列倍數':>14} | {'理論 n³/3 次乘法':>18}")
print("-" * 68)
rng = np.random.default_rng(0)
prev_t = None
for n_ in [40, 80, 160, 320]:
    M = rng.standard_normal((n_, n_))
    t0 = time.perf_counter()
    plu(M)
    t = time.perf_counter() - t0
    ratio = f"{t / prev_t:.1f}x" if prev_t else "—"
    print(f"{n_:>6} | {t:>16.4f} | {ratio:>14} | {n_ ** 3 / 3:>18,.0f}")
    prev_t = t

# 多個右側：LU 重複使用 vs 每次重新分解
n_ = 200
M = rng.standard_normal((n_, n_))
B = rng.standard_normal((n_, 30))
P_, L_, U_ = plu(M)
t0 = time.perf_counter()
X1 = np.column_stack([back_substitution(U_, forward_substitution(L_, P_ @ B[:, j]))
                      for j in range(B.shape[1])])
t_reuse = time.perf_counter() - t0
t0 = time.perf_counter()
X2 = np.column_stack([solve(M, B[:, j]) for j in range(B.shape[1])])
t_redo = time.perf_counter() - t0
print(f"\n30 個右側：重複使用 LU {t_reuse:.3f}s vs 每次重新分解 {t_redo:.3f}s"
      f"（快 {t_redo / max(t_reuse, 1e-9):.0f} 倍）")
check("兩種做法答案相同", X1, X2, tol=1e-8)

# %% [markdown]
# ## 2.4　置換與轉置
#
# ### 置換矩陣 $P$
#
# * $P$ 的每列每欄都恰有一個 1，共 $n!$ 個
# * $P^{-1}=P^{\mathsf T}$（欄正交）
# * 主元遇到 0 時換列：$PA=LU$
# * **部分軸選取**（partial pivoting）：即使主元不是 0，也挑該欄絕對值最大的當主元，
#   以降低捨入誤差。這是所有數值程式庫的標準做法。

# %%
section("2.4 PA = LU 與部分軸選取")

A2 = np.array([
    [1.0, 2.0, 5.0],
    [2.0, 4.0, 7.0],      # 第 2 列是第 1 列的 2 倍（前兩個元素）→ 主元會變 0
    [3.0, 7.0, 9.0],
])
P, Lp2, Up2 = plu(A2)
show_matrix("P", P)
show_matrix("L", Lp2)
show_matrix("U", Up2)
check("P·A = L·U", P @ A2, Lp2 @ Up2)
check("P⁻¹ = Pᵀ", np.linalg.inv(P), P.T)
check("L 的所有元素 ≤ 1（部分軸選取的保證）", np.all(np.abs(Lp2) <= 1 + 1e-12))
try:
    lu_no_pivot(A2)
except ValueError as e:
    print(f"  不換列的 LU 直接失敗：{e}")

# 捨入誤差示範：小主元會放大誤差
eps = 1e-16
bad = np.array([[eps, 1.0], [1.0, 1.0]])
rhs = np.array([1.0, 2.0])
x_nopivot_U = np.array([[eps, 1.0], [0.0, 1.0 - 1.0 / eps]])
x_nopivot_c = np.array([1.0, 2.0 - 1.0 / eps])
x_bad = back_substitution(x_nopivot_U, x_nopivot_c)
x_good = solve(bad, rhs)
print(f"\n小主元 ε = {eps:g}：")
print(f"  不換列  x = {x_bad}      殘差 ‖Ax−b‖ = {np.linalg.norm(bad @ x_bad - rhs):.3g}")
print(f"  部分軸選取 x = {x_good}  殘差 ‖Ax−b‖ = {np.linalg.norm(bad @ x_good - rhs):.3g}")

# %% [markdown]
# ### 轉置 $A^{\mathsf T}$ 的真正定義
#
# 把矩陣沿對角線翻過來只是「操作」，真正的定義來自內積：
#
# $$(A\mathbf x)^{\mathsf T}\mathbf y=\mathbf x^{\mathsf T}(A^{\mathsf T}\mathbf y)
# \qquad\text{對所有 }\mathbf x,\mathbf y$$
#
# 這個定義在無窮維也成立。例如把 $A=\dfrac{d}{dt}$ 代進去，
# 「分部積分」告訴我們
#
# $$\int \frac{dx}{dt}y\,dt = -\int x\frac{dy}{dt}\,dt
# \quad\Longrightarrow\quad \left(\frac{d}{dt}\right)^{\mathsf T}=-\frac{d}{dt}$$
#
# 微分算子是**反對稱**的。這條線索會在第 15 章（Fourier／Sturm–Liouville）開花。
#
# 規則：$(A+B)^{\mathsf T}=A^{\mathsf T}+B^{\mathsf T}$、
# $(AB)^{\mathsf T}=B^{\mathsf T}A^{\mathsf T}$、
# $(A^{-1})^{\mathsf T}=(A^{\mathsf T})^{-1}$。
# 而 $A^{\mathsf T}A$ 與 $AA^{\mathsf T}$ 永遠對稱。

# %%
section("2.4 轉置的內積定義與對稱矩陣")

Adiff = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
xv = np.array([1.0, 2.0, 4.0])
yv = np.array([3.0, 5.0])
check("(Ax)ᵀy = xᵀ(Aᵀy)", (Adiff @ xv) @ yv, xv @ (Adiff.T @ yv))
check("(AB)ᵀ = BᵀAᵀ", (M1 @ M2).T, M2.T @ M1.T)
check("AᵀA 對稱", Adiff.T @ Adiff, (Adiff.T @ Adiff).T)
check("AAᵀ 對稱", Adiff @ Adiff.T, (Adiff @ Adiff.T).T)
show_matrix("AᵀA（= 二階差分矩陣 K₃！）", Adiff.T @ Adiff)

# 離散版的分部積分：D 的轉置是「負的前向差分」
Dm = difference_matrix(5)
f_ = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
g_ = np.array([2.0, 1.0, 0.0, -1.0, 1.0])
check("離散分部積分 (Df)·g = f·(Dᵀg)", (Dm @ f_) @ g_, f_ @ (Dm.T @ g_))

# 對稱矩陣的 LDLᵗ 與 Cholesky
S = np.array([[4.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]])
Ls, Ds, Us = ldu(S)
check("對稱矩陣：S = L·D·Lᵀ（U' = Lᵀ）", Ls @ Ds @ Ls.T, S)
R = cholesky(S)
check("S = RᵀR（Cholesky）", R.T @ R, S)
print(f"  主元 = {np.diag(Ds)} 全為正 → S 正定（第 6 章會證明這個等價關係）")

# %% [markdown]
# ## 2.5　導數與有限差分矩陣
#
# 把微分換成差分，微分方程就變成矩陣方程。三個基本近似：
#
# $$\frac{dy}{dx}\approx\frac{y(x+h)-y(x)}{h}\ \ (O(h)),\qquad
# \frac{dy}{dx}\approx\frac{y(x+h)-y(x-h)}{2h}\ \ (O(h^2)),$$
#
# $$\frac{d^2y}{dx^2}\approx\frac{y(x+h)-2y(x)+y(x-h)}{h^2}\ \ (O(h^2))$$
#
# 把第三式寫在 $N$ 個網格點上，就得到**科學計算中最重要的矩陣**
#
# $$K=\begin{bmatrix}
# 2&-1&&\\-1&2&-1&\\&-1&2&-1\\&&-1&2\end{bmatrix}
# \qquad\left(\ -\frac{d^2u}{dx^2}=f \ \to\ \frac{1}{h^2}KU=F\ \right)$$
#
# $K$ 的五個性質：**對稱、帶狀（稀疏）、常對角（Toeplitz）、可逆、正定**。
# 邊界條件一變，矩陣就變：
#
# | 矩陣 | 邊界 | 性質 |
# |---|---|---|
# | $K$ | 固定–固定 | 對稱正定、可逆 |
# | $T$ | 自由–固定 | 對稱正定，且 $T=LL^{\mathsf T}$（主元全為 1）|
# | $B$ | 自由–自由 | 奇異！$B\mathbf 1=\mathbf 0$ |

# %%
section("2.5 差分近似的精度")

f = np.sin
fp, fpp = np.cos, lambda t: -np.sin(t)
x0 = 1.0
print(f"{'h':>10} | {'前向誤差':>12} | {'中央誤差':>12} | {'二階差分誤差':>14}")
print("-" * 58)
for h in [1e-1, 1e-2, 1e-3, 1e-4]:
    fwd = (f(x0 + h) - f(x0)) / h
    ctr = (f(x0 + h) - f(x0 - h)) / (2 * h)
    sec = (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h ** 2
    print(f"{h:>10.0e} | {abs(fwd - fp(x0)):>12.3e} | {abs(ctr - fp(x0)):>12.3e} "
          f"| {abs(sec - fpp(x0)):>14.3e}")
print("  前向誤差 ∝ h（一階）；中央與二階差分誤差 ∝ h²（二階）")

# %%
section("2.5 K、T、B 三兄弟")

K = second_difference_matrix(4)
T = K.copy()
T[0, 0] = 1.0              # 自由–固定
B = K.copy()
B[0, 0] = 1.0
B[-1, -1] = 1.0            # 自由–自由
show_matrix("K（固定-固定）", K)
show_matrix("T（自由-固定）", T)
show_matrix("B（自由-自由）", B)

Kinv = inverse_gauss_jordan(K)
show_matrix("K⁻¹（對稱但稠密！）", Kinv * 5)
print("  （上面乘了 5 = det K，方便看出 K⁻¹ 的整數結構 4,3,2,1 / 3,6,4,2 / ...）")
check("K⁻¹K = I", Kinv @ K, np.eye(4))
check("K 的主元全為正（正定）", np.all(np.diag(lu_no_pivot(K)[1]) > 0))
check("T 的主元全為 1", np.diag(lu_no_pivot(T)[1]), np.ones(4))
Lt = lu_no_pivot(T)[0]
check("T = L·Lᵀ", Lt @ Lt.T, T)
ones = np.ones(4)
check("B 奇異：B·(1,1,1,1)ᵀ = 0", B @ ones, np.zeros(4))
print(f"  det K = {np.linalg.det(K):.4g}，det T = {np.linalg.det(T):.4g}，"
      f"det B = {np.linalg.det(B):.4g}")
print("  物理圖像：K 兩端固定、T 一端自由、B 兩端自由（整體可平移 → 奇異）")

# %% [markdown]
# ### 真的解一個微分方程
#
# 解 $-u''(x)=f(x)$，$u(0)=u(1)=0$，取 $f(x)=\pi^2\sin\pi x$，
# 精確解是 $u(x)=\sin\pi x$。只要解 $\dfrac{1}{h^2}KU=F$。
# 誤差應該以 $O(h^2)$ 下降。

# %%
section("2.5 用 K 解兩點邊界值問題")

print(f"{'N':>6} | {'h':>10} | {'最大誤差':>12} | {'誤差比(應≈4)':>14}")
print("-" * 50)
errs, hs = [], []
for N in [8, 16, 32, 64, 128]:
    h = 1.0 / (N + 1)
    xg = np.linspace(h, 1 - h, N)
    KN = second_difference_matrix(N)
    F = np.pi ** 2 * np.sin(np.pi * xg)
    Unum = solve(KN / h ** 2, F)
    exact = np.sin(np.pi * xg)
    err = np.max(np.abs(Unum - exact))
    ratio = f"{errs[-1] / err:.2f}" if errs else "—"
    print(f"{N:>6} | {h:>10.5f} | {err:>12.3e} | {ratio:>14}")
    errs.append(err)
    hs.append(h)
check("誤差以 O(h²) 收斂（h 減半，誤差約為 1/4）", errs[-2] / errs[-1] > 3.5)

fig, ax = new_axes("Finite difference solution of -u'' = f", figsize=(5.6, 4.0), equal=False)
N = 16
h = 1.0 / (N + 1)
xg = np.linspace(h, 1 - h, N)
Unum = solve(second_difference_matrix(N) / h ** 2, np.pi ** 2 * np.sin(np.pi * xg))
xf = np.linspace(0, 1, 300)
ax.plot(xf, np.sin(np.pi * xf), "-", label="exact  u = sin(pi x)")
ax.plot(xg, Unum, "o", ms=5, label=f"K U = F  (N = {N})")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend()
finish(fig, "ch02_bvp_solution")

# %% [markdown]
# ## 動手練習
#
# 1. 對 $A=\begin{bmatrix}1&2&a\\2&4&b\\3&7&c\end{bmatrix}$ 手算消去，說明何時需要換列、
#    何時 $A$ 真的奇異。
# 2. 寫一個函式，用 $LU$ 一次解 $k$ 個右側，並驗證成本為 $\tfrac13n^3+kn^2$。
# 3. 證明 $K=A^{\mathsf T}A$（$A$ 為 $(N+1)\times N$ 的差分矩陣），並說明為什麼這保證 $K$ 半正定。
# 4. 把 $T$（自由–固定）換成 $B$（自由–自由），解 $BU=F$ 會發生什麼事？從物理上解釋。
#
# 參考解答如下。

# %%
section("練習參考解答")


def lu_solve_many(A, B):
    """練習 2：一次 LU 分解，解多個右側。"""
    P, L, U = plu(A)
    return np.column_stack([back_substitution(U, forward_substitution(L, P @ B[:, j]))
                            for j in range(B.shape[1])])


rng = np.random.default_rng(3)
Atest = rng.standard_normal((60, 60))
Btest = rng.standard_normal((60, 5))
check("練習 2：多右側解法正確", Atest @ lu_solve_many(Atest, Btest), Btest, tol=1e-8)

N = 5
Adf = np.zeros((N + 1, N))             # (N+1) x N 的差分矩陣
for i in range(N):
    Adf[i, i] = 1.0
    Adf[i + 1, i] = -1.0
check("練習 3：AᵀA = K", Adf.T @ Adf, second_difference_matrix(N))
print("  因為 xᵀ(AᵀA)x = ‖Ax‖² ≥ 0，所以 K 一定半正定；A 各欄獨立時更是正定")

Bsing = second_difference_matrix(4)
Bsing[0, 0] = Bsing[-1, -1] = 1.0
try:
    solve(Bsing, np.array([0.0, 1.0, 0.0, 0.0]))
except ValueError as e:
    print(f"  練習 4：解 BU = F 失敗 → {e}")
print("  物理意義：兩端自由的彈簧串沒有支撐點，施加淨力時整體會平移，無平衡位置")

# %% [markdown]
# ## 本章重點回顧
#
# * 消去法 = 一連串可逆的列運算，把 $A\mathbf x=\mathbf b$ 變成 $U\mathbf x=\mathbf c$。
# * 每一步都是矩陣 $E_{ij}$；它們的反矩陣按相反順序相乘，完美組成 $L$，於是 $A=LU$。
# * 成本：$A\to U$ 是 $\tfrac13n^3$，每個右側只要 $n^2$。**不要去算 $A^{-1}$**。
# * 需要換列時 $PA=LU$；部分軸選取讓 $|\ell_{ij}|\le 1$，大幅降低捨入誤差。
# * 轉置的本質是 $(A\mathbf x)\cdot\mathbf y=\mathbf x\cdot(A^{\mathsf T}\mathbf y)$；
#   連續版就是分部積分，$\left(\frac{d}{dt}\right)^{\mathsf T}=-\frac{d}{dt}$。
# * 二階差分矩陣 $K=\operatorname{tridiag}(-1,2,-1)$ 對稱、稀疏、Toeplitz、正定，
#   是後面第 6、13、17、18 章反覆出現的主角。
#
# 下一章：當 $A$ 不是方陣、或欄相依時，解的全貌是什麼？── 四個基本子空間。
