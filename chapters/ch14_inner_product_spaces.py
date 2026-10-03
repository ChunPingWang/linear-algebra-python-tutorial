# %% [markdown]
# # 第 14 章　物理與工程視角：內積空間、矩陣函數與二次式
#
# > 對應 Riley 第 8 章（8.1 向量空間與內積、8.2 線性算子、8.5 矩陣函數、
# > 8.6–8.8 轉置／Hermitian 共軛／跡、8.12 特殊方陣、8.15 基底變換與相似變換、
# > 8.17 二次式與 Hermitian 式、8.18 同時線性方程），
# > 並對照 Strang 第 1、4、6、7 章。
#
# 同一套數學，物理書的講法不一樣：
#
# | Strang 的說法 | Riley 的說法 |
# |---|---|
# | $\mathbf x^{\mathsf T}\mathbf y$ | $\langle a|b\rangle$（Dirac 括號）|
# | 對稱矩陣 $S$ | 自伴算子 $\mathcal A=\mathcal A^{\dagger}$ |
# | $A^{\mathsf T}$ | $A^{\dagger}$（Hermitian 共軛）|
# | 正交基底 | 完備正交歸一集 $\langle\hat e_i|\hat e_j\rangle=\delta_{ij}$ |
# | Gram 矩陣 $A^{\mathsf T}A$ | **度規** $G_{ij}=\langle e_i|e_j\rangle$ |
# | 最小平方 | 最小平方 ＋ **條件數**（病態問題）|
#
# ```bash
# python chapters/ch14_inner_product_spaces.py
# python tools/build_notebooks.py ch14
# ```

# %%
import pathlib
import sys

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.eigen import is_positive_definite, matrix_exp_series, rayleigh_quotient
from linalg_tutorial.elimination import cholesky, solve
from linalg_tutorial.orthogonal import least_squares_normal, min_norm_solution
from linalg_tutorial.svd_tools import condition_number
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 14.1　內積空間的公設
#
# $$\text{(i) }\langle a|b\rangle=\langle b|a\rangle^{*},\qquad
# \text{(ii) }\langle a|\lambda b+\mu c\rangle=\lambda\langle a|b\rangle+\mu\langle a|c\rangle$$
#
# 由這兩條推出**第一個位置是共軛線性的**：
#
# $$\langle\lambda a+\mu b|c\rangle=\lambda^{*}\langle a|c\rangle+\mu^{*}\langle b|c\rangle$$
#
# 範數 $\|a\|=\langle a|a\rangle^{1/2}$，正交 $\iff\langle a|b\rangle=0$。
# 物理學家把第一個位置寫成「bra」$\langle a|$、第二個寫成「ket」$|b\rangle$。

# %%
section("14.1 複數內積的公設")

inner = lambda a, b: np.vdot(a, b)        # np.vdot 自動對第一個參數取共軛

a = np.array([1 + 2j, 3 - 1j, 0 + 1j])
b = np.array([2 - 1j, 1 + 1j, 2 + 0j])
c = np.array([0 + 1j, -1 + 0j, 1 - 1j])
lam, mu = 2 - 1j, 3 + 2j

print(f"a = {a}")
print(f"b = {b}")
print(f"⟨a|b⟩ = {inner(a, b)}")
print(f"⟨b|a⟩ = {inner(b, a)}")
check("(i) ⟨a|b⟩ = ⟨b|a⟩*", inner(a, b), np.conj(inner(b, a)))
check("(ii) 第二個位置線性", inner(a, lam * b + mu * c),
      lam * inner(a, b) + mu * inner(a, c))
check("推論：第一個位置共軛線性", inner(lam * a + mu * b, c),
      np.conj(lam) * inner(a, c) + np.conj(mu) * inner(b, c))
check("⟨a|a⟩ 是非負實數", inner(a, a).imag, 0.0)
check("‖a‖² = Σ|aᵢ|²", inner(a, a).real, float(np.sum(np.abs(a) ** 2)))
print(f"‖a‖ = {np.sqrt(inner(a, a).real):.6f}")
print("  注意 ⟨λa|μb⟩ = λ*μ⟨a|b⟩：量子力學裡的相位因子就是這樣來的")
check("⟨λa|μb⟩ = λ*μ⟨a|b⟩", inner(lam * a, mu * b),
      np.conj(lam) * mu * inner(a, b))

# Schwarz 與三角不等式（複數版）
check("Schwarz：|⟨a|b⟩| ≤ ‖a‖‖b‖",
      abs(inner(a, b)) <= np.sqrt(inner(a, a).real * inner(b, b).real) + 1e-12)
check("三角：‖a+b‖ ≤ ‖a‖ + ‖b‖",
      np.linalg.norm(a + b) <= np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)

# %% [markdown]
# ### 非正交基底與度規矩陣 $G$
#
# 基底不正交時，內積**不是**座標的簡單相乘：
#
# $$G_{ij}=\langle e_i|e_j\rangle,\qquad
# \langle a|b\rangle=\sum_{i,j}a_i^{*}G_{ij}b_j=\mathbf a^{\dagger}G\mathbf b$$
#
# $G$ 就是 Strang 的 Gram 矩陣 $E^{\dagger}E$，也是張量分析裡的 **度規張量**（第 16 章）。
# 正交歸一基底 $\iff G=I$ ── 這才是「座標直接相乘」能成立的理由。

# %%
section("14.1 度規矩陣：非正交基底下的內積")

E = np.array([[1.0, 1.0, 0.0],
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 2.0]])          # 三個非正交的基底向量（當成欄）
G = E.T @ E
show_matrix("度規 G_ij = ⟨eᵢ|eⱼ⟩", G)
check("G 對稱（實數情形）", G, G.T)
check("G 正定（基底獨立 ⇒ 度規正定）", is_positive_definite(G))

coord_a = np.array([1.0, -2.0, 3.0])
coord_b = np.array([0.5, 1.0, -1.0])
vec_a, vec_b = E @ coord_a, E @ coord_b
print(f"\n向量 a 的座標 = {coord_a}，實際向量 = {vec_a}")
check("⟨a|b⟩ = aᵀGb（用座標 + 度規）", coord_a @ G @ coord_b, vec_a @ vec_b)
check("若誤用 aᵀb（當成正交基底）會算錯",
      not np.isclose(coord_a @ coord_b, vec_a @ vec_b))
print(f"  正確值 {vec_a @ vec_b:.4f}，誤算成 {coord_a @ coord_b:.4f}")
check("‖a‖² = aᵀGa", coord_a @ G @ coord_a, vec_a @ vec_a)

# 正交歸一化之後 G = I
Q, R = np.linalg.qr(E)
check("正交歸一基底的度規 = I", Q.T @ Q, np.eye(3))
coord_q = np.linalg.solve(Q, vec_a)       # 同一向量在正交基底下的座標
check("正交基底下 ⟨a|b⟩ = 座標直接相乘",
      coord_q @ np.linalg.solve(Q, vec_b), vec_a @ vec_b)
print("  ⇒ 「座標相乘就是內積」只在正交歸一基底下成立（第 16 章的張量就是在處理這件事）")

# Bessel 不等式：投影到不完備的正交集
section("14.1 Bessel 不等式：投影永遠不會變長")

rng = np.random.default_rng(0)
Qfull, _ = np.linalg.qr(rng.standard_normal((6, 6)))
x = rng.standard_normal(6)
print(f"{'保留的基底數 k':>14} | {'Σ|⟨qᵢ|x⟩|²':>14} | {'‖x‖²':>10}")
print("-" * 44)
for k in range(1, 7):
    partial = float(np.sum((Qfull[:, :k].T @ x) ** 2))
    print(f"{k:>14} | {partial:>14.6f} | {x @ x:>10.6f}")
    check(f"  k = {k}：Bessel 不等式成立", partial <= x @ x + 1e-12)
check("k = N（完備）時取等號 ⇒ Parseval 恆等式",
      float(np.sum((Qfull.T @ x) ** 2)), x @ x)
print("  完備時等號成立 = Parseval 恆等式（第 17 章 Fourier 級數的核心）")

# %% [markdown]
# ## 14.2　線性算子與相似變換
#
# 算子 $\mathcal A$ 本身與基底無關；它的**矩陣**依賴基底：
#
# $$\mathcal A\,\mathbf e_j=\sum_i A_{ij}\mathbf e_i,\qquad
# A'=S^{-1}AS\quad(\mathbf x=S\mathbf x')$$
#
# 相似變換下的**不變量**：特徵值、$\operatorname{trace}$、$\det$、秩。
# 物理意義：它們是「物理量」，不隨座標系改變。

# %%
section("14.2 相似變換與不變量")

A = np.array([[1.0, 1.0, 3.0], [1.0, 1.0, -3.0], [3.0, -3.0, -3.0]])
S_basis = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
A_new = np.linalg.inv(S_basis) @ A @ S_basis
show_matrix("A（原基底）", A)
show_matrix("A' = S⁻¹AS（新基底）", A_new)
print(f"\n{'不變量':<14} | {'原基底':>14} | {'新基底':>14}")
print("-" * 48)
for name, f in [("trace", np.trace), ("det", np.linalg.det),
                ("rank", np.linalg.matrix_rank)]:
    print(f"{name:<14} | {f(A):>14.6f} | {f(A_new):>14.6f}")
    check(f"  {name} 不變", f(A), f(A_new), tol=1e-8)
check("特徵值不變",
      np.sort(np.real(np.linalg.eigvals(A))), np.sort(np.real(np.linalg.eigvals(A_new))),
      tol=1e-8)
print(f"特徵值 = {np.sort(np.real(np.linalg.eigvals(A)))}（Riley 的例子：−6, 2, 3）")
check("trace = Σλ", np.trace(A), float(np.sum(np.linalg.eigvalsh(A))))
check("det = Πλ", np.linalg.det(A), float(np.prod(np.linalg.eigvalsh(A))), tol=1e-8)
check("tr(AB) = tr(BA)（相似不變性的根源）",
      np.trace(A @ S_basis), np.trace(S_basis @ A))

# %% [markdown]
# ## 14.3　矩陣函數
#
# 把純量函數的冪級數直接搬到矩陣上：
#
# $$e^{A}=\sum_{n}\frac{A^n}{n!},\qquad
# \sin A=\sum_{n}\frac{(-1)^nA^{2n+1}}{(2n+1)!},\qquad
# (I-A)^{-1}=\sum_{n}A^n\ (\|A\|<1)$$
#
# 可對角化時最簡單：$f(A)=Xf(\Lambda)X^{-1}$，即對每個特徵值取 $f$。
#
# Riley 的漂亮結果（8.104）：
#
# $$\boxed{\det(e^{A})=e^{\operatorname{trace}A}}$$

# %%
section("14.3 矩陣函數與 det(exp A) = exp(tr A)")

M = np.array([[0.4, 0.3, -0.2], [0.1, -0.5, 0.2], [0.3, 0.1, 0.2]])
lamM, XM = np.linalg.eig(M)

# 由特徵分解定義 f(A)
f_eig = lambda f, A: np.real_if_close(
    (lambda l, X: X @ np.diag(f(l)) @ np.linalg.inv(X))(*np.linalg.eig(A)))

check("exp：級數 = 特徵分解", matrix_exp_series(M, 1.0, 80), f_eig(np.exp, M), tol=1e-10)
print(f"det(exp M) = {np.linalg.det(matrix_exp_series(M, 1.0, 80)):.10f}")
print(f"exp(tr M)  = {np.exp(np.trace(M)):.10f}")
check("det(exp A) = exp(tr A)", np.linalg.det(matrix_exp_series(M, 1.0, 80)),
      np.exp(np.trace(M)), tol=1e-10)

from math import factorial

sin_series = sum((-1) ** n / factorial(2 * n + 1)
                 * np.linalg.matrix_power(M, 2 * n + 1) for n in range(25))
cos_series = sum((-1) ** n / factorial(2 * n)
                 * np.linalg.matrix_power(M, 2 * n) for n in range(25))
check("sin A：級數 = 特徵分解", sin_series, f_eig(np.sin, M), tol=1e-10)
check("sin²A + cos²A = I", sin_series @ sin_series + cos_series @ cos_series,
      np.eye(3), tol=1e-9)

# (I − A)⁻¹ = Σ Aⁿ（Neumann 級數，第 22 章積分方程會再用）
rho = max(abs(np.linalg.eigvals(M)))
print(f"\nM 的譜半徑 = {rho:.6f} < 1 ⇒ Neumann 級數收斂")
neumann = sum(np.linalg.matrix_power(M, n) for n in range(200))
check("(I − A)⁻¹ = I + A + A² + ...", neumann, np.linalg.inv(np.eye(3) - M),
      tol=1e-9)

# 矩陣平方根（正定矩陣）
S_pd = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
S_half = f_eig(np.sqrt, S_pd)
check("√S 的平方 = S", S_half @ S_half, S_pd, tol=1e-10)
check("√S 對稱正定", is_positive_definite(S_half))
print("  注意：矩陣平方根不唯一（每個特徵值都能取 ±√λ），我們取正定的那一個")

# 不可對角化時只能用級數
J = np.array([[2.0, 1.0], [0.0, 2.0]])
print(f"\nJordan 塊（不可對角化）：exp(J) =\n{np.round(matrix_exp_series(J, 1.0, 60), 6)}")
check("= [[e², e²], [0, e²]]", matrix_exp_series(J, 1.0, 60),
      np.array([[np.exp(2.0), np.exp(2.0)], [0.0, np.exp(2.0)]]), tol=1e-10)
check("det(exp J) = exp(tr J) 仍然成立",
      np.linalg.det(matrix_exp_series(J, 1.0, 60)), np.exp(np.trace(J)), tol=1e-9)

# %% [markdown]
# ## 14.4　特殊方陣的完整分類
#
# | 類型 | 定義 | 特徵值位置 | 可對角化？ |
# |---|---|---|---|
# | Hermitian | $A^{\dagger}=A$ | 實軸 | unitary 可對角化 |
# | 反 Hermitian | $A^{\dagger}=-A$ | 虛軸 | unitary 可對角化 |
# | unitary | $A^{\dagger}A=I$ | 單位圓 | unitary 可對角化 |
# | **normal** | $A^{\dagger}A=AA^{\dagger}$ | 任意 | **unitary 可對角化** |
# | 冪零 | $A^k=0$ | 全為 0 | 不可（除非 $A=0$）|
# | 投影 | $P^2=P$ | 只有 0, 1 | 可 |
# | Markov | 各欄和為 1 | $|\lambda|\le1$，含 1 | 通常可 |
#
# **關鍵定理**：$A$ 可被 unitary 矩陣對角化 $\iff$ $A$ 是 normal。
# 前三類都是 normal 的特例 ── 這解釋了為什麼它們的特徵向量都正交。

# %%
section("14.4 normal 矩陣 = unitary 可對角化")

mats = {
    "Hermitian": np.array([[2.0, 1 - 1j], [1 + 1j, 3.0]]),
    "反 Hermitian": np.array([[1j, 2.0], [-2.0, 3j]]),
    "unitary": np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2),
    "normal（非上述三類）": np.array([[1.0, -1.0], [1.0, 1.0]], dtype=complex),
    "非 normal": np.array([[1.0, 1.0], [0.0, 2.0]], dtype=complex),
}
print(f"{'類型':<22} | {'normal?':>8} | {'特徵值':<34} | {'特徵向量正交?':>14}")
print("-" * 86)
for name, Mx in mats.items():
    normal = np.allclose(Mx.conj().T @ Mx, Mx @ Mx.conj().T)
    lam, V = np.linalg.eig(Mx)
    Vn = V / np.linalg.norm(V, axis=0)
    orth = np.allclose(np.abs(Vn.conj().T @ Vn), np.eye(len(lam)), atol=1e-8)
    print(f"{name:<22} | {str(normal):>8} | {str(np.round(lam, 4)):<34} | {str(orth):>14}")
    check(f"  {name}：normal ⟺ 特徵向量正交", normal == orth)

H = mats["Hermitian"]
check("Hermitian 的 λ 全為實數", np.allclose(np.imag(np.linalg.eigvals(H)), 0))
check("反 Hermitian 的 λ 為純虛數",
      np.allclose(np.real(np.linalg.eigvals(mats["反 Hermitian"])), 0, atol=1e-10))
check("unitary 的 |λ| = 1", np.abs(np.linalg.eigvals(mats["unitary"])), np.ones(2))

# 投影與冪零
P = np.array([[0.5, 0.5], [0.5, 0.5]])
N = np.array([[0.0, 3.0], [0.0, 0.0]])
check("投影矩陣：P² = P 且 λ ∈ {0,1}",
      np.sort(np.real(np.linalg.eigvals(P))), np.array([0.0, 1.0]))
check("冪零矩陣：N² = 0 且所有 λ = 0", N @ N, np.zeros((2, 2)))
check("  冪零 ⇒ trace = det = 0", (np.trace(N), np.linalg.det(N)), (0.0, 0.0))
print("  冪零矩陣是「所有特徵值都是 0 但矩陣不是 0」的例子 ⇒ 特徵值不足以決定矩陣")

# %% [markdown]
# ## 14.5　二次式與 Hermitian 式
#
# $$Q(\mathbf x)=\mathbf x^{\mathsf T}A\mathbf x\quad(\text{實}),\qquad
# H(\mathbf x)=\mathbf x^{\dagger}A\mathbf x\quad(A^{\dagger}=A\Rightarrow H\text{ 為實數})$$
#
# **只有對稱部分有貢獻**：把 $M=A+B$（$A$ 對稱、$B$ 反對稱），則
# $\mathbf x^{\mathsf T}B\mathbf x=0$，所以不失一般性可假設 $A=A^{\mathsf T}$。
#
# 用特徵向量當基底，**交叉項全部消失**：
#
# $$Q=\lambda_1x_1'^2+\lambda_2x_2'^2+\dots+\lambda_Nx_N'^2$$
#
# **特徵向量的駐值性質**（Riley 8.17.1）：在 $\|\mathbf x\|=1$ 的限制下
# 使 $Q$ 駐定的 $\mathbf x$ 正好是特徵向量，駐值就是特徵值
# ── 用 Lagrange 乘子一行推出 $A\mathbf x=\lambda\mathbf x$。

# %%
section("14.5 二次式：只有對稱部分有貢獻")

M_any = np.array([[1.0, 5.0, 2.0], [-3.0, 1.0, 0.0], [4.0, -2.0, -3.0]])
A_sym = (M_any + M_any.T) / 2
B_anti = (M_any - M_any.T) / 2
check("M = A + B（對稱 + 反對稱）", A_sym + B_anti, M_any)
rng = np.random.default_rng(1)
ok = True
for _ in range(500):
    x = rng.standard_normal(3)
    ok = ok and abs(x @ B_anti @ x) < 1e-12 and abs(x @ M_any @ x - x @ A_sym @ x) < 1e-12
check("xᵀBx = 0（反對稱部分無貢獻）⇒ xᵀMx = xᵀAx", ok)

# Riley 的例子 (8.107)
A = np.array([[1.0, 1.0, 3.0], [1.0, 1.0, -3.0], [3.0, -3.0, -3.0]])
show_matrix("A（Riley 8.107）", A)
print("Q = x₁² + x₂² − 3x₃² + 2x₁x₂ + 6x₁x₃ − 6x₂x₃")
x_test = np.array([1.0, -2.0, 0.5])
Q_expand = (x_test[0] ** 2 + x_test[1] ** 2 - 3 * x_test[2] ** 2
            + 2 * x_test[0] * x_test[1] + 6 * x_test[0] * x_test[2]
            - 6 * x_test[1] * x_test[2])
check("展開式 = xᵀAx", Q_expand, x_test @ A @ x_test)

lam, S_eig = np.linalg.eigh(A)
print(f"\n特徵值 λ = {lam}（Riley：−6, 2, 3）")
S_riley = np.array([[np.sqrt(3), np.sqrt(2), 1.0],
                    [np.sqrt(3), -np.sqrt(2), -1.0],
                    [0.0, np.sqrt(2), -2.0]]) / np.sqrt(6)
check("Riley 的 S 是正交矩陣", S_riley.T @ S_riley, np.eye(3))
D_riley = S_riley.T @ A @ S_riley
show_matrix("SᵀAS（對角！交叉項消失）", D_riley)
check("SᵀAS 是對角矩陣", D_riley, np.diag(np.diag(D_riley)), tol=1e-10)
check("對角線 = (2, 3, −6)", np.round(np.diag(D_riley), 10), np.array([2.0, 3.0, -6.0]))
x_prime = S_riley.T @ x_test
check("Q = 2x'₁² + 3x'₂² − 6x'₃²",
      2 * x_prime[0] ** 2 + 3 * x_prime[1] ** 2 - 6 * x_prime[2] ** 2,
      x_test @ A @ x_test)
print("  有正有負的特徵值 ⇒ Q 是不定的（鞍形）")
check("A 不定（有正有負特徵值）", np.any(lam > 0) and np.any(lam < 0))

# 駐值性質
section("14.5 特徵向量的駐值性質（Rayleigh 商）")

S_pd = np.array([[5.0, 2.0, 1.0], [2.0, 4.0, 0.0], [1.0, 0.0, 3.0]])
lam_pd, V_pd = np.linalg.eigh(S_pd)
print(f"特徵值 = {lam_pd}")
print("在 ‖x‖ = 1 的限制下，使 Q = xᵀSx 駐定的 x 就是特徵向量：")
for k in range(3):
    v = V_pd[:, k]
    Q_val = v @ S_pd @ v
    grad = 2 * (S_pd @ v - Q_val * v)          # ∇[xᵀSx − λ(xᵀx−1)]
    print(f"  特徵向量 {k + 1}：Q = {Q_val:.6f}，λ = {lam_pd[k]:.6f}，"
          f"‖∇L‖ = {np.linalg.norm(grad):.2e}")
    check(f"  Q(vₖ) = λₖ", Q_val, lam_pd[k])
    check(f"  Lagrange 條件 Sx = λx 成立", S_pd @ v, lam_pd[k] * v)

rng = np.random.default_rng(2)
vals = [rayleigh_quotient(S_pd, rng.standard_normal(3)) for _ in range(20000)]
print(f"\n隨機方向的 Rayleigh 商範圍 = [{min(vals):.4f}, {max(vals):.4f}]")
print(f"理論範圍 [λ_min, λ_max] = [{lam_pd[0]:.4f}, {lam_pd[-1]:.4f}]")
check("Rayleigh 商的極小 = λ_min", min(vals), lam_pd[0], tol=2e-2)
check("Rayleigh 商的極大 = λ_max", max(vals), lam_pd[-1], tol=2e-2)
print("  ⇒ 這就是第 15 章 Rayleigh–Ritz 法與第 22 章變分法的理論基礎")

# 物理應用：慣性張量的主軸
section("14.5 物理應用：慣性張量的主軸")

# 四個質點的慣性張量
masses = np.array([1.0, 2.0, 1.5, 3.0])
pos = np.array([[1.0, 0.5, -1.0], [0.0, 1.0, 0.5], [-1.0, -0.5, 1.0], [0.5, -1.0, -0.5]])
I_tensor = np.zeros((3, 3))
for m, r in zip(masses, pos):
    I_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
show_matrix("慣性張量 I", I_tensor)
check("慣性張量對稱", I_tensor, I_tensor.T)
check("慣性張量正定（質量為正且不共線）", is_positive_definite(I_tensor))
I_principal, axes = np.linalg.eigh(I_tensor)
print(f"主慣性矩 = {I_principal}")
print(f"主軸（正交）=\n{np.round(axes, 4)}")
check("主軸正交", axes.T @ axes, np.eye(3))
check("在主軸座標下慣性張量是對角的（沒有交叉項）",
      axes.T @ I_tensor @ axes, np.diag(I_principal), tol=1e-10)
omega = np.array([0.3, -0.5, 0.8])
L = I_tensor @ omega
T_rot = 0.5 * omega @ I_tensor @ omega
print(f"\n角速度 ω = {omega} ⇒ 角動量 L = {np.round(L, 4)}")
print(f"轉動動能 T = ½ωᵀIω = {T_rot:.6f}（一個二次式！）")
omega_prime = axes.T @ omega
check("T 在主軸座標下 = ½Σ Iₖω'ₖ²",
      0.5 * np.sum(I_principal * omega_prime ** 2), T_rot)
check("L ∥ ω 只在 ω 沿主軸時成立",
      np.allclose(np.cross(I_tensor @ axes[:, 0], axes[:, 0]), 0, atol=1e-10))
print("  只有沿主軸旋轉時 L 才與 ω 平行 —— 這就是陀螺進動的數學原因")

# %% [markdown]
# ## 14.6　同時線性方程：三種形狀，一種工具
#
# Riley 把 $N$ 條方程、$M$ 個未知數分成三種情形：
#
# | 情形 | 幾何 | 解法 |
# |---|---|---|
# | $N=M$，$\det\ne0$ | 唯一解 | $LU$／Cholesky／Cramer／迭代 |
# | $N>M$（超定）| 通常無解 | **最小平方**（第 4 章）|
# | $N<M$（欠定）| 無窮多解 | **最小範數**（第 4 章）|
#
# 而 SVD 一次處理全部三種（偽逆 $A^{+}$）。
#
# 真正的實務風險是**病態**（ill-conditioned）：
# $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ 很大時，資料的小誤差會被放大。

# %%
section("14.6 三種形狀，SVD 一次解決")

rng = np.random.default_rng(3)
cases = {
    "N = M = 4（唯一解）": (rng.standard_normal((4, 4)), 4, 4),
    "N = 8 > M = 3（超定）": (rng.standard_normal((8, 3)), 8, 3),
    "N = 3 < M = 7（欠定）": (rng.standard_normal((3, 7)), 3, 7),
}
for name, (A_c, N_, M_) in cases.items():
    x_true = rng.standard_normal(M_)
    b_c = A_c @ x_true
    x_svd = min_norm_solution(A_c, b_c)       # A⁺b
    resid = np.linalg.norm(A_c @ x_svd - b_c)
    print(f"\n{name}")
    print(f"  SVD 解的殘差 ‖Ax−b‖ = {resid:.3e}，‖x‖ = {np.linalg.norm(x_svd):.6f}"
          f"（真值 ‖x‖ = {np.linalg.norm(x_true):.6f}）")
    if N_ == M_:
        check("  與 LU 解相同", x_svd, solve(A_c, b_c), tol=1e-8)
        check("  完全滿足方程", resid < 1e-10)
    elif N_ > M_:
        check("  與最小平方解相同", x_svd, least_squares_normal(A_c, b_c), tol=1e-8)
        check("  b 在 C(A) 內 ⇒ 殘差為 0", resid < 1e-10)
    else:
        check("  滿足方程", resid < 1e-10)
        check("  ‖x_SVD‖ ≤ ‖x_true‖（最小範數）",
              np.linalg.norm(x_svd) <= np.linalg.norm(x_true) + 1e-9)

# 超定但不相容的情形
A_over = rng.standard_normal((8, 3))
b_bad = rng.standard_normal(8)
x_ls = min_norm_solution(A_over, b_bad)
e_ls = A_over @ x_ls - b_bad
check("\n不相容的超定系統：殘差 ⟂ C(A)", A_over.T @ e_ls, np.zeros(3), tol=1e-10)
print(f"  最小殘差 = {np.linalg.norm(e_ls):.6f}（無法再更小）")

# %%
section("14.6 病態系統：Riley 的警告")

print("Riley 的例子：係數只差一點，解卻天差地別")
A_ill = np.array([[1.0, 2.0], [1.0, 2.001]])
b1 = np.array([3.0, 3.001])
b2 = np.array([3.0, 3.002])
x1, x2 = solve(A_ill, b1), solve(A_ill, b2)
print(f"  A = {A_ill.tolist()}")
print(f"  b = {b1} → x = {x1}")
print(f"  b = {b2} → x = {x2}")
print(f"  b 的相對變化 = {np.linalg.norm(b2 - b1) / np.linalg.norm(b1):.3e}")
print(f"  x 的相對變化 = {np.linalg.norm(x2 - x1) / np.linalg.norm(x1):.3e}")
kappa = condition_number(A_ill)
print(f"  κ(A) = {kappa:.1f}（放大倍數的上界）")
amp = ((np.linalg.norm(x2 - x1) / np.linalg.norm(x1))
       / (np.linalg.norm(b2 - b1) / np.linalg.norm(b1)))
check("實際放大倍數 ≤ κ(A)", amp <= kappa * 1.01)
print(f"  實際放大 {amp:.1f} 倍")

print(f"\n{'矩陣':<24} | {'κ(A)':>12} | {'可信的有效位數':>16}")
print("-" * 58)
for name, Mx in [("良態（正交）", np.linalg.qr(rng.standard_normal((6, 6)))[0]),
                 ("中等", rng.standard_normal((6, 6))),
                 ("Riley 的病態例子", A_ill),
                 ("Hilbert 8x8", np.array([[1 / (i + j + 1) for j in range(8)]
                                           for i in range(8)]))]:
    k = condition_number(Mx)
    digits = max(0, 16 - np.log10(k))
    print(f"{name:<24} | {k:>12.4g} | {digits:>16.1f}")
    check(f"  {name}：κ ≥ 1", k >= 1 - 1e-12)
print("  經驗法則：雙精度有 16 位有效位數，κ = 10ᵏ 就會失去 k 位")

# 對稱正定的捷徑：Cholesky 比 LU 快一倍
S_big = rng.standard_normal((200, 200))
S_big = S_big @ S_big.T + 200 * np.eye(200)
import time
t0 = time.perf_counter()
R_chol = cholesky(S_big)
t_chol = time.perf_counter() - t0
check("Cholesky 正確：S = RᵀR", R_chol.T @ R_chol, S_big, tol=1e-8)
print(f"\n對稱正定矩陣用 Cholesky（n³/6）而非 LU（n³/3）：約省一半運算")
print(f"  200x200 的 Cholesky 耗時 {t_chol:.4f} 秒")
check("Cholesky 的 R 是上三角", R_chol, np.triu(R_chol))

# %% [markdown]
# ## 動手練習
#
# 1. 用 Dirac 符號驗證 $\|a+b\|^2+\|a-b\|^2=2\|a\|^2+2\|b\|^2$（複數版）。
# 2. 給三個非正交基底向量，算出度規 $G$，並用它求某個向量的長度與兩向量夾角。
# 3. 驗證 $\det(e^{A})=e^{\operatorname{tr}A}$ 對隨機的 $5\times5$ 矩陣（含複數特徵值）。
# 4. 把二次式 $Q=2x^2+3y^2+2z^2+2xy+2yz$ 對角化，判斷它是否正定。
# 5. 建一個 $\kappa\approx10^{10}$ 的矩陣，觀察雙精度能保留幾位有效位數。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
check("練習 1：平行四邊形恆等式（複數）",
      np.linalg.norm(a + b) ** 2 + np.linalg.norm(a - b) ** 2,
      2 * np.linalg.norm(a) ** 2 + 2 * np.linalg.norm(b) ** 2)

# 練習 2
E2 = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
G2 = E2.T @ E2
show_matrix("練習 2：度規 G", G2)
u_c, v_c = np.array([1.0, -1.0, 2.0]), np.array([2.0, 0.0, -1.0])
len_u = np.sqrt(u_c @ G2 @ u_c)
cos_t = (u_c @ G2 @ v_c) / (len_u * np.sqrt(v_c @ G2 @ v_c))
print(f"  ‖u‖ = {len_u:.6f}，夾角 = {np.degrees(np.arccos(cos_t)):.4f}°")
check("  用度規算的長度 = 實際長度", len_u, np.linalg.norm(E2 @ u_c))
check("  用度規算的夾角 = 實際夾角", cos_t,
      (E2 @ u_c) @ (E2 @ v_c) / (np.linalg.norm(E2 @ u_c) * np.linalg.norm(E2 @ v_c)))

# 練習 3
print("\n練習 3：")
for trial in range(3):
    Mr = rng.standard_normal((5, 5)) * 0.3
    eM = matrix_exp_series(Mr, 1.0, 100)
    print(f"  #{trial + 1} det(exp A) = {np.linalg.det(eM):.10f}，"
          f"exp(tr A) = {np.exp(np.trace(Mr)):.10f}，"
          f"複特徵值：{np.any(np.abs(np.imag(np.linalg.eigvals(Mr))) > 1e-10)}")
    check(f"  #{trial + 1} 相等", np.linalg.det(eM), np.exp(np.trace(Mr)), tol=1e-9)

# 練習 4
A4 = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
lam4, S4 = np.linalg.eigh(A4)
print(f"\n練習 4：λ = {lam4}")
check("  SᵀAS 對角", S4.T @ A4 @ S4, np.diag(lam4), tol=1e-10)
check("  全部 λ > 0 ⇒ 正定", np.all(lam4 > 0))
check("  與五個測試一致（第 7 章）", is_positive_definite(A4))
x4 = np.array([1.0, -1.0, 0.5])
check("  Q = 2x²+3y²+2z²+2xy+2yz 的展開 = xᵀAx",
      2 * x4[0] ** 2 + 3 * x4[1] ** 2 + 2 * x4[2] ** 2
      + 2 * x4[0] * x4[1] + 2 * x4[1] * x4[2], x4 @ A4 @ x4)

# 練習 5
print("\n練習 5：人造病態矩陣")
U5, _ = np.linalg.qr(rng.standard_normal((6, 6)))
V5, _ = np.linalg.qr(rng.standard_normal((6, 6)))
for target_kappa in [1e4, 1e8, 1e12]:
    sv = np.logspace(0, -np.log10(target_kappa), 6)
    A5 = U5 @ np.diag(sv) @ V5.T
    x_t = np.ones(6)
    b5 = A5 @ x_t
    x_num = np.linalg.solve(A5, b5)
    rel = np.linalg.norm(x_num - x_t) / np.linalg.norm(x_t)
    print(f"  κ = {condition_number(A5):>10.3g}：解的相對誤差 = {rel:.3e}"
          f"（≈ κ·ε = {condition_number(A5) * 2.2e-16:.3e}）")
    check(f"  誤差 ≲ κ·ε_machine", rel <= max(condition_number(A5) * 2.2e-16 * 100, 1e-14))

# %% [markdown]
# ## 本章重點回顧
#
# * 內積的兩條公設（共軛對稱 + 第二位置線性）推出第一位置共軛線性；
#   Dirac 的 bra–ket 就是這個結構。
# * 基底不正交時，內積要用**度規矩陣** $G_{ij}=\langle e_i|e_j\rangle$：
#   $\langle a|b\rangle=\mathbf a^{\dagger}G\mathbf b$。$G=I$ 才能「座標直接相乘」。
# * **Bessel 不等式**：部分投影的能量不超過總能量；完備時取等號（Parseval）。
# * 相似變換 $A'=S^{-1}AS$ 下 $\lambda$、trace、det、rank 都不變
#   ── 它們才是「物理量」。
# * 矩陣函數用冪級數或 $f(A)=Xf(\Lambda)X^{-1}$ 定義；
#   最漂亮的恆等式是 $\det(e^{A})=e^{\operatorname{tr}A}$。
# * **normal 矩陣**（$A^{\dagger}A=AA^{\dagger}$）$\iff$ unitary 可對角化
#   $\iff$ 特徵向量正交。Hermitian／反 Hermitian／unitary 都是它的特例。
# * 二次式只有對稱部分有貢獻；用特徵向量當基底就消掉所有交叉項。
#   **特徵向量 = Rayleigh 商的駐點**，駐值 = 特徵值。
#   慣性張量的主軸、應力張量的主應力都是這一件事。
# * 同時線性方程的三種形狀（唯一／超定／欠定）被 SVD 的偽逆統一處理；
#   實務上真正的敵人是**病態**：$\kappa=10^k$ 就吃掉 $k$ 位有效位數。
#
# 下一章：法模態 ── 廣義特徵值問題 $K\mathbf x=\omega^2M\mathbf x$ 與對稱性。
