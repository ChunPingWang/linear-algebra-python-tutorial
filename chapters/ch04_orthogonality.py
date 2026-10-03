# %% [markdown]
# # 第 4 章　正交性：投影、最小平方、$A=QR$ 與偽逆
#
# > 對應 Strang 第 4 章（4.1 向量與子空間的正交、4.2 投影、4.3 最小平方、
# > 4.4 正交基底與 Gram–Schmidt、4.5 偽逆），以及 Riley 8.1、31.6（最小平方）。
#
# **線性代數基本定理（第二部分）**：四個子空間不只維度互補，還兩兩垂直
#
# $$N(A)=C(A^{\mathsf T})^{\perp}\ \text{（在 }\mathbb R^n），\qquad
# N(A^{\mathsf T})=C(A)^{\perp}\ \text{（在 }\mathbb R^m）$$
#
# 本章主線：$\mathbf b$ 不在 $C(A)$ 裡怎麼辦？→ 投影到最近點 → 最小平方 →
# 把基底先正交化（$A=QR$）→ 秩不滿時用偽逆 $A^{+}$。
#
# ```bash
# python chapters/ch04_orthogonality.py
# python tools/build_notebooks.py ch04
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

from linalg_tutorial.orthogonal import (
    fit_polynomial,
    gram_matrix,
    gram_schmidt,
    householder_qr,
    is_orthonormal,
    least_squares_normal,
    least_squares_qr,
    min_norm_solution,
    project_onto_subspace,
    project_onto_vector,
    projection_matrix,
    pseudoinverse,
    qr_gram_schmidt,
)
from linalg_tutorial.subspaces import cr_factor, left_nullspace, nullspace, rank, row_space
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import draw_vector, finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 4.1　正交向量與正交子空間
#
# $$\mathbf v^{\mathsf T}\mathbf w=0
# \iff \|\mathbf v\|^2+\|\mathbf w\|^2=\|\mathbf v+\mathbf w\|^2
# \quad(\text{畢氏定理})$$
#
# 子空間 $V\perp W$：$V$ 中每個向量都垂直 $W$ 中每個向量。
# 關鍵限制：若 $V\perp W$ 在 $\mathbb R^n$ 中，則 $\dim V+\dim W\le n$。
# （所以房間的牆與地板**不是**正交子空間：$2+2>3$，它們交於一條線。）
#
# **正交補**：維度剛好加滿的一對正交子空間。$A$ 的四個子空間剛好是兩對正交補：
#
# $$\mathbb R^n=C(A^{\mathsf T})\oplus N(A),\qquad
# \mathbb R^m=C(A)\oplus N(A^{\mathsf T})$$
#
# 因此每個 $\mathbf x$ 唯一分解成 $\mathbf x=\mathbf x_{\text{row}}+\mathbf x_{\text{null}}$。

# %%
section("4.1 四個子空間的正交性")

A = np.array([
    [1.0, 1.0, -2.0],
    [1.0, 0.0, -1.0],
])
show_matrix("A", A)
Nn = nullspace(A)
Rs = row_space(A)
show_matrix("N(A) 基底", Nn)
show_matrix("C(Aᵀ) 基底（列空間）", Rs)
check("N(A) ⟂ C(Aᵀ)", Rs.T @ Nn, np.zeros((Rs.shape[1], Nn.shape[1])))
check("Ax = 0 對每一列都給 0 內積", A @ Nn, np.zeros((2, Nn.shape[1])))
print(f"維度：dim C(Aᵀ) + dim N(A) = {Rs.shape[1]} + {Nn.shape[1]} = n = {A.shape[1]}")

# 牆與地板不能正交
print("\n牆與地板的反例：兩個 2 維子空間在 R³ 中必有交線")
wall = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])     # xz 平面
floor = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])    # xy 平面
check("牆 ⟂ 地板不成立", not np.allclose(wall.T @ floor, 0))
print(f"  交線方向 = x 軸，同時在兩個子空間裡；2 + 2 = 4 > 3 ⇒ 不可能正交")

# 任何 x 分解成列空間 + 零空間兩部分
A2 = np.array([[1.0, 2.0], [3.0, 6.0]])
x = np.array([4.0, 3.0])
Prow = projection_matrix(row_space(A2))
x_row = Prow @ x
x_null = x - x_row
print(f"\nx = {x} 分解為：")
print(f"  列空間部分 x_row  = {x_row}")
print(f"  零空間部分 x_null = {x_null}")
check("x_row + x_null = x", x_row + x_null, x)
check("x_row ⟂ x_null", x_row @ x_null, 0.0)
check("x_null 在 N(A) 中", A2 @ x_null, np.zeros(2))
check("長度分解 ‖x‖² = ‖x_row‖² + ‖x_null‖²",
      x @ x, x_row @ x_row + x_null @ x_null)

# %% [markdown]
# ## 4.2　投影
#
# **投影到一條線**（方向 $\mathbf a$）：
#
# $$\hat x=\frac{\mathbf a^{\mathsf T}\mathbf b}{\mathbf a^{\mathsf T}\mathbf a},\qquad
# \mathbf p=\hat x\,\mathbf a,\qquad
# P=\frac{\mathbf a\mathbf a^{\mathsf T}}{\mathbf a^{\mathsf T}\mathbf a}\ \ (\text{秩 }1)$$
#
# **投影到子空間** $C(A)$（$A$ 各欄獨立）：誤差 $\mathbf e=\mathbf b-A\hat{\mathbf x}$
# 必須垂直每一欄，即 $A^{\mathsf T}(\mathbf b-A\hat{\mathbf x})=\mathbf 0$，於是
#
# $$\boxed{A^{\mathsf T}A\hat{\mathbf x}=A^{\mathsf T}\mathbf b}\qquad
# \mathbf p=A\hat{\mathbf x},\qquad P=A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$$
#
# 投影矩陣的兩個特徵：$P^2=P$（投兩次等於投一次）、$P^{\mathsf T}=P$。
# 而 $I-P$ 是投影到正交補空間。
#
# 重要定理：**$A$ 各欄獨立 $\iff$ $A^{\mathsf T}A$ 可逆**
# （證明：$A^{\mathsf T}A\mathbf x=\mathbf 0\Rightarrow\|A\mathbf x\|^2=0\Rightarrow A\mathbf x=\mathbf 0$）

# %%
section("4.2 投影到線與子空間")

a = np.array([1.0, 2.0, 2.0])
b = np.array([1.0, 1.0, 1.0])
p = project_onto_vector(a, b)
e = b - p
print(f"a = {a}, b = {b}")
print(f"x̂ = aᵀb/aᵀa = {a @ b:.4g}/{a @ a:.4g} = {(a @ b) / (a @ a):.4f}")
print(f"投影 p = {p}")
print(f"誤差 e = {e}")
check("e ⟂ a", e @ a, 0.0)
check("‖b‖² = ‖p‖² + ‖e‖²", b @ b, p @ p + e @ e)

P1 = projection_matrix(a)
show_matrix("P = aaᵀ/aᵀa", P1)
check("P² = P", P1 @ P1, P1)
check("Pᵀ = P", P1.T, P1)
check("rank(P) = 1", rank(P1) == 1)
check("trace(P) = 1 = 投影維度", np.trace(P1), 1.0)
check("Pb = p", P1 @ b, p)
check("I − P 投影到正交補", (np.eye(3) - P1) @ b, e)

# 投影到平面（兩欄）
A3 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
b3 = np.array([6.0, 0.0, 0.0])
G = gram_matrix(A3)
show_matrix("AᵀA（Gram 矩陣，元素是欄的兩兩內積）", G)
check("AᵀA 對稱", G, G.T)
check("A 各欄獨立 ⇒ AᵀA 可逆", abs(np.linalg.det(G)) > 1e-12)

xhat = least_squares_normal(A3, b3)
p3, e3 = project_onto_subspace(A3, b3)
print(f"\n解正規方程 AᵀA x̂ = Aᵀb → x̂ = {xhat}")
print(f"投影 p = {p3}，誤差 e = {e3}")
check("e ⟂ A 的每一欄", A3.T @ e3, np.zeros(2))
P3 = projection_matrix(A3)
check("p = Pb", P3 @ b3, p3)
check("P² = P 且 Pᵀ = P", P3 @ P3, P3)
check("trace(P) = 2 = 投影到的維度", np.trace(P3), 2.0)

# 相依欄會讓 AᵀA 奇異
Adep = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
check("相依欄 ⇒ AᵀA 奇異", abs(np.linalg.det(Adep.T @ Adep)) < 1e-10)
check("N(AᵀA) = N(A)", rank(Adep.T @ Adep) == rank(Adep))

# %% [markdown]
# ## 4.3　最小平方近似
#
# $A\mathbf x=\mathbf b$ 方程太多（$m>n$）而無解時，退而求其次：
#
# $$\min_{\mathbf x}\ E=\|A\mathbf x-\mathbf b\|^2
# \quad\Longleftrightarrow\quad A^{\mathsf T}A\hat{\mathbf x}=A^{\mathsf T}\mathbf b$$
#
# 三條路殊途同歸：
#
# 1. **幾何**：誤差必須垂直 $C(A)$
# 2. **代數**：$\mathbf b=\mathbf p+\mathbf e$，解可解的 $A\hat{\mathbf x}=\mathbf p$
# 3. **微積分**：$\partial E/\partial x_i=0$
#
# 配直線 $b=C+Dt$ 時
# $$A=\begin{bmatrix}1&t_1\\ \vdots&\vdots\\ 1&t_m\end{bmatrix},\quad
# A^{\mathsf T}A=\begin{bmatrix}m&\sum t_i\\ \sum t_i&\sum t_i^2\end{bmatrix},\quad
# A^{\mathsf T}\mathbf b=\begin{bmatrix}\sum b_i\\ \sum t_ib_i\end{bmatrix}$$

# %%
section("4.3 最小平方配直線：三種方法同一答案")

t = np.array([0.0, 1.0, 2.0])
b = np.array([6.0, 0.0, 0.0])
A = np.column_stack([np.ones_like(t), t])

# 方法 1/2：正規方程
xhat = least_squares_normal(A, b)
# 方法 3：微積分（對 C, D 偏微分 = 0 的線性方程，手動組出來）
m = len(t)
AtA = np.array([[m, t.sum()], [t.sum(), (t ** 2).sum()]])
Atb = np.array([b.sum(), (t * b).sum()])
x_calc = np.linalg.solve(AtA, Atb)
# 方法：numpy 內建
x_np = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"正規方程      x̂ = {xhat}")
print(f"微積分（偏微分）x̂ = {x_calc}")
print(f"numpy lstsq   x̂ = {x_np}")
check("三種方法一致", xhat, x_calc)
check("與 numpy 一致", xhat, x_np)
print(f"\n最佳直線：b = {xhat[0]:.4g} + ({xhat[1]:.4g})t")
pred = A @ xhat
err = b - pred
print(f"預測值 p = {pred}，誤差 e = {err}")
check("誤差總和 = 0（因為 e ⟂ 全 1 欄）", err.sum(), 0.0)
check("e ⟂ t 欄", err @ t, 0.0)
print(f"E = ‖e‖² = {err @ err:.4g}")

# 驗證這真的是最小值
rng = np.random.default_rng(0)
E_best = err @ err
worse = True
for _ in range(200):
    x_try = xhat + 0.1 * rng.standard_normal(2)
    e_try = A @ x_try - b
    worse = worse and (e_try @ e_try >= E_best - 1e-12)
check("隨機擾動都讓 E 變大 ⇒ x̂ 真的是最小值", worse)

# %% [markdown]
# ### 配拋物線與高次多項式
#
# 即使模型含 $t^2$、$t^3$，**未知係數仍然是線性出現**，所以還是線性代數問題。
# 設計矩陣是 Vandermonde 矩陣 $A=[\,\mathbf 1\ \ \mathbf t\ \ \mathbf t^2\ \cdots\,]$。

# %%
section("4.3 多項式擬合與過度配適")

rng = np.random.default_rng(42)
t_data = np.linspace(-2, 2, 11)
true_f = lambda s: 1.0 + 0.5 * s - 0.8 * s ** 2
b_data = true_f(t_data) + 0.25 * rng.standard_normal(len(t_data))

print(f"{'次數':>6} | {'‖e‖²':>10} | {'係數':<40}")
print("-" * 64)
for deg in [0, 1, 2, 3, 6]:
    coef = fit_polynomial(t_data, b_data, deg)
    Ad = np.vander(t_data, deg + 1, increasing=True)
    resid = b_data - Ad @ coef
    print(f"{deg:>6} | {resid @ resid:>10.4f} | {np.round(coef, 3)}")
coef2 = fit_polynomial(t_data, b_data, 2)
print(f"\n真實係數 (1, 0.5, −0.8) vs 擬合 {np.round(coef2, 3)}")
check("2 次擬合接近真實係數", np.allclose(coef2, [1.0, 0.5, -0.8], atol=0.25))

fig, ax = new_axes("Least squares polynomial fits", figsize=(6.0, 4.2), equal=False)
ts = np.linspace(-2.2, 2.2, 300)
ax.plot(t_data, b_data, "ko", ms=5, label="data")
ax.plot(ts, true_f(ts), "k--", lw=1, label="true quadratic")
for deg, c in zip([1, 2, 6], ["C0", "C2", "C3"]):
    coef = fit_polynomial(t_data, b_data, deg)
    ax.plot(ts, np.vander(ts, deg + 1, increasing=True) @ coef, c, label=f"degree {deg}")
ax.set_xlabel("t")
ax.set_ylabel("b")
ax.legend(fontsize=8)
finish(fig, "ch04_poly_fit")

# %% [markdown]
# ### 離群值：最小平方 vs 最小絕對值
#
# Strang 的經典例子：9 個 0 加上一個 40。三種誤差準則給出完全不同的答案：
#
# | 準則 | 最佳水平線 | 統計意義 |
# |---|---|---|
# | $\sum e_i^2$（最小平方）| $C=4$ | **平均數** |
# | $\max|e_i|$ | $C=20$ | 中點 |
# | $\sum|e_i|$ | $C=0$ | **中位數** |
#
# 最小平方對離群值敏感（因為平方放大大誤差），但它的方程是線性的 ──
# 這正是它普及的原因。

# %%
section("4.3 離群值與三種誤差準則")

b_out = np.array([0.0] * 9 + [40.0])


def golden_min(f, lo, hi, tol=1e-12):
    """黃金分割搜尋：在 [lo, hi] 上找單峰（convex）函數的極小點，不需要微分。"""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    while hi - lo > tol:
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    return (lo + hi) / 2


E2 = lambda c: np.sum((c - b_out) ** 2)
Einf = lambda c: np.max(np.abs(c - b_out))
E1 = lambda c: np.sum(np.abs(c - b_out))
best = {}
for name, f in [("最小平方 Σe²", E2), ("最大誤差 max|e|", Einf), ("絕對值和 Σ|e|", E1)]:
    x_best = golden_min(f, -10.0, 50.0)
    best[name] = x_best
    print(f"  {name:<18} 最佳 C = {x_best:8.4f}")
check("最小平方 = 平均數 4", best["最小平方 Σe²"], float(np.mean(b_out)), tol=1e-5)
check("最大誤差 = 中點 20", best["最大誤差 max|e|"], 20.0, tol=1e-4)
check("絕對值和 = 中位數 0", best["絕對值和 Σ|e|"], float(np.median(b_out)), tol=1e-4)

# 線性縮放性質
A1 = np.ones((10, 1))
c0 = least_squares_normal(A1, b_out)
c1 = least_squares_normal(A1, 3 * b_out + 30)
check("b → 3b + 30 ⇒ C → 3C + 30（線性）", c1, 3 * c0 + 30)

# %% [markdown]
# ## 4.4　正交基底、正交矩陣與 Gram–Schmidt
#
# $$Q^{\mathsf T}Q=I\quad(\text{各欄為單位且兩兩正交})$$
#
# $Q$ 是方陣時 $Q^{\mathsf T}=Q^{-1}$，稱為**正交矩陣**。三大家族：
# **旋轉**、**置換**、**反射** $Q=I-2\mathbf u\mathbf u^{\mathsf T}$。
#
# 正交矩陣**保長保角**：$\|Q\mathbf x\|=\|\mathbf x\|$、
# $(Q\mathbf x)^{\mathsf T}(Q\mathbf y)=\mathbf x^{\mathsf T}\mathbf y$
# ── 數值計算最愛的性質（誤差不會被放大）。
#
# 投影與最小平方立刻變簡單：
# $$\hat{\mathbf x}=Q^{\mathsf T}\mathbf b,\qquad P=QQ^{\mathsf T},\qquad
# \mathbf b=\mathbf q_1(\mathbf q_1^{\mathsf T}\mathbf b)+\dots+\mathbf q_n(\mathbf q_n^{\mathsf T}\mathbf b)$$
#
# 最後一式就是 **Fourier 級數與所有積分變換的原型**（第 15 章）。

# %%
section("4.4 三種正交矩陣")

theta = np.pi / 6
Qrot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Qperm = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
u = np.array([-1.0, 1.0]) / np.sqrt(2)
Qref = np.eye(2) - 2 * np.outer(u, u)

for name, Q in [("旋轉 30°", Qrot), ("置換", Qperm), ("反射（過 45° 線）", Qref)]:
    n_ = Q.shape[0]
    print(f"\n{name}：")
    show_matrix("  Q", Q)
    check("  QᵀQ = I", Q.T @ Q, np.eye(n_))
    check("  Q⁻¹ = Qᵀ", np.linalg.inv(Q), Q.T)
    v = np.arange(1.0, n_ + 1)
    check("  保長 ‖Qv‖ = ‖v‖", np.linalg.norm(Q @ v), np.linalg.norm(v))
    print(f"  det Q = {np.linalg.det(Q):+.4g}"
          f"（+1 = 旋轉類，−1 = 含反射）")
check("反射矩陣對稱且自逆：Q² = I", Qref @ Qref, np.eye(2))

# Hadamard 矩陣：元素只有 ±1 的正交矩陣
H2 = np.array([[1.0, 1.0], [1.0, -1.0]])
H4 = np.block([[H2, H2], [H2, -H2]])
H8 = np.block([[H4, H4], [H4, -H4]])
check("H₄ᵀH₄ = 4I", H4.T @ H4, 4 * np.eye(4))
check("H₈ᵀH₈ = 8I", H8.T @ H8, 8 * np.eye(8))
check("H₄/2 是正交矩陣", is_orthonormal(H4 / 2))
print("\nHadamard 矩陣是 Fourier/Walsh 變換的雛形（第 15 章還會見到）")

# 用正交基底重建向量（Fourier 級數的原型）
Q = np.array([[2.0, -1.0, 2.0], [2.0, 2.0, -1.0], [-1.0, 2.0, 2.0]]) / 3
check("Q 正交", is_orthonormal(Q))
bb = np.array([0.0, 0.0, 1.0])
pieces = [Q[:, k] * (Q[:, k] @ bb) for k in range(3)]
print("\n把 b 拆成三個 1 維投影再加回來：")
for k, pc in enumerate(pieces):
    print(f"  q{k + 1}(q{k + 1}ᵀb) = {np.round(pc, 4)}")
check("Σ qₖ(qₖᵀb) = b", sum(pieces), bb)
check("P = QQᵀ = I（投影到整個空間）", Q @ Q.T, np.eye(3))

# %% [markdown]
# ### Gram–Schmidt 與 $A=QR$
#
# 一句話：**把每個新向量減掉它在已確定方向上的投影**。
#
# $$\mathbf B=\mathbf b-\frac{A^{\mathsf T}\mathbf b}{A^{\mathsf T}A}A,\qquad
# \mathbf C=\mathbf c-\frac{A^{\mathsf T}\mathbf c}{A^{\mathsf T}A}A-\frac{B^{\mathsf T}\mathbf c}{B^{\mathsf T}B}\mathbf B,\ \dots$$
#
# 最後正規化，並把係數收進上三角矩陣：
#
# $$A=QR,\qquad R=Q^{\mathsf T}A=\begin{bmatrix}
# \mathbf q_1^{\mathsf T}\mathbf a&\mathbf q_1^{\mathsf T}\mathbf b&\mathbf q_1^{\mathsf T}\mathbf c\\
# &\mathbf q_2^{\mathsf T}\mathbf b&\mathbf q_2^{\mathsf T}\mathbf c\\ &&\mathbf q_3^{\mathsf T}\mathbf c\end{bmatrix}$$
#
# 最小平方因此化為一次回代：
# $R^{\mathsf T}R\hat{\mathbf x}=R^{\mathsf T}Q^{\mathsf T}\mathbf b
# \Rightarrow R\hat{\mathbf x}=Q^{\mathsf T}\mathbf b$。

# %%
section("4.4 Gram–Schmidt 與 A = QR")

A = np.array([
    [1.0, 2.0, 3.0],
    [-1.0, 0.0, -3.0],
    [0.0, -2.0, 3.0],
])
show_matrix("A（三個獨立但不正交的欄）", A)

# 一步一步做 Gram-Schmidt
Avec, Bvec, Cvec = A[:, 0].copy(), None, None
Bvec = A[:, 1] - (Avec @ A[:, 1]) / (Avec @ Avec) * Avec
Cvec = (A[:, 2] - (Avec @ A[:, 2]) / (Avec @ Avec) * Avec
        - (Bvec @ A[:, 2]) / (Bvec @ Bvec) * Bvec)
print(f"\nA = a            = {Avec}")
print(f"B = b − proj_A b = {Bvec}")
print(f"C = c − proj_A c − proj_B c = {Cvec}")
check("AᵀB = 0", Avec @ Bvec, 0.0)
check("AᵀC = 0", Avec @ Cvec, 0.0)
check("BᵀC = 0", Bvec @ Cvec, 0.0)
print(f"長度 ‖A‖ = {np.linalg.norm(Avec):.4f} (=√2), "
      f"‖B‖ = {np.linalg.norm(Bvec):.4f} (=√6), "
      f"‖C‖ = {np.linalg.norm(Cvec):.4f} (=√3)")

Q, R = qr_gram_schmidt(A)
show_matrix("Q", Q)
show_matrix("R = QᵀA（上三角，對角線 = 那些長度）", R)
check("A = QR", Q @ R, A)
check("QᵀQ = I", Q.T @ Q, np.eye(3))
check("R 上三角", R, np.triu(R))
check("Gram–Schmidt 的 Q 第 1 欄 = a/‖a‖", Q[:, 0], Avec / np.linalg.norm(Avec))
check("AᵀA = RᵀR（所以 Cholesky 與 QR 相通）", A.T @ A, R.T @ R)

# %% [markdown]
# ### 數值穩定性：古典 vs 修正 Gram–Schmidt vs Householder
#
# 對條件極差的矩陣（例如 Vandermonde），古典 Gram–Schmidt 會失去正交性。
# 實務上用 **Householder 反射** 來做 QR（LAPACK / numpy 的做法）。

# %%
section("4.4 三種 QR 的正交性誤差比較")


def classical_gram_schmidt(A):
    """古典 Gram–Schmidt：一次扣掉所有投影（數值上較差）。"""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j] - sum((Q[:, i] @ A[:, j]) * Q[:, i] for i in range(j)) \
            if j else A[:, j].copy()
        Q[:, j] = v / np.linalg.norm(v)
    return Q


print(f"{'矩陣':<28} | {'古典 GS':>12} | {'修正 GS':>12} | {'Householder':>12}")
print("-" * 74)
for n_ in [6, 10, 14]:
    x = np.linspace(0, 1, 3 * n_)
    V = np.vander(x, n_, increasing=True)          # 惡名昭彰的病態矩陣
    errs = []
    for f in [classical_gram_schmidt, gram_schmidt,
              lambda M: householder_qr(M)[0][:, :M.shape[1]]]:
        Qi = f(V)
        errs.append(np.linalg.norm(Qi.T @ Qi - np.eye(n_)))
    print(f"Vandermonde {3 * n_}x{n_:<14} | {errs[0]:>12.2e} | {errs[1]:>12.2e} "
          f"| {errs[2]:>12.2e}")
check("Householder 的正交性誤差最小", errs[2] <= errs[1] <= errs[0] * 10)

# QR 解最小平方：比正規方程更穩定
x_line = np.linspace(0, 1, 40)
Vb = np.vander(x_line, 8, increasing=True)
y_true = np.exp(x_line)
x_normal = least_squares_normal(Vb, y_true)
x_qr = least_squares_qr(Vb, y_true)
print(f"\n用 8 次多項式逼近 e^x（條件數 κ(A) = {np.linalg.cond(Vb):.2e}）：")
print(f"  正規方程殘差 = {np.linalg.norm(Vb @ x_normal - y_true):.3e}")
print(f"  QR 法殘差     = {np.linalg.norm(Vb @ x_qr - y_true):.3e}")
print(f"  κ(AᵀA) = κ(A)² = {np.linalg.cond(Vb.T @ Vb):.2e} ← 正規方程的代價")
check("兩者答案接近", x_normal, x_qr, tol=1e-4)

# %% [markdown]
# ## 4.5　偽逆 $A^{+}$
#
# 單邊反矩陣：
#
# | 情形 | 條件 | 公式 |
# |---|---|---|
# | 左反矩陣 $A^{+}A=I$ | 欄獨立（$r=n\le m$）| $A^{+}=(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ |
# | 右反矩陣 $AA^{+}=I$ | 列獨立（$r=m\le n$）| $A^{+}=A^{\mathsf T}(AA^{\mathsf T})^{-1}$ |
# | 都不滿 | $r<m,\ r<n$ | 由 SVD 或 $A=CR$ 定義 |
#
# 關鍵圖像：**$A$ 從列空間到欄空間是可逆的**（$r\times r$），
# $A^{+}$ 就把這部分反轉回去，並把 $N(A^{\mathsf T})$ 送成 0。
#
# $$A^{+}A=P_{\text{row}},\qquad AA^{+}=P_{\text{col}}$$
#
# $\mathbf x^{+}=A^{+}\mathbf b$ 是**最小範數最小平方解**：既讓 $\|A\mathbf x-\mathbf b\|$ 最小，
# 又在所有這樣的解中長度最短（零空間成分為 0）。

# %%
section("4.5 偽逆：左反、右反與最小範數解")

# 左反矩陣
Atall = np.array([[1.0], [1.0]])
Aplus = pseudoinverse(Atall)
show_matrix("A（2x1，欄獨立）", Atall)
show_matrix("A⁺", Aplus)
check("A⁺A = I（左反）", Aplus @ Atall, np.eye(1))
check("A⁺ = (AᵀA)⁻¹Aᵀ", Aplus, np.linalg.inv(Atall.T @ Atall) @ Atall.T)
check("AA⁺ ≠ I", not np.allclose(Atall @ Aplus, np.eye(2)))

# 右反矩陣
Awide = np.array([[1.0, 1.0]])
Awp = pseudoinverse(Awide)
check("AA⁺ = I（右反）", Awide @ Awp, np.eye(1))
check("A⁺ = Aᵀ(AAᵀ)⁻¹", Awp, Awide.T @ np.linalg.inv(Awide @ Awide.T))

# 對角矩陣的偽逆：能倒就倒，1/0 當 0
D = np.diag([2.0, 3.0, 0.0])
Dp = pseudoinverse(D)
show_matrix("D", D)
show_matrix("D⁺（1/0 視為 0）", Dp)
check("D⁺ = diag(1/2, 1/3, 0)", Dp, np.diag([0.5, 1 / 3, 0.0]))

# 秩不滿：最小範數解
A5 = np.array([[1.0, 1.0], [1.0, 1.0]])
b5 = np.array([3.0, 1.0])
x_plus = min_norm_solution(A5, b5)
print(f"\nA = [[1,1],[1,1]]（秩 1），b = {b5}")
print(f"投影 p = {A5 @ x_plus}（= (2,2)）")
print(f"最小範數解 x⁺ = {x_plus}，‖x⁺‖ = {np.linalg.norm(x_plus):.4f}")
for cand in [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([3.0, -1.0])]:
    same_err = np.allclose(A5 @ cand, A5 @ x_plus)
    print(f"  另一組解 {cand}：誤差相同 = {same_err}，長度 = {np.linalg.norm(cand):.4f}")
check("x⁺ 的長度最短", np.linalg.norm(x_plus) <= 2.0 + 1e-12)
check("x⁺ 在列空間中（零空間成分 = 0）",
      np.linalg.norm(nullspace(A5)[:, 0] @ x_plus), 0.0)

# Penrose 四條件 與 A⁺A = P_row
rng = np.random.default_rng(1)
Atest = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
Ap = pseudoinverse(Atest)
print("\nPenrose 四條件：")
check("  A A⁺ A = A", Atest @ Ap @ Atest, Atest)
check("  A⁺ A A⁺ = A⁺", Ap @ Atest @ Ap, Ap)
check("  (A⁺A)ᵀ = A⁺A", (Ap @ Atest).T, Ap @ Atest)
check("  (AA⁺)ᵀ = AA⁺", (Atest @ Ap).T, Atest @ Ap)
Prow_t = projection_matrix(row_space(Atest))
Pcol_t = projection_matrix(Atest[:, [0, 2]] if rank(Atest) == 2 else Atest)
check("A⁺A = 投影到列空間", Ap @ Atest, Prow_t)
check("(A⁺A)² = A⁺A", (Ap @ Atest) @ (Ap @ Atest), Ap @ Atest)

# A = CR 路線算偽逆（整數矩陣可得精確分數）
C, R = cr_factor(Atest)
A_plus_cr = R.T @ np.linalg.inv(C.T @ Atest @ R.T) @ C.T
check("A⁺ = Rᵀ(CᵀA Rᵀ)⁻¹Cᵀ 與 SVD 版一致", A_plus_cr, Ap, tol=1e-8)

# %% [markdown]
# ### 入射矩陣的偽逆
#
# 圖的入射矩陣秩 $r=n-1$（零空間是常數電位）。它的偽逆把「邊上的電壓降」
# 反推回「節點電位」，並自動選出**和為零**的那一組（最小範數）。
# 這正是電路理論與圖訊號處理的基礎。

# %%
section("4.5 入射矩陣的偽逆")

edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
A6 = np.zeros((len(edges), 4))
for k, (i, j) in enumerate(edges):
    A6[k, i], A6[k, j] = -1.0, 1.0
A6p = pseudoinverse(A6)
show_matrix("入射矩陣 A", A6)
show_matrix("A⁺", A6p)
print(f"rank = {rank(A6)} = n − 1")
check("A 每列和為 0 ⇒ A⁺ 每欄和為 0", A6p.sum(axis=0), np.zeros(5))
x_true = np.array([0.0, 2.0, 5.0, 7.0])
v = A6 @ x_true
x_rec = A6p @ v
print(f"\n真實電位 x = {x_true}（平均 {x_true.mean():.2f}）")
print(f"由電壓降還原 A⁺(Ax) = {np.round(x_rec, 4)}（平均 {x_rec.mean():.2f}）")
check("還原結果與真值相差一個常數", x_true - x_rec, np.full(4, x_true.mean()))
check("A⁺A = 投影到列空間（扣掉常數成分）", A6p @ A6,
      np.eye(4) - np.ones((4, 4)) / 4)

# %% [markdown]
# ## 動手練習
#
# 1. 證明 $P=A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ 滿足 $P^2=P$ 與 $P^{\mathsf T}=P$，
#    並說明 $\operatorname{trace}(P)=r$。
# 2. 用最小平方把你家附近的氣溫資料配上 $a+b\cos(2\pi t/365)+c\sin(2\pi t/365)$。
#    （提示：係數仍是線性的！）
# 3. 對 $t_i$ 先減去平均再配直線，確認 $A^{\mathsf T}A$ 變成對角矩陣。
# 4. 自己寫出 $I-2\mathbf u\mathbf u^{\mathsf T}$，驗證它把 $\mathbf u$ 映到 $-\mathbf u$、
#    把垂直 $\mathbf u$ 的向量保持不動。
#
# 參考解答：

# %%
section("練習參考解答")

rng = np.random.default_rng(7)
Ae = rng.standard_normal((8, 3))
Pe = projection_matrix(Ae)
check("練習 1：P² = P", Pe @ Pe, Pe)
check("練習 1：Pᵀ = P", Pe.T, Pe)
check("練習 1：trace(P) = r = 3", np.trace(Pe), 3.0)

# 練習 2：週期性回歸（用合成資料）
days = np.arange(0, 365, 7.0)
temp = 22 + 8 * np.cos(2 * np.pi * days / 365) + 3 * np.sin(2 * np.pi * days / 365) \
    + 0.8 * rng.standard_normal(len(days))
Aseason = np.column_stack([np.ones_like(days),
                           np.cos(2 * np.pi * days / 365),
                           np.sin(2 * np.pi * days / 365)])
coef = least_squares_normal(Aseason, temp)
print(f"\n練習 2：擬合係數 (a,b,c) = {np.round(coef, 3)}，真值 (22, 8, 3)")
check("  係數接近真值", np.allclose(coef, [22, 8, 3], atol=0.6))

# 練習 3：平移時間讓 AᵀA 對角化
t_raw = np.array([1.0, 3.0, 5.0])
T = t_raw - t_raw.mean()
A_shift = np.column_stack([np.ones_like(T), T])
G_shift = A_shift.T @ A_shift
show_matrix("練習 3：平移後的 AᵀA", G_shift)
check("  非對角元素為 0", G_shift[0, 1], 0.0)

# 練習 4：反射矩陣
u4 = np.array([1.0, 2.0, 2.0])
u4 = u4 / np.linalg.norm(u4)
H = np.eye(3) - 2 * np.outer(u4, u4)
check("練習 4：Hu = −u", H @ u4, -u4)
w = np.array([2.0, -1.0, 0.0])            # 與 u 垂直
w = w - (w @ u4) * u4
check("練習 4：垂直 u 的向量不動", H @ w, w)

# %% [markdown]
# ## 本章重點回顧
#
# * **基本定理第二部分**：$N(A)\perp C(A^{\mathsf T})$、$N(A^{\mathsf T})\perp C(A)$，
#   而且是正交補 ── 每個向量都能唯一分解。
# * 投影：$P=A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$，特徵是 $P^2=P=P^{\mathsf T}$，
#   $\operatorname{trace}P=r$。
# * 最小平方 $A^{\mathsf T}A\hat{\mathbf x}=A^{\mathsf T}\mathbf b$：幾何、代數、微積分三路同歸。
#   最小平方 ↔ 平均數；最小絕對值 ↔ 中位數。
# * 正交矩陣保長保角；$A^{\mathsf T}A$ 退化成 $I$，投影變成一堆 1 維投影相加
#   （= Fourier 級數的原型）。
# * Gram–Schmidt 給出 $A=QR$；實務用 Householder 反射，數值穩定得多。
#   最小平方因此化為 $R\hat{\mathbf x}=Q^{\mathsf T}\mathbf b$ 的回代。
# * 偽逆 $A^{+}$ 統一了反矩陣、左反、右反：它反轉「列空間 → 欄空間」那個 $r\times r$ 的部分，
#   給出最小範數最小平方解。
#
# 下一章：行列式 ── 面積、體積與可逆性的單一數字判準。
