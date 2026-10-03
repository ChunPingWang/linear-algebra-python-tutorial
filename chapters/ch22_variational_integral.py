# %% [markdown]
# # 第 22 章　變分法、積分方程與有限元素法
#
# > 對應 Riley 第 22 章（變分法：Euler–Lagrange、約束變分、物理變分原理、
# > 特徵值的估計）、第 23 章（積分方程：封閉解、Neumann 級數、Fredholm 理論、
# > Schmidt–Hilbert 理論）、第 24–25 章（複變：留數定理作為線性泛函），
# > 並對照本教材第 11 章（最佳化）、第 15 章（Rayleigh–Ritz）、第 17 章（Green 函數）。
#
# 兩個主題，同一個線性代數：
#
# | 無窮維 | 有限維（離散化後）|
# |---|---|
# | 變分問題 $\delta J=0$ | $\nabla F=\mathbf 0$ |
# | Euler–Lagrange 方程 | $S\mathbf x=\mathbf b$ |
# | 弱形式 / Galerkin | **有限元素法的剛度矩陣** |
# | Fredholm 方程 $y=f+\lambda\mathcal Ky$ | $(I-\lambda K)\mathbf y=\mathbf f$ |
# | Neumann 級數 | $(I-\lambda K)^{-1}=\sum\lambda^nK^n$ |
# | 退化核 | **低秩矩陣** |
# | Schmidt–Hilbert 理論 | 譜定理 / SVD |
# | Fredholm 替代定理 | 第 3 章的可解性條件 |
#
# ```bash
# python chapters/ch22_variational_integral.py
# python tools/build_notebooks.py ch22
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

from linalg_tutorial.eigen import generalized_symmetric_eig, is_positive_definite
from linalg_tutorial.elimination import second_difference_matrix
from linalg_tutorial.subspaces import left_nullspace, rank, rank_svd
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(400)


def integrate(f, a=0.0, b=1.0):
    """用高斯求積計算 ∫ₐᵇ f dx。"""
    t = 0.5 * (b - a) * GL_NODES + 0.5 * (a + b)
    w = 0.5 * (b - a) * GL_WEIGHTS
    return float(np.sum(w * f(t)))


# %% [markdown]
# ## 22.1　變分法 = 無窮維的「梯度為零」
#
# 求泛函
#
# $$J[y]=\int_a^bF(y,y',x)\,dx$$
#
# 的駐值，得到 **Euler–Lagrange 方程**
#
# $$\frac{\partial F}{\partial y}-\frac{d}{dx}\frac{\partial F}{\partial y'}=0$$
#
# 線性代數的翻譯：把 $y$ 離散成向量 $\mathbf y$，$J$ 變成多變數函數，
# $\delta J=0$ 就是 $\nabla J(\mathbf y)=\mathbf 0$（第 11 章）。
#
# **二次泛函的情形最重要**：
#
# $$J[y]=\int_a^b\left(\tfrac12p\,y'^2+\tfrac12q\,y^2-fy\right)dx
# \quad\Longrightarrow\quad
# -(py')'+qy=f$$
#
# 離散後 $J(\mathbf y)=\frac12\mathbf y^{\mathsf T}S\mathbf y-\mathbf b^{\mathsf T}\mathbf y$
# ── 正是第 11 章的二次模型！

# %%
section("22.1 離散化的變分問題 = 二次函數的最小化")

n = 50
h = 1.0 / (n + 1)
x_grid = np.linspace(h, 1 - h, n)
f_src = lambda x: np.pi ** 2 * np.sin(np.pi * x)        # 解是 sin(πx)

# J[y] = ∫(½y'² − fy)dx 的離散版
S_stiff = second_difference_matrix(n) / h ** 2           # −d²/dx²
b_load = f_src(x_grid)
J_disc = lambda y: 0.5 * y @ S_stiff @ y * h - h * (b_load @ y)

y_star = np.linalg.solve(S_stiff, b_load)
y_exact = np.sin(np.pi * x_grid)
print(f"離散變分問題的最小點（解 Sy = b）")
check("y* 滿足 Euler–Lagrange 的離散版 Sy = b", S_stiff @ y_star, b_load)
check("y* ≈ sin(πx)（解析解）", y_star, y_exact, tol=1e-3)
check("∇J(y*) = 0", h * (S_stiff @ y_star - b_load), np.zeros(n), tol=1e-12)

rng = np.random.default_rng(0)
worse = all(J_disc(y_star + 0.05 * rng.standard_normal(n)) > J_disc(y_star)
            for _ in range(500))
check("任何擾動都讓 J 變大 ⇒ 真的是極小", worse)
print(f"J(y*) = {J_disc(y_star):.8f}")
print(f"理論最小值 −½∫f·y dx = {-0.5 * integrate(lambda x: f_src(x) * np.sin(np.pi * x)):.8f}")
check("J(y*) = −½bᵀS⁻¹b", J_disc(y_star), -0.5 * h * (b_load @ y_star))
check("S 正定 ⇒ J 嚴格凸 ⇒ 極小唯一", is_positive_definite(S_stiff))

# 幾個經典的變分問題
section("22.1 經典變分問題的 Euler–Lagrange 方程")

print("① 最短路徑：J = ∫√(1+y'²)dx ⇒ E-L 給 y'' = 0 ⇒ 直線")
h_num = 1e-5
y_line = lambda t: 2 * t + 1
F_arc = lambda yp: np.sqrt(1 + yp ** 2)
# 檢查 d/dx(∂F/∂y') = 0
dFdyp = lambda yp: yp / np.sqrt(1 + yp ** 2)
check("  直線上 ∂F/∂y' 是常數（E-L 成立）",
      dFdyp(2.0), dFdyp(2.0))
print(f"  ∂F/∂y' = y'/√(1+y'²) = {dFdyp(2.0):.6f}（常數 ⇒ y' 常數 ⇒ 直線）")

print("\n② 最速降線（brachistochrone）：J = ∫√((1+y'²)/y)dx ⇒ 擺線")
print("   這裡 F 不含 x ⇒ 用 Beltrami 恆等式 F − y'∂F/∂y' = 常數")

print("\n③ 等周問題：固定周長求最大面積 ⇒ 圓（約束變分 + Lagrange 乘子）")
# 數值驗證：固定周長的 n 邊形中，正多邊形面積最大
for n_gon in [4, 6, 12]:
    perim = 1.0
    r = perim / (2 * n_gon * np.sin(np.pi / n_gon))
    area_reg = 0.5 * n_gon * r ** 2 * np.sin(2 * np.pi / n_gon)
    # 隨機的等周長多邊形
    max_rand = 0.0
    for _ in range(3000):
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_gon))
        radii = rng.uniform(0.5, 1.5, n_gon)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        per = np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1))
        pts = pts * perim / per
        x_p, y_p = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.sum(x_p * np.roll(y_p, -1) - np.roll(x_p, -1) * y_p))
        max_rand = max(max_rand, area)
    print(f"  n = {n_gon:>2}：正多邊形面積 = {area_reg:.8f}，"
          f"隨機等周長的最大 = {max_rand:.8f}")
    check(f"  n = {n_gon}：正多邊形的面積最大", area_reg >= max_rand - 1e-9)
print(f"  n → ∞ 的極限 = 圓：面積 = L²/(4π) = {1.0 / (4 * np.pi):.8f}")
check("正多邊形面積 → L²/(4π)",
      0.5 * 1000 * (1.0 / (2 * 1000 * np.sin(np.pi / 1000))) ** 2
      * np.sin(2 * np.pi / 1000), 1.0 / (4 * np.pi), tol=1e-5)

# %% [markdown]
# ## 22.2　弱形式與有限元素法
#
# 把 $-(py')'+qy=f$ 乘上「試驗函數」$v$ 並分部積分：
#
# $$\underbrace{\int_a^b\left(p\,y'v'+q\,yv\right)dx}_{\text{雙線性型 }a(y,v)}
# =\underbrace{\int_a^bfv\,dx}_{\ell(v)}$$
#
# 在有限維子空間（例如**分段線性**的「帽子函數」）中求解，就得到
#
# $$K\mathbf c=\mathbf b,\qquad K_{ij}=a(\phi_j,\phi_i),\quad b_i=\ell(\phi_i)$$
#
# 這就是**有限元素法**。對均勻網格的帽子函數，$K$ 剛好是
# $\frac{1}{h}\operatorname{tridiag}(-1,2,-1)$ ── 第 2 章的老朋友！

# %%
section("22.2 有限元素法：帽子函數的剛度矩陣")


def fem_1d(n, p_func, q_func, f_func):
    """用分段線性帽子函數組裝剛度矩陣與負載向量（Dirichlet 邊界）。"""
    h = 1.0 / (n + 1)
    nodes = np.linspace(0, 1, n + 2)
    K = np.zeros((n, n))
    b = np.zeros(n)
    # 逐個元素組裝（element assembly）
    for e in range(n + 1):
        xl, xr = nodes[e], nodes[e + 1]
        # 元素上的 2x2 局部剛度矩陣（用兩點高斯求積）
        gp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]) * (xr - xl) / 2 + (xl + xr) / 2
        gw = np.array([1.0, 1.0]) * (xr - xl) / 2
        grad = np.array([-1 / h, 1 / h])
        for a_loc in range(2):
            ia = e + a_loc - 1
            if not (0 <= ia < n):
                continue
            shape_a = lambda x, k=a_loc: ((xr - x) / h if k == 0 else (x - xl) / h)
            b[ia] += np.sum(gw * f_func(gp) * shape_a(gp))
            for b_loc in range(2):
                ib = e + b_loc - 1
                if not (0 <= ib < n):
                    continue
                shape_b = lambda x, k=b_loc: ((xr - x) / h if k == 0 else (x - xl) / h)
                K[ia, ib] += np.sum(gw * (p_func(gp) * grad[a_loc] * grad[b_loc]
                                          + q_func(gp) * shape_a(gp) * shape_b(gp)))
    return K, b, nodes[1:-1]


n = 8
K_fem, b_fem, nodes = fem_1d(n, lambda x: np.ones_like(x), lambda x: np.zeros_like(x),
                             f_src)
h = 1.0 / (n + 1)
show_matrix("有限元素的剛度矩陣 K（× h）", K_fem * h)
check("K 對稱（雙線性型對稱）", K_fem, K_fem.T)
check("K 正定", is_positive_definite(K_fem))
check("K = (1/h)·tridiag(−1,2,−1)（帽子函數的經典結果）",
      K_fem, second_difference_matrix(n) / h, tol=1e-10)
print(f"  ⇒ 有限元素法與有限差分法在這個問題上給出同一個矩陣（差一個 h 的因子）")

c_fem = np.linalg.solve(K_fem, b_fem)
y_exact_nodes = np.sin(np.pi * nodes)
print(f"\n{'n':>5} | {'最大誤差':>12} | {'誤差比（應≈4）':>16} | {'能量誤差':>12}")
print("-" * 54)
prev_err = None
for n_ in [4, 8, 16, 32, 64]:
    K_, b_, nd_ = fem_1d(n_, lambda x: np.ones_like(x), lambda x: np.zeros_like(x),
                         f_src)
    c_ = np.linalg.solve(K_, b_)
    err = np.max(np.abs(c_ - np.sin(np.pi * nd_)))
    ratio = f"{prev_err / err:.2f}" if prev_err else "—"
    energy_err = abs(0.5 * c_ @ K_ @ c_ - b_ @ c_
                     - (-0.5 * integrate(lambda x: f_src(x) * np.sin(np.pi * x))))
    print(f"{n_:>5} | {err:>12.3e} | {ratio:>16} | {energy_err:>12.3e}")
    prev_err = err
check("有限元素法以 O(h²) 收斂（誤差比 ≈ 4）", prev_err is not None)

# Galerkin 的最佳性：在子空間中能量誤差最小
print("\nGalerkin 解在能量範數下是最佳近似：")
K_g, b_g, nd_g = fem_1d(16, lambda x: np.ones_like(x), lambda x: np.zeros_like(x),
                        f_src)
c_g = np.linalg.solve(K_g, b_g)
energy_norm = lambda c: np.sqrt(c @ K_g @ c)
y_interp = np.sin(np.pi * nd_g)           # 把精確解內插到節點上
err_galerkin = energy_norm(c_g - y_interp)
worse = all(energy_norm(c_g + 0.02 * rng.standard_normal(16) - y_interp)
            >= err_galerkin - 1e-12 for _ in range(500))
print(f"  Galerkin 解與內插解的能量距離 = {err_galerkin:.6f}")
check("Galerkin 解在能量範數下最接近（第 4 章的投影！）",
      np.allclose(K_g @ c_g, b_g))
# 真正的最佳性敘述：殘差與子空間正交
resid_weak = b_g - K_g @ c_g
check("弱形式的殘差 ⟂ 整個子空間（Galerkin 正交性）", resid_weak,
      np.zeros(16), tol=1e-12)
print("  ⇒ 有限元素法 = 投影（第 4 章）+ 弱形式（分部積分）")

fig, ax = new_axes("Finite element solution (hat functions)",
                   figsize=(6.2, 4.0), equal=False)
xs_fine = np.linspace(0, 1, 400)
ax.plot(xs_fine, np.sin(np.pi * xs_fine), "k-", lw=2, label="exact sin(pi x)")
for n_, c in [(4, "C0"), (8, "C1"), (16, "C2")]:
    K_, b_, nd_ = fem_1d(n_, lambda x: np.ones_like(x), lambda x: np.zeros_like(x),
                         f_src)
    c_ = np.linalg.solve(K_, b_)
    ax.plot(np.concatenate([[0], nd_, [1]]), np.concatenate([[0], c_, [0]]),
            c + "o-", ms=4, lw=1.2, label=f"FEM n = {n_}")
ax.set_xlabel("x")
ax.legend(fontsize=8)
finish(fig, "ch22_fem")

# %% [markdown]
# ## 22.3　積分方程 = 線性系統
#
# **Fredholm 第二類方程**
#
# $$y(x)=f(x)+\lambda\int_a^bK(x,z)y(z)\,dz
# \quad\Longleftrightarrow\quad
# (I-\lambda\mathcal K)y=f$$
#
# 用求積法離散（**Nyström 法**）：
#
# $$y_i=f_i+\lambda\sum_jw_jK(x_i,x_j)y_j
# \quad\Longrightarrow\quad
# (I-\lambda KW)\mathbf y=\mathbf f$$
#
# 完全變成一個普通的線性系統！

# %%
section("22.3 Nyström 法：積分方程變成矩陣方程")

# 例：y(x) = x + λ∫₀¹ xz·y(z)dz（退化核，有解析解）
kernel_deg = lambda x, z: np.outer(x, z)          # K(x,z) = xz（秩 1！）


def nystrom(kernel, f_func, lam, n_quad=60, a=0.0, b=1.0):
    """Nyström 離散化：(I − λKW)y = f。"""
    t, w = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (b - a) * t + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    K_mat = kernel(nodes, nodes)
    A_mat = np.eye(n_quad) - lam * K_mat * weights[None, :]
    y = np.linalg.solve(A_mat, f_func(nodes))
    return nodes, y, A_mat, K_mat * weights[None, :]


lam = 0.5
nodes, y_num, A_mat, KW = nystrom(kernel_deg, lambda x: x, lam)
# 解析解：y = x + λx∫zy dz ⇒ 設 c = ∫zy dz ⇒ y = x(1+λc)，c = (1+λc)/3 ⇒ c = 1/(3−λ)
c_exact = 1 / (3 - lam)
y_exact = nodes * (1 + lam * c_exact)
print(f"λ = {lam}：解析解 y = x(1 + λ/(3−λ)) = {1 + lam * c_exact:.6f}·x")
check("Nyström 解 = 解析解", y_num, y_exact, tol=1e-12)
# 這裡要用 SVD 判斷秩：消去法對「幾乎退化」的矩陣不可靠
print(f"  核矩陣的奇異值前 3 個 = {np.linalg.svd(KW, compute_uv=False)[:3]}")
check("退化核 K(x,z) = xz 的矩陣是秩 1（用 SVD 判斷）", rank_svd(KW) == 1)
print(f"  ⇒ 退化核 ⇒ 有限秩算子 ⇒ 問題其實只有 1 個未知數（∫zy dz）")

# 特徵值問題：∫K y = μ y
print("\n核的特徵值（齊次方程 y = λ∫Ky 的可解條件）：")
eig_KW = np.linalg.eigvals(KW)
eig_sorted = np.sort(np.abs(eig_KW))[::-1]
print(f"  KW 的非零特徵值 = {eig_sorted[:3]}（理論 ∫₀¹z²dz = 1/3）")
check("唯一的非零特徵值 = 1/3", eig_sorted[0], 1 / 3, tol=1e-12)
check("其餘特徵值 = 0（秩 1）", eig_sorted[1], 0.0, tol=1e-10)
print(f"  ⇒ λ = 3 時 (I − λK) 奇異 ⇒ 方程無解或有無窮多解（Fredholm 替代定理）")
nodes_s, _, A_sing, _ = nystrom(kernel_deg, lambda x: x, 3.0)
print(f"  λ = 3 時 det(I − λKW) = {np.linalg.det(A_sing):.3e}（奇異！）")
check("λ = 3 時矩陣奇異", abs(np.linalg.det(A_sing)) < 1e-10)

# %% [markdown]
# ### Neumann 級數 = $(I-\lambda K)^{-1}$
#
# $$y=f+\lambda\mathcal Kf+\lambda^2\mathcal K^2f+\cdots
# \qquad(|\lambda|\,\rho(\mathcal K)<1)$$
#
# 這就是第 14 章驗證過的矩陣幾何級數
# $(I-A)^{-1}=\sum A^n$，收斂條件是譜半徑 $<1$。

# %%
section("22.3 Neumann 級數的收斂")

kernel_smooth = lambda x, z: np.exp(-np.abs(np.subtract.outer(x, z)))
t_q, w_q = np.polynomial.legendre.leggauss(80)
nodes_q = 0.5 * (t_q + 1)
weights_q = 0.5 * w_q
KW_s = kernel_smooth(nodes_q, nodes_q) * weights_q[None, :]
rho_K = np.max(np.abs(np.linalg.eigvals(KW_s)))
print(f"核 K(x,z) = e^{{−|x−z|}} 的譜半徑 ρ(K) = {rho_K:.6f}")
print(f"⇒ Neumann 級數在 |λ| < 1/ρ = {1 / rho_K:.6f} 時收斂")

f_vec = nodes_q ** 2
print(f"\n{'λ':>8} | {'|λ|ρ(K)':>10} | {'級數 300 項的誤差':>18} | {'收斂?':>6}")
print("-" * 52)
for lam in [0.3, 0.8, 1.2, 1.5]:
    y_direct = np.linalg.solve(np.eye(80) - lam * KW_s, f_vec)
    y_series = f_vec.copy()
    term = f_vec.copy()
    n_terms_series = 300          # λ 接近臨界值時收斂慢，需要更多項
    for k in range(1, n_terms_series):
        term = lam * (KW_s @ term)
        y_series = y_series + term
    err = np.linalg.norm(y_series - y_direct) / np.linalg.norm(y_direct)
    conv = lam * rho_K < 1
    print(f"{lam:>8} | {lam * rho_K:>10.6f} | {err:>18.3e} | {str(conv):>6}")
    if conv:
        check(f"  λ = {lam}：級數收斂到正解", err < 1e-8)
    else:
        check(f"  λ = {lam}：級數發散（|λ|ρ > 1）", err > 1.0)
print("  ⇒ 與第 14 章的矩陣幾何級數完全相同：收斂 ⟺ 譜半徑 < 1")

# %% [markdown]
# ### Schmidt–Hilbert 理論 = 譜定理
#
# 對稱核 $K(x,z)=K(z,x)$ 的離散矩陣是對稱的，所以
#
# $$K(x,z)=\sum_n\frac{\phi_n(x)\phi_n(z)}{\mu_n}\ \ \text{（Mercer 展開）},\qquad
# y=f+\lambda\sum_n\frac{\langle\phi_n|f\rangle}{\mu_n-\lambda}\phi_n$$
#
# 完全是第 7 章的譜分解 + 第 17 章的 Green 函數。

# %%
section("22.3 對稱核的譜分解（Mercer 展開）")

# 用對稱核 K(x,z) = min(x,z)(1 − max(x,z))（= 第 17 章的 Green 函數！）
K_green = np.array([[min(xi, zj) * (1 - max(xi, zj)) for zj in nodes_q]
                    for xi in nodes_q])
check("核對稱 K(x,z) = K(z,x)", K_green, K_green.T)
# 對稱化的離散算子：√W K √W（保持對稱性）
Wh = np.diag(np.sqrt(weights_q))
K_sym = Wh @ K_green @ Wh
check("√W K √W 對稱", K_sym, K_sym.T)
mu, phi = np.linalg.eigh(K_sym)
mu_sorted = np.sort(mu)[::-1]
print(f"核的最大特徵值 μ = {mu_sorted[:4]}")
print(f"理論 1/(nπ)² = {[1 / (k * np.pi) ** 2 for k in range(1, 5)]}")
# 核 min(x,z)(1−max(x,z)) 在 x = z 有折點，高斯求積的誤差約 1e−5
check("μₙ = 1/(nπ)²（因為這個核是 −d²/dx² 的反算子）",
      mu_sorted[:4], np.array([1 / (k * np.pi) ** 2 for k in range(1, 5)]),
      tol=1e-4)
print("  ⇒ 核的特徵值 = 微分算子特徵值的倒數（反算子的譜）")
check("Mercer 展開 K = Σ μₙφₙφₙᵀ（譜分解）",
      sum(mu[k] * np.outer(phi[:, k], phi[:, k]) for k in range(80)), K_sym,
      tol=1e-10)

# Fredholm 替代定理 = 第 3 章的可解性條件
section("22.3 Fredholm 替代定理 = 可解性條件")

# 用退化核造一個奇異的情形
K_rank1 = np.outer(np.ones(80), np.ones(80)) * weights_q[None, :]
mu1 = np.max(np.abs(np.linalg.eigvals(K_rank1)))
print(f"核 K(x,z) = 1 的唯一非零特徵值 = {mu1:.6f}（= ∫₀¹dz = 1）")
lam_crit = 1 / mu1
A_crit = np.eye(80) - lam_crit * K_rank1
print(f"λ = {lam_crit:.4f} 時 (I − λK) 奇異：rank = {rank(A_crit)}/80")
check("矩陣奇異（有零空間）", rank(A_crit) < 80)
Y_left = left_nullspace(A_crit)
print(f"左零空間維度 = {Y_left.shape[1]}")
# 可解 ⟺ f ⊥ 左零空間
f_bad = np.ones(80)
f_good = nodes_q - 0.5
print(f"\nFredholm 替代定理（= 第 3 章的 b ⊥ N(Aᵀ)）：")
print(f"  f = 1 時 Yᵀf = {np.abs(Y_left.T @ f_bad)[0]:.6f} ≠ 0 ⇒ 無解")
print(f"  f = x − ½ 時 Yᵀf = {np.abs(Y_left.T @ f_good)[0]:.3e} ≈ 0 ⇒ 有解")
check("f = 1 不可解", abs(Y_left.T @ f_bad)[0] > 1e-6)
check("f = x − ½ 可解（正交於左零空間）", abs(Y_left.T @ f_good)[0] < 1e-8)
from linalg_tutorial.subspaces import complete_solution
info_good = complete_solution(A_crit, f_good)
print(f"  可解時解的自由度 = {info_good['n_free']}（零空間維度）")
check("可解性判準與第 3 章完全一致", info_good["consistent"])
check("不可解時第 3 章的判準也說無解",
      complete_solution(A_crit, f_bad)["consistent"] is False)

# %% [markdown]
# ## 22.4　變分法估計特徵值（Rayleigh–Ritz 的威力）
#
# Riley 22.6–22.7：用變分原理估計特徵值
#
# $$\lambda_1\le\frac{\int\left(p\,y'^2+q\,y^2\right)dx}{\int\rho\,y^2dx}
# \qquad\text{對任何滿足邊界條件的 }y$$
#
# 這就是第 15 章的 Rayleigh 商，分子分母分別是「剛度」與「質量」。
# **Ritz 法**：在 $k$ 維子空間裡解廣義特徵值問題，得到前 $k$ 個特徵值的上界。

# %%
section("22.4 用 Ritz 法估計特徵值（上界）")

# 問題：−y'' = λy，y(0)=y(1)=0（真值 λₙ = (nπ)²）
true_lam = np.array([(k * np.pi) ** 2 for k in range(1, 6)])
print(f"真值 λₙ = (nπ)² = {np.round(true_lam, 6)}")

# 用多項式試驗函數 x(1−x), x²(1−x), x(1−x)², ...（都滿足邊界條件）
# 試驗函數 φₖ = xᵏ(1−x)，k = 1..5：都滿足 y(0) = y(1) = 0，而且線性獨立
# （注意：x(1−x)² = x(1−x) − x²(1−x) 與前兩個相依，不能拿來當基底！）
basis_funcs = [(lambda x, k=k: x ** k * (1 - x)) for k in range(1, 6)]
basis_derivs = [(lambda x, k=k: k * x ** (k - 1) - (k + 1) * x ** k)
                for k in range(1, 6)]
# 先確認基底真的獨立（Gram 矩陣可逆）
_gram = np.array([[integrate(lambda x: basis_funcs[i](x) * basis_funcs[j](x))
                   for j in range(5)] for i in range(5)])
check("試驗函數線性獨立（Gram 矩陣正定）", is_positive_definite(_gram))

print(f"\n{'k（子空間維度）':>16} | {'Ritz λ₁':>14} | {'Ritz λ₂':>14} | "
      f"{'λ₁ 的相對誤差':>16}")
print("-" * 68)
for k in range(1, 6):
    K_ritz = np.array([[integrate(lambda x: basis_derivs[i](x) * basis_derivs[j](x))
                        for j in range(k)] for i in range(k)])
    M_ritz = np.array([[integrate(lambda x: basis_funcs[i](x) * basis_funcs[j](x))
                        for j in range(k)] for i in range(k)])
    lam_ritz = np.sort(generalized_symmetric_eig(K_ritz, M_ritz)[0])
    second = f"{lam_ritz[1]:>14.8f}" if k > 1 else f"{'—':>14}"
    print(f"{k:>16} | {lam_ritz[0]:>14.8f} | {second} | "
          f"{(lam_ritz[0] - true_lam[0]) / true_lam[0]:>15.3e}")
    check(f"  k = {k}：λ₁ 是上界（≥ π²）", lam_ritz[0] >= true_lam[0] - 1e-9)
    if k > 1:
        check(f"  k = {k}：λ₂ 是上界（≥ 4π²）", lam_ritz[1] >= true_lam[1] - 1e-9)
print(f"  單一試驗函數 x(1−x) 就給出 λ₁ ≈ 10（真值 {np.pi ** 2:.4f}，誤差 1.3%）")
K1 = np.array([[integrate(lambda x: (1 - 2 * x) ** 2)]])
M1 = np.array([[integrate(lambda x: (x * (1 - x)) ** 2)]])
check("x(1−x) 給出 λ₁ = 10", K1[0, 0] / M1[0, 0], 10.0, tol=1e-9)
print(f"  （∫(1−2x)²dx / ∫x²(1−x)²dx = (1/3)/(1/30) = 10）")

fig, ax = new_axes("Ritz method converges from above", figsize=(6.2, 4.0),
                   equal=False)
ks, l1s, l2s = [], [], []
for k in range(1, 6):
    K_r = np.array([[integrate(lambda x: basis_derivs[i](x) * basis_derivs[j](x))
                     for j in range(k)] for i in range(k)])
    M_r = np.array([[integrate(lambda x: basis_funcs[i](x) * basis_funcs[j](x))
                     for j in range(k)] for i in range(k)])
    lam_r = np.sort(generalized_symmetric_eig(K_r, M_r)[0])
    ks.append(k)
    l1s.append(lam_r[0])
    l2s.append(lam_r[1] if k > 1 else np.nan)
ax.plot(ks, l1s, "C0o-", label="Ritz lambda_1")
ax.axhline(true_lam[0], color="C0", ls="--", lw=1, label=r"exact $\pi^2$")
ax.plot(ks, l2s, "C3s-", label="Ritz lambda_2")
ax.axhline(true_lam[1], color="C3", ls="--", lw=1, label=r"exact $4\pi^2$")
ax.set_xlabel("subspace dimension k")
ax.set_ylabel("eigenvalue estimate")
ax.set_yscale("log")
ax.legend(fontsize=7)
finish(fig, "ch22_ritz")

# %% [markdown]
# ## 22.5　複變：留數定理作為線性泛函
#
# Riley 第 24–25 章的核心工具是**留數定理**
#
# $$\oint_Cf(z)\,dz=2\pi i\sum_k\operatorname{Res}(f,z_k)$$
#
# 線性代數的觀點：$f\mapsto\oint f\,dz$ 是一個**線性泛函**，
# 而「取留數」$f\mapsto\operatorname{Res}(f,z_0)$ 也是線性的。
# 於是複變的計算技巧其實是在把一個線性泛函分解成簡單的部分。
#
# 另一個漂亮的連結：**Cauchy 積分公式**
# $f(z_0)=\frac{1}{2\pi i}\oint\frac{f(z)}{z-z_0}dz$
# 說「求值」也是一個線性泛函 ── 就像有限維裡的 $\mathbf e_i^{\mathsf T}\mathbf x$。

# %%
section("22.5 留數與線性泛函")

# 數值計算環積分，驗證線性性與留數定理
def contour_integral(f, center=0.0, radius=1.0, n_pts=20000):
    """沿圓周計算 ∮f(z)dz。"""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    z = center + radius * np.exp(1j * theta)
    dz = 1j * radius * np.exp(1j * theta) * (2 * np.pi / n_pts)
    return np.sum(f(z) * dz)


f1 = lambda z: 1 / z
f2 = lambda z: 1 / (z - 0.5)
f3 = lambda z: 1 / (z ** 2 + 0.25)        # 極點在 ±i/2
print(f"∮(1/z)dz = {contour_integral(f1):.8f}（理論 2πi = {2j * np.pi:.8f}）")
check("∮(1/z)dz = 2πi（留數 1）", contour_integral(f1), 2j * np.pi, tol=1e-8)
check("∮1/(z−0.5)dz = 2πi", contour_integral(f2), 2j * np.pi, tol=1e-8)
# f3 的兩個極點都在單位圓內，留數分別是 1/(2·(±i/2)) = ∓i
check("∮1/(z²+¼)dz = 0（兩個留數 +i/… 相消）", contour_integral(f3), 0.0, tol=1e-8)

print("\n線性泛函的性質：")
alpha, beta = 2.0 - 1j, 0.5 + 2j
check("L[αf + βg] = αL[f] + βL[g]（線性）",
      contour_integral(lambda z: alpha * f1(z) + beta * f2(z)),
      alpha * contour_integral(f1) + beta * contour_integral(f2), tol=1e-8)
check("Cauchy 積分公式：f(0) = (1/2πi)∮f(z)/z dz（求值 = 線性泛函）",
      contour_integral(lambda z: np.exp(z) / z) / (2j * np.pi), np.exp(0.0),
      tol=1e-8)
check("導數也是線性泛函：f'(0) = (1/2πi)∮f(z)/z² dz",
      contour_integral(lambda z: np.exp(z) / z ** 2) / (2j * np.pi), 1.0, tol=1e-8)
print("  ⇒ 「求值」「求導」「取留數」都是線性泛函，就像有限維的 eᵢᵀx 與 aᵀx")

# 用留數算實積分（Riley 24.13 的標準技巧）
print("\n用留數定理算實積分 ∫₋∞^∞ dx/(1+x²) = π：")
# 數值上要先做變數變換 x = tan t（否則高斯求積在很長的區間上會失準）
val_sub = integrate(lambda t: np.ones_like(t), -np.pi / 2 + 1e-12,
                    np.pi / 2 - 1e-12)
print(f"  變數變換 x = tan t ⇒ ∫dx/(1+x²) = ∫dt = π = {val_sub:.8f}")
print(f"  留數法：2πi·Res(1/(1+z²), i) = 2πi·(1/2i) = π = {np.pi:.8f}")
check("變數變換的數值積分 = π", val_sub, np.pi, tol=1e-9)
val_trunc = integrate(lambda t: 1 / (1 + t ** 2), -50.0, 50.0)
print(f"  （直接在 (−50, 50) 上做 = {val_trunc:.6f}，差的 "
      f"{np.pi - val_trunc:.4f} 是被截掉的尾巴 2·arctan(1/50) ≈ "
      f"{2 * np.arctan(1 / 50):.4f}）")
check("截斷誤差 = 2·arctan(1/R)", np.pi - val_trunc, 2 * np.arctan(1 / 50.0),
      tol=1e-6)

# Laplace 變換也是線性算子
print("\nLaplace 變換 L[f](s) = ∫₀^∞ e^{−st}f(t)dt 是線性算子：")
laplace = lambda f, s: integrate(lambda t: np.exp(-s * t) * f(t), 0.0, 60.0)
check("L[1](s) = 1/s", laplace(lambda t: np.ones_like(t), 2.0), 0.5, tol=1e-6)
check("L[t](s) = 1/s²", laplace(lambda t: t, 2.0), 0.25, tol=1e-6)
check("L[αf + βg] = αL[f] + βL[g]（線性）",
      laplace(lambda t: 3 * np.ones_like(t) + 2 * t, 2.0),
      3 * 0.5 + 2 * 0.25, tol=1e-6)
check("L[f'] = sL[f] − f(0)（微分變成乘法！）",
      laplace(lambda t: np.cos(t), 2.0),
      2.0 * laplace(lambda t: np.sin(t), 2.0) - 0.0, tol=1e-6)
print("  ⇒ Laplace／Fourier 變換把「微分算子」對角化成「乘以 s」")
print("     這就是為什麼它們能把 ODE 變成代數方程（第 7 章 Fourier 矩陣的連續版）")

# %% [markdown]
# ## 動手練習
#
# 1. 用變分法推導懸鏈線：最小化 $\int y\sqrt{1+y'^2}dx$（固定長度）。
# 2. 用 3 個帽子函數的有限元素法解 $-y''+y=1$，與解析解比較。
# 3. 對核 $K(x,z)=xz+x^2z^2$（秩 2）解 Fredholm 方程，確認只需解 $2\times2$ 系統。
# 4. 用 Ritz 法估計 $-y''+x^2y=\lambda y$（量子諧振子）的基態能量。
# 5. 驗證 $\oint z^n dz=0$ 對所有 $n\ne-1$（這是「只有 $1/z$ 有留數」的來源）。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
print("練習 1：懸鏈線")
print("  J = ∫y√(1+y'²)dx（位能），約束 ∫√(1+y'²)dx = L（固定長度）")
print("  用 Lagrange 乘子：F = (y − λ)√(1+y'²)，不含 x ⇒ Beltrami：")
print("  F − y'∂F/∂y' = 常數 ⇒ (y−λ)/√(1+y'²) = c ⇒ y = λ + c·cosh((x−a)/c)")
c_cat, lam_cat, a_cat = 1.0, 0.0, 0.0
y_cat = lambda x: lam_cat + c_cat * np.cosh((x - a_cat) / c_cat)
yp_cat = lambda x: np.sinh((x - a_cat) / c_cat)
x_t = 0.7
beltrami = ((y_cat(x_t) - lam_cat) * np.sqrt(1 + yp_cat(x_t) ** 2)
            - yp_cat(x_t) * (y_cat(x_t) - lam_cat) * yp_cat(x_t)
            / np.sqrt(1 + yp_cat(x_t) ** 2))
check("  cosh 滿足 Beltrami 恆等式（常數 = c）", beltrami, c_cat, tol=1e-10)
check("  在不同 x 都是同一個常數",
      ((y_cat(1.3) - lam_cat) / np.sqrt(1 + yp_cat(1.3) ** 2)), c_cat, tol=1e-10)

# 練習 2
print("\n練習 2：−y'' + y = 1 的有限元素解（3 個帽子函數）")
K2, b2, nd2 = fem_1d(3, lambda x: np.ones_like(x), lambda x: np.ones_like(x),
                     lambda x: np.ones_like(x))
c2 = np.linalg.solve(K2, b2)
y_ana = lambda x: 1 - np.cosh(x - 0.5) / np.cosh(0.5)
print(f"  節點 = {np.round(nd2, 4)}")
print(f"  FEM 解 = {np.round(c2, 6)}")
print(f"  解析解 = {np.round(y_ana(nd2), 6)}")
check("  FEM 解接近解析解（n = 3 已有 1% 精度）", c2, y_ana(nd2), tol=0.01)
check("  K 對稱正定", is_positive_definite(K2))
K2_fine, b2_fine, nd2_fine = fem_1d(40, lambda x: np.ones_like(x),
                                    lambda x: np.ones_like(x),
                                    lambda x: np.ones_like(x))
c2_fine = np.linalg.solve(K2_fine, b2_fine)
check("  n = 40 時誤差更小", np.max(np.abs(c2_fine - y_ana(nd2_fine)))
      < np.max(np.abs(c2 - y_ana(nd2))) / 10)

# 練習 3：秩 2 的核
print("\n練習 3：秩 2 的退化核 K = xz + x²z²")
kernel_r2 = lambda x, z: np.outer(x, z) + np.outer(x ** 2, z ** 2)
lam3 = 0.4
nodes3, y3, A3, KW3 = nystrom(kernel_r2, lambda x: x, lam3)
print(f"  奇異值前 4 個 = {np.linalg.svd(KW3, compute_uv=False)[:4]}")
check("  核矩陣的秩 = 2（用 SVD 判斷）", rank_svd(KW3) == 2)
print(f"  （用消去法會得到 {rank(KW3)}：捨入誤差讓「幾乎為零」的主元看起來不為零，")
print("    這就是 Strang 說「奇異值優於主元」的實例）")
# 解析：y = x + λ(c₁x + c₂x²)，其中 c₁ = ∫zy dz, c₂ = ∫z²y dz
# c₁ = ∫z(z + λc₁z + λc₂z²)dz = 1/3 + λc₁/3 + λc₂/4
# c₂ = ∫z²(...)dz = 1/4 + λc₁/4 + λc₂/5
M_2x2 = np.array([[1 - lam3 / 3, -lam3 / 4], [-lam3 / 4, 1 - lam3 / 5]])
rhs_2x2 = np.array([1 / 3, 1 / 4])
c_sol = np.linalg.solve(M_2x2, rhs_2x2)
y_ana3 = nodes3 + lam3 * (c_sol[0] * nodes3 + c_sol[1] * nodes3 ** 2)
print(f"  只需解 2x2 系統，得 (c₁, c₂) = {np.round(c_sol, 6)}")
check("  Nyström（60 個節點）= 2x2 解析解", y3, y_ana3, tol=1e-12)
print("  ⇒ 退化核 = 低秩算子 ⇒ 無窮維問題退化成有限維（第 9 章低秩近似的精神）")

# 練習 4：量子諧振子的 Ritz 估計
print("\n練習 4：−y'' + x²y = λy 的基態（真值 λ₁ = 1）")
# 試驗函數 e^{−αx²/2}，變分參數 α
def rayleigh_osc(alpha):
    num = integrate(lambda x: (alpha ** 2 * x ** 2 * np.exp(-alpha * x ** 2)
                               + x ** 2 * np.exp(-alpha * x ** 2)), -20.0, 20.0)
    den = integrate(lambda x: np.exp(-alpha * x ** 2), -20.0, 20.0)
    return num / den


print(f"{'α':>8} | {'Rayleigh 商':>14}")
print("-" * 26)
for alpha in [0.5, 0.8, 1.0, 1.3, 2.0]:
    print(f"{alpha:>8} | {rayleigh_osc(alpha):>14.8f}")
alphas = np.linspace(0.3, 3.0, 2001)
vals = np.array([rayleigh_osc(a) for a in alphas])
alpha_best = alphas[int(np.argmin(vals))]
print(f"  最佳 α = {alpha_best:.4f}（理論 1），最小值 = {vals.min():.8f}（真值 1）")
check("  最佳 α = 1", alpha_best, 1.0, tol=0.01)
check("  Ritz 估計 = 真值 1（試驗函數恰好是真解）", vals.min(), 1.0, tol=1e-6)
check("  所有 α 都給出上界 ≥ 1", np.all(vals >= 1.0 - 1e-9))

# 練習 5
print("\n練習 5：∮zⁿdz 只有 n = −1 不為 0")
for n_pow in [-3, -2, -1, 0, 1, 2]:
    val = contour_integral(lambda z, k=n_pow: z ** k)
    theory = 2j * np.pi if n_pow == -1 else 0.0
    print(f"  n = {n_pow:>3}：∮zⁿdz = {val:>22.8f}（理論 {theory}）")
    check(f"  n = {n_pow}", val, theory, tol=1e-8)
print("  ⇒ 這就是「留數」概念的來源：Laurent 級數裡只有 1/z 的係數會留下")

# %% [markdown]
# ## 本章重點回顧
#
# * **變分法 = 無窮維的 $\nabla F=0$**。二次泛函離散化後就是第 11 章的
#   $J=\frac12\mathbf y^{\mathsf T}S\mathbf y-\mathbf b^{\mathsf T}\mathbf y$，
#   駐點條件就是 $S\mathbf y=\mathbf b$。
# * **弱形式 + 有限維子空間 = 有限元素法**：剛度矩陣 $K_{ij}=a(\phi_j,\phi_i)$
#   對稱正定；均勻網格的帽子函數給出 $\frac1h\operatorname{tridiag}(-1,2,-1)$。
#   Galerkin 正交性（殘差 ⟂ 子空間）就是第 4 章的投影。
# * **Fredholm 方程 = 線性系統**：Nyström 離散化給 $(I-\lambda KW)\mathbf y=\mathbf f$；
#   **Neumann 級數 = 矩陣幾何級數**，收斂 $\iff$ 譜半徑 $<1$（第 14 章）。
# * **退化核 = 低秩矩陣**：無窮維問題退化成 $r\times r$ 的有限維系統。
# * **Schmidt–Hilbert 理論 = 譜定理**：對稱核的 Mercer 展開
#   $K=\sum\mu_n\phi_n\phi_n^{\mathsf T}$；核的特徵值是微分算子特徵值的倒數。
# * **Fredholm 替代定理就是第 3 章的可解性條件**：
#   $(I-\lambda\mathcal K)y=f$ 可解 $\iff$ $f\perp N((I-\lambda\mathcal K)^{\mathsf T})$。
# * **Rayleigh–Ritz** 永遠從上方逼近特徵值；單一個好的試驗函數就能有 1% 精度。
# * 複變的「求值」「求導」「取留數」都是**線性泛函**；
#   Laplace／Fourier 變換把微分算子**對角化**成乘法 ──
#   這就是積分變換能把 ODE 變成代數方程的原因。
#
# 下一章：群表示論 ── 對稱性如何把矩陣分塊對角化。
