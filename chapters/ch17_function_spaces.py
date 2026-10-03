# %% [markdown]
# # 第 17 章　無窮維線性代數：Fourier 級數、正交函數與 Sturm–Liouville
#
# > 對應 Riley 第 12 章（Fourier 級數）、第 17 章（特徵函數法：函數集合、
# > 自伴與 Hermitian 算子、Sturm–Liouville 方程、Green 函數）、
# > 第 18 章（特殊函數：Legendre、Chebyshev、Hermite、Laguerre、Bessel），
# > 並對照 Strang 4.4（正交基底）、6.3（譜定理）、8.3（函數空間的好基底）。
#
# 本章的唯一訊息：**把向量換成函數，線性代數一字不改**。
#
# | 有限維（第 1–13 章）| 無窮維（本章）|
# |---|---|
# | $\mathbf x\in\mathbb R^n$ | $f(x)$ on $[a,b]$ |
# | $\mathbf x^{\mathsf T}\mathbf y=\sum x_iy_i$ | $\langle f|g\rangle=\int_a^b f^{*}g\,\rho\,dx$ |
# | 正交基底 $\mathbf q_i$ | 正交函數 $\hat\phi_n(x)$ |
# | $\mathbf x=\sum(\mathbf q_i^{\mathsf T}\mathbf x)\mathbf q_i$ | $f=\sum\langle\hat\phi_n|f\rangle\hat\phi_n$（Fourier！）|
# | 對稱矩陣 $S=S^{\mathsf T}$ | 自伴算子 $\mathcal L=\mathcal L^{\dagger}$ |
# | $S\mathbf q=\lambda\mathbf q$ | $\mathcal Ly=\lambda\rho y$（Sturm–Liouville）|
# | $S=Q\Lambda Q^{\mathsf T}$ | 特徵函數展開 |
# | $S^{-1}=Q\Lambda^{-1}Q^{\mathsf T}$ | **Green 函數** |
#
# ```bash
# python chapters/ch17_function_spaces.py
# python tools/build_notebooks.py ch17
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

from linalg_tutorial.eigen import generalized_symmetric_eig
from linalg_tutorial.elimination import second_difference_matrix
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# 高斯–勒讓德求積：把積分變成有限和（離散化的內積）
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(600)


def inner(f, g, rho=None, a=-1.0, b=1.0):
    """⟨f|g⟩ = ∫ₐᵇ f*(x)g(x)ρ(x)dx，用高斯求積精確計算。"""
    t = 0.5 * (b - a) * GL_NODES + 0.5 * (a + b)
    w = 0.5 * (b - a) * GL_WEIGHTS
    vals = np.conj(f(t)) * g(t) * (1.0 if rho is None else rho(t))
    return np.sum(w * vals)


norm = lambda f, rho=None, a=-1.0, b=1.0: np.sqrt(np.real(inner(f, f, rho, a, b)))

# %% [markdown]
# ## 17.1　函數空間是向量空間
#
# 公設逐條對應（加法封閉、純量乘法、零元素、負元素…）。
# 加上內積
#
# $$\langle f|g\rangle=\int_a^b f^{*}(x)g(x)\rho(x)\,dx$$
#
# 就成為 **Hilbert 空間**。$\rho(x)\ge0$ 是**權重函數**（由座標系決定）。
#
# 三個不等式一字不改地成立：Schwarz、三角、Bessel。

# %%
section("17.1 函數空間的內積與不等式")

f1 = lambda x: x ** 2 - 0.3
f2 = lambda x: np.sin(3 * x)
f3 = lambda x: np.exp(x)

print(f"⟨f₁|f₂⟩ = {inner(f1, f2):.10f}")
print(f"⟨f₂|f₁⟩ = {inner(f2, f1):.10f}")
check("實函數：⟨f|g⟩ = ⟨g|f⟩", inner(f1, f2), inner(f2, f1))
check("線性：⟨f|2g+3h⟩ = 2⟨f|g⟩+3⟨f|h⟩",
      inner(f1, lambda x: 2 * f2(x) + 3 * f3(x)),
      2 * inner(f1, f2) + 3 * inner(f1, f3))
print(f"‖f₁‖ = {norm(f1):.10f}，‖f₂‖ = {norm(f2):.10f}")
check("Schwarz：|⟨f|g⟩| ≤ ‖f‖‖g‖", abs(inner(f1, f2)) <= norm(f1) * norm(f2) + 1e-12)
check("三角：‖f+g‖ ≤ ‖f‖+‖g‖",
      norm(lambda x: f1(x) + f2(x)) <= norm(f1) + norm(f2) + 1e-12)
check("平行四邊形恆等式",
      norm(lambda x: f1(x) + f2(x)) ** 2 + norm(lambda x: f1(x) - f2(x)) ** 2,
      2 * norm(f1) ** 2 + 2 * norm(f2) ** 2)

# 複數函數：e^{inx} 在 [0, 2π] 上正交
print("\n複數指數基底 e^{inx} 在 [0, 2π] 上的正交性：")
for n, m in [(1, 1), (1, 2), (3, -3), (2, 5)]:
    val = inner(lambda x: np.exp(1j * n * x), lambda x: np.exp(1j * m * x),
                a=0.0, b=2 * np.pi)
    expected = 2 * np.pi if n == m else 0.0
    print(f"  ⟨e^{{i{n}x}}|e^{{i{m}x}}⟩ = {val.real:>10.6f} + {val.imag:>10.6f}i"
          f"（理論 {expected:.4f}）")
    check(f"  n={n}, m={m}", val.real, expected, tol=1e-8)
print("  ⇒ 這就是量子力學與訊號處理最重要的正交基底")

# %% [markdown]
# ### Gram–Schmidt 作用在 $1,x,x^2,\dots$ → Legendre 多項式
#
# 第 4 章的 Gram–Schmidt 一字不改地搬到函數空間：
#
# $$\phi_n=y_n-\sum_{k<n}\hat\phi_k\langle\hat\phi_k|y_n\rangle$$
#
# 從單項式出發，在 $[-1,1]$、$\rho=1$ 下正交化，得到的就是
# **Legendre 多項式**（Riley 17.1 的例子）。換權重就換出別的正交多項式家族。

# %%
section("17.1 Gram–Schmidt 生出 Legendre 多項式")


def gram_schmidt_functions(basis, rho=None, a=-1.0, b=1.0, n_max=5):
    """對函數做 Gram–Schmidt，回傳正交歸一化後的函數清單（以係數向量表示）。"""
    t = 0.5 * (b - a) * GL_NODES + 0.5 * (a + b)
    w = 0.5 * (b - a) * GL_WEIGHTS * (1.0 if rho is None else rho(t))
    V = np.array([basis(k)(t) for k in range(n_max)])      # 每列是一個基底函數的取樣
    Q = np.zeros_like(V)
    for k in range(n_max):
        v = V[k].copy()
        for j in range(k):
            v -= (w * Q[j] * V[k]).sum() * Q[j]
        v /= np.sqrt((w * v * v).sum())
        Q[k] = v
    return Q, t, w


Q_leg, t_grid, w_grid = gram_schmidt_functions(lambda k: (lambda x: x ** k), n_max=5)
print("Gram–Schmidt 的結果 vs 正規化的 Legendre 多項式：")
for n in range(5):
    P_n = np.polynomial.legendre.Legendre.basis(n)(t_grid)
    P_n = P_n / np.sqrt((w_grid * P_n * P_n).sum())
    sign = np.sign((w_grid * Q_leg[n] * P_n).sum())
    err = np.max(np.abs(sign * Q_leg[n] - P_n))
    print(f"  n = {n}：最大差異 = {err:.3e}")
    check(f"  φ̂{n} = 正規化的 P{n}", sign * Q_leg[n], P_n, tol=1e-10)
print("前三個：φ̂₀ = √(1/2)，φ̂₁ = √(3/2)x，φ̂₂ = ½√(5/2)(3x²−1)  ← Riley 17.1 的答案")
check("φ̂₀ = √(1/2)", Q_leg[0], np.full_like(t_grid, np.sqrt(0.5)), tol=1e-12)
check("φ̂₁ = √(3/2)·x", np.abs(Q_leg[1]), np.abs(np.sqrt(1.5) * t_grid), tol=1e-12)
check("φ̂₂ = ½√(5/2)(3x²−1)", np.abs(Q_leg[2]),
      np.abs(0.5 * np.sqrt(2.5) * (3 * t_grid ** 2 - 1)), tol=1e-10)

# 換權重 → 換正交多項式家族
print("\n換權重函數，就換出不同的正交多項式家族：")
families = [
    ("ρ = 1 on [−1,1]", None, -1.0, 1.0, "Legendre"),
    ("ρ = 1/√(1−x²)", lambda x: 1 / np.sqrt(np.maximum(1 - x ** 2, 1e-300)), -1.0, 1.0,
     "Chebyshev"),
    ("ρ = e^{−x²} on R", lambda x: np.exp(-x ** 2), -8.0, 8.0, "Hermite"),
    ("ρ = e^{−x} on [0,∞)", lambda x: np.exp(-x), 0.0, 40.0, "Laguerre"),
]
for label, rho, a_, b_, name in families:
    Qf, tf, wf = gram_schmidt_functions(lambda k: (lambda x: x ** k), rho, a_, b_, 4)
    gram = np.array([[(wf * Qf[i] * Qf[j]).sum() for j in range(4)] for i in range(4)])
    print(f"  {label:<22} → {name:<10} 正交性誤差 = "
          f"{np.max(np.abs(gram - np.eye(4))):.2e}")
    check(f"  {name} 家族正交歸一", gram, np.eye(4), tol=1e-8)
print("  ⇒ 「正交多項式」不是魔法，就是對不同權重做 Gram–Schmidt")

# %% [markdown]
# ## 17.2　Fourier 級數 = 正交投影
#
# $$f(x)=\sum_n c_n\hat\phi_n(x),\qquad
# \boxed{c_n=\langle\hat\phi_n|f\rangle}$$
#
# 和第 4 章的 $\hat{\mathbf x}=Q^{\mathsf T}\mathbf b$ 一模一樣！
# 截斷到前 $N$ 項就是**投影到有限維子空間** ── 因此它是該子空間裡的
# **最小平方最佳近似**。
#
# * **Bessel 不等式**：$\sum_{n\le N}|c_n|^2\le\|f\|^2$
# * **Parseval 恆等式**（完備時）：$\sum_n|c_n|^2=\|f\|^2$

# %%
section("17.2 Fourier 係數 = 內積；截斷 = 最小平方投影")

# 在 [−π, π] 上用 {1, cos nx, sin nx} 展開方波
L_half = np.pi
square = lambda x: np.sign(np.sin(x))


def fourier_coeffs(f, N, a=-np.pi, b=np.pi):
    """回傳 (a₀, aₙ, bₙ)：f ≈ a₀/2 + Σ(aₙcos nx + bₙsin nx)。"""
    a0 = inner(lambda x: np.ones_like(x), f, a=a, b=b) / np.pi
    an = np.array([inner(lambda x, n=n: np.cos(n * x), f, a=a, b=b) / np.pi
                   for n in range(1, N + 1)])
    bn = np.array([inner(lambda x, n=n: np.sin(n * x), f, a=a, b=b) / np.pi
                   for n in range(1, N + 1)])
    return np.real(a0), np.real(an), np.real(bn)


N = 15
a0, an, bn = fourier_coeffs(square, N)
print(f"方波的 Fourier 係數：a₀ = {a0:.6f}，aₙ 全部 ≈ 0（奇函數）")
print(f"  b₁, b₂, ..., b₆ = {np.round(bn[:6], 6)}")
print(f"  理論：bₙ = 4/(nπ) 對奇數 n，0 對偶數 n")
check("a₀ = 0（平均為 0）", a0, 0.0, tol=1e-8)
check("aₙ = 0（奇函數沒有餘弦成分）", an, np.zeros(N), tol=1e-8)
for n in [1, 3, 5]:
    # 方波不連續 ⇒ 高斯求積（假設被積函數平滑）有 O(1e−5) 的誤差
    check(f"  b{n} = 4/({n}π)（求積誤差 < 1e−4）", bn[n - 1], 4 / (n * np.pi),
          tol=1e-4)
for n in [2, 4, 6]:
    # 方波不連續，高斯求積會有 O(1e−5) 的誤差（被積函數不是多項式）
    check(f"  b{n} = 0（求積誤差 < 1e−4）", bn[n - 1], 0.0, tol=1e-4)


def partial_sum(x, a0, an, bn, N):
    out = np.full_like(x, a0 / 2)
    for n in range(1, N + 1):
        out = out + an[n - 1] * np.cos(n * x) + bn[n - 1] * np.sin(n * x)
    return out


# 截斷 = 最小平方投影：隨機擾動係數只會變差
print("\n截斷的 Fourier 級數是該子空間裡的最佳近似：")
err_best = norm(lambda x: square(x) - partial_sum(x, a0, an, bn, 5),
                a=-np.pi, b=np.pi)
rng = np.random.default_rng(0)
worse = True
for _ in range(300):
    da = 0.05 * rng.standard_normal(5)
    db = 0.05 * rng.standard_normal(5)
    e = norm(lambda x: square(x) - partial_sum(x, a0, an[:5] + da, bn[:5] + db, 5),
             a=-np.pi, b=np.pi)
    worse = worse and (e >= err_best - 1e-12)
print(f"  最佳（Fourier）誤差 = {err_best:.8f}")
check("任何擾動都讓誤差變大 ⇒ Fourier 係數就是最小平方解", worse)

# Bessel 與 Parseval
energy = norm(square, a=-np.pi, b=np.pi) ** 2
print(f"\n‖f‖² = ∫f² dx = {energy:.8f}（方波 = 2π）")
print(f"{'N':>5} | {'Σ 能量（前 N 項）':>20} | {'佔比':>8} | {'L² 誤差':>12}")
print("-" * 54)
for N_ in [1, 3, 5, 15, 50]:
    a0_, an_, bn_ = fourier_coeffs(square, N_)
    partial_energy = np.pi * (a0_ ** 2 / 2 + np.sum(an_ ** 2) + np.sum(bn_ ** 2))
    err = norm(lambda x: square(x) - partial_sum(x, a0_, an_, bn_, N_),
               a=-np.pi, b=np.pi)
    print(f"{N_:>5} | {partial_energy:>20.8f} | {partial_energy / energy:>7.2%} "
          f"| {err:>12.3e}")
    check(f"  N = {N_}：Bessel 不等式", partial_energy <= energy + 1e-8)
    check(f"  N = {N_}：誤差² = ‖f‖² − 部分能量", err ** 2,
          energy - partial_energy, tol=1e-6)
print("  ⇒ 「誤差² = 剩下的能量」正是第 9 章 Eckart–Young 的無窮維版本")

# Gibbs 現象
a0_g, an_g, bn_g = fourier_coeffs(square, 60)
xs = np.linspace(-np.pi, np.pi, 4000)
overshoot = partial_sum(xs, a0_g, an_g, bn_g, 60).max()
print(f"\nGibbs 現象：N = 60 時最大過衝 = {overshoot:.6f}")
print(f"  理論極限 = (2/π)∫₀^π sinc = {1.17897974:.6f}（不隨 N 消失！）")
check("過衝約 1.179（Gibbs 常數）", overshoot, 1.17898, tol=0.02)
print("  原因：方波不連續 ⇒ L² 收斂但非一致收斂（逐點不均勻）")

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
ax.plot(xs, square(xs), "k-", lw=2, label="square wave")
for N_, c in [(1, "C0"), (3, "C1"), (15, "C2")]:
    a0_, an_, bn_ = fourier_coeffs(square, N_)
    ax.plot(xs, partial_sum(xs, a0_, an_, bn_, N_), c, lw=1.3, label=f"N = {N_}")
ax.set_title("Fourier partial sums (Gibbs overshoot)", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
Ns = np.arange(1, 60)
errs = []
for N_ in Ns:
    a0_, an_, bn_ = fourier_coeffs(square, int(N_))
    errs.append(norm(lambda x: square(x) - partial_sum(x, a0_, an_, bn_, int(N_)),
                     a=-np.pi, b=np.pi))
ax.loglog(Ns, errs, "o-", ms=3, label="L2 error of square wave")
ax.loglog(Ns, 2.0 / np.sqrt(Ns), "--", label=r"$\propto N^{-1/2}$")
ax.set_xlabel("N")
ax.set_ylabel("L2 error")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
finish(fig, "ch17_fourier")

# %% [markdown]
# ## 17.3　自伴算子 = 對稱矩陣
#
# 伴隨算子由內積定義（和第 2 章的轉置一模一樣）：
#
# $$\langle f|\mathcal Lg\rangle=\langle\mathcal L^{\dagger}f|g\rangle$$
#
# 分部積分告訴我們：
#
# $$\left(\frac{d}{dx}\right)^{\dagger}=-\frac{d}{dx}\ (\text{反對稱}),\qquad
# \left(\frac{d^2}{dx^2}\right)^{\dagger}=\frac{d^2}{dx^2}\ (\text{對稱})$$
#
# 邊界條件是關鍵：**邊界項必須消失**才算自伴。
# 把算子離散化成矩陣，自伴就變成**矩陣對稱** ── 可以直接驗證。

# %%
section("17.3 離散化：微分算子 → 矩陣，自伴 → 對稱")

n = 60
h = 1.0 / (n + 1)
x_grid = np.linspace(h, 1 - h, n)

# 一階導數的中央差分矩陣（Dirichlet 邊界）
D1 = np.zeros((n, n))
for i in range(n):
    if i > 0:
        D1[i, i - 1] = -1 / (2 * h)
    if i < n - 1:
        D1[i, i + 1] = 1 / (2 * h)
check("d/dx 的矩陣是反對稱 ⇒ (d/dx)† = −d/dx", D1.T, -D1)

# 二階導數
D2 = -second_difference_matrix(n) / h ** 2
check("d²/dx² 的矩陣是對稱 ⇒ 自伴", D2.T, D2)
check("−d²/dx² 正定（Dirichlet 邊界）",
      np.all(np.linalg.eigvalsh(-D2) > 0))

print("離散的分部積分（Dirichlet 邊界下邊界項消失）：")
rng = np.random.default_rng(0)
for _ in range(3):
    f_v = rng.standard_normal(n)
    g_v = rng.standard_normal(n)
    check("  ⟨f|D₁g⟩ = −⟨D₁f|g⟩", f_v @ (D1 @ g_v), -(D1 @ f_v) @ g_v)
    check("  ⟨f|D₂g⟩ = ⟨D₂f|g⟩", f_v @ (D2 @ g_v), (D2 @ f_v) @ g_v)

# 自伴 ⇒ 實特徵值 + 正交特徵函數
lam_d, V_d = np.linalg.eigh(-D2)
print(f"\n−d²/dx² 的前 5 個特徵值 = {lam_d[:5]}")
print(f"理論 (nπ)² = {np.array([(k * np.pi) ** 2 for k in range(1, 6)])}")
check("λₙ ≈ (nπ)²", lam_d[:5], np.array([(k * np.pi) ** 2 for k in range(1, 6)]),
      tol=1.0)
check("特徵值全為實數", np.all(np.isreal(lam_d)))
check("特徵函數正交", V_d.T @ V_d, np.eye(n), tol=1e-10)
for k in [1, 2, 3]:
    v_th = np.sin(k * np.pi * x_grid)
    v_th = v_th / np.linalg.norm(v_th)
    v_num = V_d[:, k - 1]
    if v_num @ v_th < 0:
        v_num = -v_num
    check(f"第 {k} 個特徵函數 ≈ sin({k}πx)", v_num, v_th, tol=1e-3)
print("  ⇒ 「自伴算子的特徵函數正交」就是譜定理（第 7 章）在無窮維的版本")

# 邊界條件改變一切
print("\n邊界條件決定自伴性與特徵值：")
for bc_name, M_bc in [("Dirichlet（兩端固定）", -second_difference_matrix(n) / h ** 2),
                      ("Neumann（兩端自由）", None),
                      ("週期（首尾相接）", None)]:
    if bc_name.startswith("Neumann"):
        M_bc = -second_difference_matrix(n) / h ** 2
        M_bc[0, 0] = -1 / h ** 2
        M_bc[-1, -1] = -1 / h ** 2
        M_bc = -M_bc
        M_bc = second_difference_matrix(n) / h ** 2
        M_bc[0, 0] = 1 / h ** 2
        M_bc[-1, -1] = 1 / h ** 2
    elif bc_name.startswith("週期"):
        M_bc = second_difference_matrix(n) / h ** 2
        M_bc[0, -1] = -1 / h ** 2
        M_bc[-1, 0] = -1 / h ** 2
    lam_bc = np.sort(np.linalg.eigvalsh(M_bc))
    print(f"  {bc_name:<22} 對稱：{np.allclose(M_bc, M_bc.T)}，"
          f"最小 λ = {lam_bc[0]:>10.4f}，λ = 0？{abs(lam_bc[0]) < 1e-8}")
    check(f"  {bc_name}：矩陣對稱（自伴）", M_bc, M_bc.T)
print("  Neumann 與週期邊界有 λ = 0（常數函數）⇒ 算子奇異（對應第 2 章的 B 矩陣）")

# %% [markdown]
# ## 17.4　Sturm–Liouville = 廣義特徵值問題
#
# $$\mathcal Ly=-\frac{d}{dx}\!\left(p(x)\frac{dy}{dx}\right)+q(x)y
# =\lambda\,\rho(x)\,y$$
#
# 離散化後**正好是第 15 章的廣義特徵值問題**：
#
# $$K\mathbf y=\lambda M\mathbf y,\qquad
# K=\text{（對稱剛度矩陣）},\quad M=\operatorname{diag}(\rho)$$
#
# 所有物理上重要的方程都是 Sturm–Liouville 型：
#
# | 方程 | $p$ | $q$ | $\rho$ | 特徵函數 |
# |---|---|---|---|---|
# | $-y''=\lambda y$ | 1 | 0 | 1 | $\sin,\cos$ |
# | Legendre | $1-x^2$ | 0 | 1 | $P_n(x)$ |
# | Chebyshev | $\sqrt{1-x^2}$ | 0 | $1/\sqrt{1-x^2}$ | $T_n(x)$ |
# | Hermite | $e^{-x^2}$ | 0 | $e^{-x^2}$ | $H_n(x)$ |
# | Bessel | $x$ | $-m^2/x$ | $x$ | $J_m(kx)$ |
#
# 結論永遠相同：**$\lambda$ 實數、特徵函數在權重 $\rho$ 下正交、完備**。

# %%
section("17.4 Sturm–Liouville 的離散化")


def sturm_liouville_matrices(p, q, rho, a, b, n):
    """用有限差分把 −(p y')' + q y = λ ρ y 離散成 K y = λ M y。

    用「交錯網格」（staggered grid）保持 K 的對稱性：
    K[i,i±1] = −p(x_{i±1/2})/h²，K[i,i] = (p_{i−1/2}+p_{i+1/2})/h² + q(xᵢ)
    """
    h = (b - a) / (n + 1)
    x = a + h * np.arange(1, n + 1)
    x_half = a + h * (np.arange(0, n + 1) + 0.5)
    p_half = p(x_half)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = (p_half[i] + p_half[i + 1]) / h ** 2 + q(x)[i]
        if i > 0:
            K[i, i - 1] = -p_half[i] / h ** 2
        if i < n - 1:
            K[i, i + 1] = -p_half[i + 1] / h ** 2
    M = np.diag(rho(x))
    return K, M, x


# 例 1：−y'' = λy，y(0)=y(1)=0 → λₙ = (nπ)², yₙ = sin(nπx)
n = 120
K1, M1, x1 = sturm_liouville_matrices(lambda x: np.ones_like(x),
                                      lambda x: np.zeros_like(x),
                                      lambda x: np.ones_like(x), 0.0, 1.0, n)
check("K 對稱", K1, K1.T)
check("M 對稱正定", np.all(np.diag(M1) > 0))
lam1, Y1 = generalized_symmetric_eig(K1, M1)
order = np.argsort(lam1)
lam1, Y1 = lam1[order], Y1[:, order]
print("例 1：−y'' = λy, y(0)=y(1)=0")
print(f"{'n':>3} | {'數值 λ':>14} | {'理論 (nπ)²':>14} | {'相對誤差':>10}")
print("-" * 50)
for k in range(1, 6):
    theory = (k * np.pi) ** 2
    print(f"{k:>3} | {lam1[k - 1]:>14.6f} | {theory:>14.6f} | "
          f"{abs(lam1[k-1]-theory)/theory:>9.2e}")
    check(f"  λ{k} ≈ (={k}π)²", lam1[k - 1], theory, tol=theory * 1e-3)
check("特徵函數在 M-內積下正交（= 權重正交）", Y1.T @ M1 @ Y1, np.eye(n), tol=1e-8)

# 例 2：Legendre 方程 −((1−x²)y')' = λy → λ = n(n+1)
#
# 注意：這是「奇異」的 Sturm–Liouville 問題（端點 x = ±1 處 p = 1−x² 退化為 0）。
# 自然邊界條件是「在端點保持有界」，而不是 y(±1) = 0，所以不能直接套用
# 上面的有限差分（它強制了 Dirichlet 條件）。這裡改用 Galerkin／Ritz 法：
# 在 Legendre 多項式張出的子空間裡組出弱形式的矩陣。
print("\n例 2：Legendre 方程 −((1−x²)y')' = λy（奇異 SL 問題，用 Galerkin 法）")
n_leg = 7
t_leg = GL_NODES
w_leg = GL_WEIGHTS
P_vals = np.array([np.polynomial.legendre.Legendre.basis(k)(t_leg)
                   for k in range(n_leg)])
P_derivs = np.array([np.polynomial.legendre.Legendre.basis(k).deriv()(t_leg)
                     for k in range(n_leg)])
# 弱形式：K_ij = ∫(1−x²)P'ᵢP'ⱼ dx（分部積分後邊界項因 p(±1)=0 而消失），M_ij = ∫PᵢPⱼ dx
K_gal = np.einsum("q,iq,jq->ij", w_leg * (1 - t_leg ** 2), P_derivs, P_derivs)
M_gal = np.einsum("q,iq,jq->ij", w_leg, P_vals, P_vals)
check("Galerkin 的 K 對稱", K_gal, K_gal.T, tol=1e-10)
check("M 對角（Legendre 本身正交）", M_gal, np.diag(np.diag(M_gal)), tol=1e-10)
check("M 的對角線 = 2/(2n+1)", np.diag(M_gal),
      np.array([2 / (2 * k + 1) for k in range(n_leg)]), tol=1e-10)
check("K 也對角（Legendre 就是特徵函數！）", K_gal, np.diag(np.diag(K_gal)),
      tol=1e-10)
lam_gal = np.diag(K_gal) / np.diag(M_gal)
print(f"  λ = K_nn/M_nn = {np.round(lam_gal, 8)}")
print(f"  理論 n(n+1)   = {[k * (k + 1) for k in range(n_leg)]}")
check("λₙ = n(n+1)", lam_gal,
      np.array([float(k * (k + 1)) for k in range(n_leg)]), tol=1e-9)
print("  ⇒ K 與 M 同時對角 ⟺ Legendre 多項式正是這個 SL 問題的特徵函數")

# 例 3：量子諧振子 −y'' + x²y = λ y → λ = 2n+1
K3, M3, x3 = sturm_liouville_matrices(lambda x: np.ones_like(x),
                                      lambda x: x ** 2,
                                      lambda x: np.ones_like(x), -10.0, 10.0, 400)
lam3 = np.sort(generalized_symmetric_eig(K3, M3)[0])
print("\n例 3：量子諧振子 −y'' + x²y = λy（Hermite）")
print(f"  數值 λ 的前 6 個 = {np.round(lam3[:6], 6)}")
print(f"  理論 2n+1（n = 0..5）= {[2 * k + 1 for k in range(6)]}")
for k in range(6):
    check(f"  λ{k} ≈ {2*k+1}（能階量子化！）", lam3[k], 2 * k + 1, tol=0.01)
print("  ⇒ 能階量子化就是「自伴算子的特徵值是離散的」——"
      "量子力學的核心（第 21 章）")

# 基態波函數 = e^{−x²/2}
K3b, M3b, x3b = sturm_liouville_matrices(lambda x: np.ones_like(x),
                                         lambda x: x ** 2,
                                         lambda x: np.ones_like(x), -8.0, 8.0, 400)
lam3b, Y3b = generalized_symmetric_eig(K3b, M3b)
idx0 = np.argmin(lam3b)
psi0 = Y3b[:, idx0]
psi0 = psi0 / np.linalg.norm(psi0) * np.sign(psi0[len(psi0) // 2])
theory0 = np.exp(-x3b ** 2 / 2)
theory0 = theory0 / np.linalg.norm(theory0)
check("基態波函數 ∝ e^{−x²/2}", psi0, theory0, tol=1e-3)

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
for k in range(4):
    v = Y1[:, k] / np.max(np.abs(Y1[:, k]))
    ax.plot(x1, v + 0 * k, lw=1.5, label=f"n = {k+1}, lambda = {lam1[k]:.1f}")
ax.set_title("Eigenfunctions of -y'' on [0,1]", fontsize=9)
ax.set_xlabel("x")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
idx_sorted = np.argsort(lam3b)
for k in range(4):
    v = Y3b[:, idx_sorted[k]]
    v = v / np.max(np.abs(v))
    ax.plot(x3b, v + 2.2 * k, lw=1.4, label=f"n = {k}, E = {lam3b[idx_sorted[k]]:.2f}")
    ax.axhline(2.2 * k, color="k", lw=0.4, alpha=0.3)
ax.plot(x3b, x3b ** 2 / 10, "k--", lw=1, alpha=0.6, label="potential x^2 (scaled)")
ax.set_xlim(-6, 6)
ax.set_title("Quantum harmonic oscillator eigenfunctions", fontsize=9)
ax.set_xlabel("x")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
finish(fig, "ch17_sturm_liouville")

# %% [markdown]
# ## 17.5　Green 函數 = 算子的反矩陣
#
# 要解 $\mathcal Ly=f$，有限維的答案是 $\mathbf y=S^{-1}\mathbf f$；
# 無窮維的答案是
#
# $$y(x)=\int_a^b G(x,z)f(z)\,dz$$
#
# **$G$ 就是 $\mathcal L^{-1}$ 的「矩陣元素」**。用譜分解寫出來：
#
# $$S^{-1}=\sum_n\frac{\mathbf q_n\mathbf q_n^{\mathsf T}}{\lambda_n}
# \qquad\Longleftrightarrow\qquad
# G(x,z)=\sum_n\frac{y_n(x)y_n^{*}(z)}{\lambda_n}$$
#
# 完全是同一個公式。下面用離散化直接驗證 $G\approx h\cdot K^{-1}$。

# %%
section("17.5 Green 函數 = K⁻¹")

n = 80
h = 1.0 / (n + 1)
x_g = np.linspace(h, 1 - h, n)
K = second_difference_matrix(n) / h ** 2        # −d²/dx²，Dirichlet
Kinv = np.linalg.inv(K)

# 理論 Green 函數：−G'' = δ(x−z)，G(0)=G(1)=0 ⇒ G(x,z) = x(1−z) for x<z
G_theory = np.array([[min(xi, zj) * (1 - max(xi, zj)) for zj in x_g] for xi in x_g])
print("理論 G(x,z) = x(1−z) (x<z)，z(1−x) (x>z)")
check("K⁻¹ = h·G（離散 Green 函數）", Kinv, h * G_theory, tol=1e-12)
check("G 對稱（自伴算子的 Green 函數必對稱）", G_theory, G_theory.T)
check("K⁻¹ 對稱", Kinv, Kinv.T)

# 譜分解：G = Σ yₙ(x)yₙ(z)/λₙ
lam_K, Y_K = np.linalg.eigh(K)
G_spectral = sum(np.outer(Y_K[:, k], Y_K[:, k]) / lam_K[k] for k in range(n))
check("K⁻¹ = Σ qₙqₙᵀ/λₙ（譜分解 = Green 函數的級數）", G_spectral, Kinv, tol=1e-10)
print(f"  用前 {5} 個模態近似的相對誤差 = "
      f"{np.linalg.norm(sum(np.outer(Y_K[:, k], Y_K[:, k]) / lam_K[k] for k in range(5)) - Kinv) / np.linalg.norm(Kinv):.3%}")
print("  （低頻模態貢獻最大，因為分母 λₙ 最小 —— 這是低秩近似的另一個例子）")

# 用 Green 函數解方程
f_rhs = np.sin(np.pi * x_g) + 0.5 * np.sin(4 * np.pi * x_g)
y_direct = np.linalg.solve(K, f_rhs)
y_green = h * (G_theory @ f_rhs)                # ∫G(x,z)f(z)dz ≈ h Σⱼ G(x,zⱼ)f(zⱼ)
check("用 Green 函數積分 = 直接解方程（K⁻¹ = h·G）", y_green, y_direct, tol=1e-6)
y_exact = (np.sin(np.pi * x_g) / np.pi ** 2
           + 0.5 * np.sin(4 * np.pi * x_g) / (4 * np.pi) ** 2)
check("與解析解 sin(πx)/π² + 0.5sin(4πx)/(4π)² 相符", y_direct, y_exact, tol=1e-3)

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
im = ax.imshow(G_theory, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
ax.set_title("Green's function G(x,z) = inverse of -d2/dx2", fontsize=9)
ax.set_xlabel("z")
ax.set_ylabel("x")
fig.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(1, 2, 2)
for z0, c in [(0.25, "C0"), (0.5, "C1"), (0.75, "C2")]:
    j = int(np.argmin(np.abs(x_g - z0)))
    ax.plot(x_g, G_theory[:, j], c, lw=1.6, label=f"G(x, z={z0})")
ax.set_title("Response to a point load (tent functions)", fontsize=9)
ax.set_xlabel("x")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
finish(fig, "ch17_green_function")

# %% [markdown]
# ## 17.6　完備性與展開定理
#
# **完備**（completeness）= 特徵函數張出整個空間 = Parseval 成立：
#
# $$\sum_n|c_n|^2=\|f\|^2,\qquad
# \sum_n y_n(x)y_n^{*}(z)=\frac{\delta(x-z)}{\rho(x)}$$
#
# 離散版就是 $QQ^{\mathsf T}=I$（第 4 章）── 用矩陣可以**精確**驗證。

# %%
section("17.6 完備性 = QQᵀ = I")

check("Σ qₙqₙᵀ = I（完備性的離散版 = δ 函數的展開）", Y_K @ Y_K.T, np.eye(n),
      tol=1e-10)
rng = np.random.default_rng(1)
f_any = rng.standard_normal(n)
coeffs = Y_K.T @ f_any
check("Parseval：Σ|cₙ|² = ‖f‖²", float(coeffs @ coeffs), float(f_any @ f_any))
check("展開定理：f = Σ cₙqₙ 完全還原", Y_K @ coeffs, f_any)
print(f"{'保留模態數':>12} | {'還原誤差':>12} | {'能量佔比':>10}")
print("-" * 40)
for k in [1, 5, 20, 50, n]:
    recon = Y_K[:, :k] @ coeffs[:k]
    print(f"{k:>12} | {np.linalg.norm(f_any - recon):>12.6f} | "
          f"{np.sum(coeffs[:k] ** 2) / np.sum(coeffs ** 2):>9.2%}")
    check(f"  k = {k}：Bessel 不等式",
          np.sum(coeffs[:k] ** 2) <= np.sum(coeffs ** 2) + 1e-10)
print("  ⇒ 「無窮維的 Fourier 展開」與「有限維的正交基底展開」是同一件事")

# 不完備的例子
print("\n不完備基底的後果：")
partial = Y_K[:, :n // 2]
P_half = partial @ partial.T
check("不完備時 QQᵀ ≠ I（是投影矩陣）", not np.allclose(P_half, np.eye(n)))
check("  但它仍是投影矩陣（P² = P）", P_half @ P_half, P_half, tol=1e-10)
check("  trace P = 保留的模態數", np.trace(P_half), float(n // 2))
print(f"  trace(P) = {np.trace(P_half):.1f} = {n // 2} 個模態")

# %% [markdown]
# ## 動手練習
#
# 1. 用 Gram–Schmidt 從 $1,x,x^2,x^3$ 在權重 $e^{-x}$、區間 $[0,\infty)$ 上
#    造出前四個 Laguerre 多項式。
# 2. 求 $f(x)=x$ 在 $[-\pi,\pi]$ 的 Fourier 級數，並驗證 Parseval
#    給出 $\sum 1/n^2=\pi^2/6$。
# 3. 把 Bessel 方程 $-(xy')'+\frac{m^2}{x}y=\lambda xy$ 離散化，
#    求 $m=0$ 的前幾個特徵值，與 $J_0$ 的零點平方比較。
# 4. 驗證 $-y''=\lambda y$ 在 Neumann 邊界下有 $\lambda=0$，特徵函數是常數。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1：Laguerre
Q_lag, t_lag, w_lag = gram_schmidt_functions(lambda k: (lambda x: x ** k),
                                             lambda x: np.exp(-x), 0.0, 60.0, 4)
print("練習 1：Laguerre 多項式（權重 e^{−x}）")
for k in range(4):
    L_k = np.polynomial.laguerre.Laguerre.basis(k)(t_lag)
    L_k = L_k / np.sqrt((w_lag * L_k * L_k).sum())
    sign = np.sign((w_lag * Q_lag[k] * L_k).sum())
    check(f"  φ̂{k} = 正規化的 L{k}", sign * Q_lag[k], L_k, tol=1e-6)
print("  （L₀ = 1, L₁ = 1−x, L₂ = 1−2x+x²/2, ...）")

# 練習 2：f(x) = x 的 Fourier 級數
a0_x, an_x, bn_x = fourier_coeffs(lambda x: x, 400)
print(f"\n練習 2：f(x) = x 的 Fourier 係數")
print(f"  a₀ = {a0_x:.2e}，aₙ ≈ 0（奇函數）")
print(f"  b₁..b₅ = {np.round(bn_x[:5], 6)}，理論 bₙ = 2(−1)^{{n+1}}/n")
for k in range(1, 6):
    check(f"  b{k} = 2(−1)^{k+1}/{k}", bn_x[k - 1], 2 * (-1) ** (k + 1) / k, tol=1e-6)
energy_x = norm(lambda x: x, a=-np.pi, b=np.pi) ** 2
sum_bn2 = np.pi * np.sum(bn_x ** 2)
print(f"  ‖f‖² = 2π³/3 = {2 * np.pi ** 3 / 3:.6f}，Σ 能量 = {sum_bn2:.6f}")
print(f"  400 項已涵蓋 {sum_bn2 / energy_x:.4%} 的能量"
      f"（尾項 ~ Σ4/n² ≈ {4 / 400:.4f}）")
check("  Parseval（400 項已達 99.8%）", sum_bn2 / energy_x > 0.998)
# Parseval ⇒ Σ 4/n² = 2π²/3 ⇒ Σ1/n² = π²/6
partial_zeta = np.sum([1 / k ** 2 for k in range(1, 100001)])
print(f"  Parseval 給出 Σ1/n² = π²/6 = {np.pi ** 2 / 6:.8f}")
print(f"  直接求和（10⁵ 項）= {partial_zeta:.8f}")
check("  Σ1/n² = π²/6", partial_zeta, np.pi ** 2 / 6, tol=1e-4)

# 練習 3：Bessel（x = 0 是奇異端點，自然邊界條件是「有界」而非 y(0) = 0，
#          所以改成直接驗證解析解 J₀(αₖx) 的 Sturm–Liouville 性質）
print("\n練習 3：Bessel 方程 −(xy')' = λxy，y(1) = 0，在 x=0 有界")
try:
    from scipy.special import j0, j1, jn_zeros
    alphas = jn_zeros(0, 5)
    print(f"  J₀ 的前 5 個零點 α = {np.round(alphas, 6)}")
    print(f"  對應的 λ = α² = {np.round(alphas ** 2, 4)}")
    # (i) 滿足邊界條件
    for k, al in enumerate(alphas[:3]):
        check(f"  J₀(α{k+1}·1) = 0（邊界條件）", j0(al), 0.0, tol=1e-10)
    # (ii) 滿足微分方程：−(x y')' = λ x y，其中 y = J₀(αx)
    xs_b = np.linspace(0.05, 0.95, 400)
    for k, al in enumerate(alphas[:3]):
        y = lambda t: j0(al * t)
        hb = 1e-5
        # (x y')' 用中央差分
        flux = lambda t: t * (y(t + hb) - y(t - hb)) / (2 * hb)
        lhs = -(flux(xs_b + hb) - flux(xs_b - hb)) / (2 * hb)
        rhs = al ** 2 * xs_b * y(xs_b)
        check(f"  J₀(α{k+1}x) 滿足 −(xy')' = α²xy",
              np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)) < 1e-4)
    # (iii) 在權重 ρ = x 下正交（SL 理論的核心結論）
    print("  權重正交性 ∫₀¹ J₀(αⱼx)J₀(αₖx)·x dx：")
    for jj in range(3):
        for kk in range(jj, 3):
            val = inner(lambda t: j0(alphas[jj] * t), lambda t: j0(alphas[kk] * t),
                        rho=lambda t: t, a=0.0, b=1.0)
            theory = 0.0 if jj != kk else 0.5 * j1(alphas[jj]) ** 2
            print(f"    j={jj+1}, k={kk+1}：{val:>12.8f}（理論 {theory:.8f}）")
            check(f"    j={jj+1}, k={kk+1}", float(np.real(val)), theory, tol=1e-9)
    print("  ⇒ 不同特徵值的特徵函數在權重 ρ = x 下正交，正是 SL 理論的保證")
except ImportError:
    print("  （未安裝 scipy，略過 Bessel 的驗證）")

# 練習 4：Neumann 邊界
print("\n練習 4：Neumann 邊界")
n4 = 60
h4 = 1.0 / n4
K4 = second_difference_matrix(n4) / h4 ** 2
K4[0, 0] = 1 / h4 ** 2
K4[-1, -1] = 1 / h4 ** 2
lam4 = np.sort(np.linalg.eigvalsh(K4))
print(f"  最小特徵值 = {lam4[0]:.3e}（理論 0）")
check("  λ = 0 存在（Neumann）", abs(lam4[0]) < 1e-8)
v0 = np.linalg.eigh(K4)[1][:, np.argmin(np.abs(lam4))]
check("  對應的特徵函數是常數", np.std(v0 / v0[0]) < 1e-8)
print(f"  次小特徵值 = {lam4[1]:.4f}（理論 π² = {np.pi ** 2:.4f}）")
check("  λ₂ ≈ π²", lam4[1], np.pi ** 2, tol=0.3)
print("  ⇒ Neumann 邊界讓算子奇異（常數函數在零空間）——"
      "正是第 2 章「自由–自由」矩陣的連續版")

# %% [markdown]
# ## 本章重點回顧
#
# * 函數空間 + 內積 $\langle f|g\rangle=\int f^{*}g\rho\,dx$ = **Hilbert 空間**；
#   Schwarz、三角、Bessel 三個不等式一字不改。
# * **Gram–Schmidt 作用在單項式上就生出正交多項式家族**：
#   權重 1 → Legendre、$1/\sqrt{1-x^2}$ → Chebyshev、$e^{-x^2}$ → Hermite、
#   $e^{-x}$ → Laguerre。
# * **Fourier 係數 = 內積**，截斷 = 投影到有限維子空間 = 最小平方最佳近似。
#   Bessel 不等式 ⇒ 誤差² = 被丟掉的能量（Eckart–Young 的無窮維版本）。
#   不連續點造成 **Gibbs 過衝 1.179**，不隨 $N$ 消失。
# * 伴隨算子由內積定義；分部積分給出
#   $(d/dx)^{\dagger}=-d/dx$、$(d^2/dx^2)^{\dagger}=d^2/dx^2$。
#   離散化後「自伴」就是「**矩陣對稱**」，可以直接驗證。
# * **Sturm–Liouville 方程離散化就是第 15 章的廣義特徵值問題** $K\mathbf y=\lambda M\mathbf y$。
#   所有物理重要方程（Legendre、Chebyshev、Hermite、Bessel）都是這一型；
#   結論永遠是「$\lambda$ 實數、特徵函數權重正交、完備」。
#   **量子化的能階就是離散特徵值**。
# * **Green 函數 = 算子的反矩陣**：
#   $G(x,z)=\sum_n y_n(x)y_n(z)/\lambda_n$ 就是 $S^{-1}=\sum\mathbf q\mathbf q^{\mathsf T}/\lambda$。
# * 完備性 = $QQ^{\mathsf T}=I$ = Parseval = $\delta$ 函數的展開。
#
# 下一章：線性 ODE 系統的線性代數結構 ── Wronskian、解空間與級數解。
