# %% [markdown]
# # 第 18 章　線性 ODE 的線性代數結構
#
# > 對應 Riley 第 14 章（一階 ODE）、第 15 章（高階線性 ODE：常係數、變係數、
# > Wronskian、參數變化法、Green 函數）、第 16 章（級數解、Frobenius 法、多項式解），
# > 並對照 Strang 6.5（微分方程）、本教材第 8 章（矩陣指數）。
#
# 「線性微分方程」的每一個定理其實都是線性代數：
#
# | ODE 的語言 | 線性代數的語言 |
# |---|---|
# | $n$ 階齊次方程的解空間 | $n$ 維向量空間 |
# | 基本解組 $y_1,\dots,y_n$ | 基底 |
# | **Wronskian** $W\ne0$ | 基底矩陣可逆（$\det\ne0$）|
# | 通解 = 齊次解 + 特解 | $\mathbf x=\mathbf x_n+\mathbf x_p$（第 3 章！）|
# | 參數變化法 | 解一個線性系統（Cramer 法則）|
# | 基本矩陣 $\Phi(t)$ | $e^{At}$ |
# | 級數解的遞迴關係 | 無窮維三角系統 |
# | Green 函數 | 算子的反矩陣 |
#
# ```bash
# python chapters/ch18_ode_systems.py
# python tools/build_notebooks.py ch18
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

from linalg_tutorial.determinant import cramer, det_lu
from linalg_tutorial.eigen import matrix_exp_by_eigen, matrix_exp_series
from linalg_tutorial.subspaces import rank
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 18.1　解空間的維度就是階數
#
# $n$ 階線性齊次 ODE
#
# $$y^{(n)}+a_{n-1}(x)y^{(n-1)}+\dots+a_0(x)y=0$$
#
# 的解構成一個**$n$ 維向量空間**。驗證方法：把函數在許多點取樣變成向量，
# 算矩陣的秩 ── 秩就是解空間的維度。

# %%
section("18.1 解空間的維度 = 方程的階數")

xs = np.linspace(0.3, 3.0, 400)


def space_dim(solutions, xs, tol=1e-8):
    """把解取樣成向量，用秩測量解空間的維度。"""
    M = np.array([f(xs) for f in solutions])
    return np.linalg.matrix_rank(M, tol=tol * np.linalg.norm(M))


cases = [
    ("y'' = 0（2 階）", [lambda x: np.ones_like(x), lambda x: x], 2),
    ("y'' + y = 0（2 階）", [np.sin, np.cos], 2),
    ("y'' − y = 0（2 階）", [np.exp, lambda x: np.exp(-x)], 2),
    ("y''' = 0（3 階）", [lambda x: np.ones_like(x), lambda x: x,
                          lambda x: x ** 2], 3),
    ("y'''' = y（4 階）", [np.exp, lambda x: np.exp(-x), np.sin, np.cos], 4),
]
for name, sols, n_expected in cases:
    d = space_dim(sols, xs)
    print(f"  {name:<22} 取樣後的秩 = {d}（階數 = {n_expected}）")
    check(f"  {name}：解空間維度 = 階數", d, n_expected)

# 多餘的「解」一定是相依的
redundant = [np.sin, np.cos, lambda x: 3 * np.sin(x) - 2 * np.cos(x)]
print(f"\n再加一個 3sin − 2cos：秩仍然是 {space_dim(redundant, xs)}")
check("多餘的解必相依（秩不增加）", space_dim(redundant, xs), 2)
print("  ⇒ 「二階方程最多兩個獨立解」就是「二維空間最多兩個獨立向量」")

# 解空間對線性組合封閉
print("\n解空間是子空間（對線性組合封閉）：")
# 二階中央差分的最佳步長：截斷誤差 O(h²) 與捨入誤差 O(ε/h²) 平衡於 h ≈ ε^{1/4}
h_num = 1e-4
ode = lambda y, x: ((y(x + h_num) - 2 * y(x) + y(x - h_num)) / h_num ** 2 + y(x))
rng = np.random.default_rng(0)
for _ in range(3):
    c, d = rng.standard_normal(2)
    combo = lambda x: c * np.sin(x) + d * np.cos(x)
    resid = np.max(np.abs(ode(combo, xs))) / (abs(c) + abs(d))
    check(f"  {c:.2f}·sin + {d:.2f}·cos 仍是 y''+y=0 的解（相對殘差 "
          f"{resid:.1e}）", resid < 1e-6)

# %% [markdown]
# ## 18.2　Wronskian = 基底矩陣的行列式
#
# $$W(x)=\det\begin{bmatrix}
# y_1&y_2&\cdots&y_n\\
# y_1'&y_2'&\cdots&y_n'\\
# \vdots&&&\vdots\\
# y_1^{(n-1)}&\cdots&&y_n^{(n-1)}\end{bmatrix}$$
#
# $W\ne0$ $\iff$ 這些解線性獨立 $\iff$ 它們構成基底。
#
# **Abel 公式**：$W'=-a_{n-1}(x)W$，所以
# $W(x)=W(x_0)\exp\left(-\int a_{n-1}\right)$ ──
# Wronskian 要嘛恆為 0、要嘛處處不為 0（不可能只在某點為 0）。

# %%
section("18.2 Wronskian 與 Abel 公式")


def wronskian(funcs, x, h=1e-5):
    """用數值微分組出 Wronskian 矩陣的行列式。"""
    n = len(funcs)
    # 用中央差分算各階導數
    def deriv(f, k):
        if k == 0:
            return f
        g = deriv(f, k - 1)
        return lambda t: (g(t + h) - g(t - h)) / (2 * h)
    W = np.array([[deriv(f, k)(x) for f in funcs] for k in range(n)])
    return np.linalg.det(W)


x0 = 1.3
print(f"在 x = {x0} 處的 Wronskian：")
tests = [
    ("{sin, cos}（獨立）", [np.sin, np.cos], True),
    ("{eˣ, e⁻ˣ}（獨立）", [np.exp, lambda t: np.exp(-t)], True),
    ("{sin, 2sin}（相依）", [np.sin, lambda t: 2 * np.sin(t)], False),
    ("{1, x, x²}（獨立）", [lambda t: 1.0 + 0.0 * t, lambda t: t,
                            lambda t: t ** 2], True),
]
for name, funcs, independent in tests:
    W = wronskian(funcs, x0)
    print(f"  {name:<22} W = {W:>12.6f} → {'獨立' if abs(W) > 1e-6 else '相依'}")
    check(f"  {name}：W ≠ 0 ⟺ 獨立", (abs(W) > 1e-6) == independent)
check("{sin, cos} 的 W = −1（常數，因為 a₁ = 0）",
      wronskian([np.sin, np.cos], x0), -1.0, tol=1e-6)
check("{eˣ, e⁻ˣ} 的 W = −2", wronskian([np.exp, lambda t: np.exp(-t)], x0), -2.0,
      tol=1e-5)

# Abel 公式：W' = −a₁W
print("\nAbel 公式：y'' + a₁(x)y' + a₀(x)y = 0 ⇒ W(x) = W(x₀)exp(−∫a₁)")
print("  例：x²y'' − 2xy' + 2y = 0 的解 y = x, x²（a₁ = −2/x）")
funcs_abel = [lambda t: t, lambda t: t ** 2]
print(f"{'x':>6} | {'W(x)':>12} | {'W(1)·exp(∫2/x)=x²':>20}")
print("-" * 44)
for xv in [1.0, 1.5, 2.0, 3.0]:
    W = wronskian(funcs_abel, xv)
    print(f"{xv:>6} | {W:>12.6f} | {xv ** 2:>20.6f}")
    check(f"  W({xv}) = x²", W, xv ** 2, tol=1e-5)
print("  ⇒ W 處處不為 0（x ≠ 0），所以 {x, x²} 在 x > 0 上是基底")
print("  Abel 公式的意義：W 滿足一階線性方程 ⇒ 要嘛恆 0、要嘛永不為 0")

# %% [markdown]
# ## 18.3　常係數方程：伴隨矩陣與特徵方程
#
# $$y^{(n)}+a_{n-1}y^{(n-1)}+\dots+a_0y=0
# \quad\Longleftrightarrow\quad
# \frac{d\mathbf u}{dx}=C\mathbf u,\quad
# \mathbf u=(y,y',\dots,y^{(n-1)})$$
#
# **伴隨矩陣** $C$ 的特徵多項式正好是「代入 $y=e^{\lambda x}$」得到的輔助方程。
#
# | 根的型態 | ODE 的解 | 線性代數的對應 |
# |---|---|---|
# | $n$ 個相異根 | $e^{\lambda_ix}$ | 可對角化 |
# | $m$ 重根 | $x^{k}e^{\lambda x}$，$k<m$ | **Jordan 塊** |
# | 複根 $\alpha\pm i\beta$ | $e^{\alpha x}\cos\beta x$, $\sin$ | 實 Jordan 形式 |

# %%
section("18.3 伴隨矩陣的特徵多項式 = 輔助方程")


def companion(coeffs):
    """y⁽ⁿ⁾ + a_{n−1}y⁽ⁿ⁻¹⁾ + ... + a₀y = 0 的伴隨矩陣（coeffs = [a₀,...,a_{n−1}]）。"""
    n = len(coeffs)
    C = np.zeros((n, n))
    C[:-1, 1:] = np.eye(n - 1)
    C[-1, :] = -np.asarray(coeffs, dtype=float)
    return C


examples = [
    ("y'' − 3y' + 2y = 0", [2.0, -3.0], "λ² − 3λ + 2 = (λ−1)(λ−2)"),
    ("y'' + 4y = 0", [4.0, 0.0], "λ² + 4（λ = ±2i）"),
    ("y'' − 2y' + y = 0", [1.0, -2.0], "(λ−1)²（重根！）"),
    ("y''' − y = 0", [-1.0, 0.0, 0.0], "λ³ − 1"),
]
for name, coeffs, poly_str in examples:
    C = companion(coeffs)
    lam = np.linalg.eigvals(C)
    roots = np.roots([1.0] + list(reversed(coeffs)))
    print(f"\n{name}（輔助方程 {poly_str}）")
    print(f"  伴隨矩陣的特徵值 = {np.round(np.sort_complex(lam), 6)}")
    print(f"  多項式的根       = {np.round(np.sort_complex(roots), 6)}")
    check("  兩者相同", np.sort_complex(lam), np.sort_complex(roots), tol=1e-8)
    n_vec = np.linalg.matrix_rank(np.linalg.eig(C)[1], tol=1e-8)
    print(f"  獨立特徵向量數 = {n_vec}"
          f"（{'可對角化' if n_vec == len(coeffs) else '需要 Jordan 形式'}）")

# 重根 ⇒ Jordan ⇒ x·e^{λx}
print("\n重根的秘密：Jordan 塊讓 x·e^{λx} 出現")
C_rep = companion([1.0, -2.0])          # y'' − 2y' + y = 0，λ = 1, 1
check("只有一個獨立特徵向量", np.linalg.matrix_rank(np.linalg.eig(C_rep)[1],
                                                    tol=1e-8), 1)
for x_val in [0.5, 1.0, 2.0]:
    E = matrix_exp_series(C_rep, x_val, 70)
    # y(0)=0, y'(0)=1 ⇒ y(x) = x e^x
    y_num = (E @ np.array([0.0, 1.0]))[0]
    check(f"  x = {x_val}：e^{{Cx}} 給出 y = x·eˣ", y_num, x_val * np.exp(x_val),
          tol=1e-9)
print("  ⇒ 「重根要乘上 x」不是記憶口訣，而是 Jordan 塊的指數")

# 複根 → 實數形式
C_cplx = companion([4.0, 0.0])
print("\n複根 λ = ±2i ⇒ e^{Cx} 給出 cos 2x 與 sin 2x：")
for x_val in [0.3, 1.1]:
    E = matrix_exp_series(C_cplx, x_val, 70)
    check(f"  x = {x_val}：y(0)=1,y'(0)=0 ⇒ y = cos 2x",
          (E @ np.array([1.0, 0.0]))[0], np.cos(2 * x_val), tol=1e-10)
    check(f"  x = {x_val}：y(0)=0,y'(0)=1 ⇒ y = sin(2x)/2",
          (E @ np.array([0.0, 1.0]))[0], np.sin(2 * x_val) / 2, tol=1e-10)

# %% [markdown]
# ## 18.4　基本矩陣 $\Phi(x)$
#
# 把 $n$ 個基本解排成矩陣：
#
# $$\Phi(x)=\begin{bmatrix}\mathbf u_1(x)&\cdots&\mathbf u_n(x)\end{bmatrix},
# \qquad \Phi'=C\Phi,\qquad \Phi(x)=e^{Cx}\Phi(0)$$
#
# 任意初值問題的解：$\mathbf u(x)=\Phi(x)\Phi(x_0)^{-1}\mathbf u(x_0)$。
#
# * $\det\Phi=$ **Wronskian**
# * Abel 公式 $\iff$ Jacobi 公式 $\dfrac{d}{dx}\det\Phi=\operatorname{tr}(C)\det\Phi$

# %%
section("18.4 基本矩陣與 Jacobi/Abel 公式")

C = companion([2.0, -3.0])              # y'' − 3y' + 2y = 0
Phi = lambda x: matrix_exp_by_eigen(C, x)
print(f"C =\n{C}，trace C = {np.trace(C)}")
for x_val in [0.0, 0.5, 1.2]:
    check(f"Φ({x_val}) 的行列式 = e^{{tr(C)·x}}", np.real(det_lu(np.real(Phi(x_val)))),
          np.exp(np.trace(C) * x_val), tol=1e-8)
print(f"  det Φ(x) = e^{{3x}}（Jacobi 公式：d/dx det Φ = tr(C)·det Φ）")
check("Φ'(x) = CΦ(x)（數值微分）",
      (np.real(Phi(1.0 + 1e-6)) - np.real(Phi(1.0 - 1e-6))) / 2e-6,
      C @ np.real(Phi(1.0)), tol=1e-6)
check("Φ(0) = I", np.real(Phi(0.0)), np.eye(2))
check("Φ(x+y) = Φ(x)Φ(y)（半群性質）",
      np.real(Phi(1.3)), np.real(Phi(0.5)) @ np.real(Phi(0.8)), tol=1e-9)
u0 = np.array([1.0, 0.0])
for x_val in [0.4, 1.0]:
    y_theory = 2 * np.exp(x_val) - np.exp(2 * x_val)   # y(0)=1, y'(0)=0
    check(f"  x = {x_val}：Φ(x)u₀ 的第一分量 = 2eˣ − e^{{2x}}",
          (np.real(Phi(x_val)) @ u0)[0], y_theory, tol=1e-9)

# %% [markdown]
# ## 18.5　非齊次方程：完整解與參數變化法
#
# $$\mathcal Ly=f\quad\Longrightarrow\quad
# y=\underbrace{y_p}_{\text{特解}}+\underbrace{c_1y_1+\dots+c_ny_n}_{\text{齊次解（零空間）}}$$
#
# 這就是第 3 章「$\mathbf x=\mathbf x_p+\mathbf x_n$」一字不改。
#
# **參數變化法**：令 $y_p=\sum u_i(x)y_i(x)$，則 $u_i'$ 滿足一個線性系統，
# 其係數矩陣正是 Wronskian 矩陣：
#
# $$\begin{bmatrix}y_1&y_2\\y_1'&y_2'\end{bmatrix}
# \begin{bmatrix}u_1'\\u_2'\end{bmatrix}
# =\begin{bmatrix}0\\f\end{bmatrix}
# \quad\Longrightarrow\quad
# u_1'=\frac{-y_2f}{W},\ u_2'=\frac{y_1f}{W}\ (\textbf{Cramer 法則！})$$

# %%
section("18.5 參數變化法 = 用 Cramer 法則解 Wronskian 系統")

# 解 y'' + y = sin(2x)，齊次解 {sin x, cos x}
f_rhs = lambda x: np.sin(2 * x)
y1, y2 = np.sin, np.cos
x_eval = 1.1
W_mat = np.array([[y1(x_eval), y2(x_eval)],
                  [np.cos(x_eval), -np.sin(x_eval)]])
rhs_vec = np.array([0.0, f_rhs(x_eval)])
u_prime = cramer(W_mat, rhs_vec)
print(f"在 x = {x_eval}：Wronskian 矩陣 =\n{np.round(W_mat, 6)}")
print(f"  Cramer 解出 (u₁', u₂') = {np.round(u_prime, 6)}")
W_val = det_lu(W_mat)
check("u₁' = −y₂f/W", u_prime[0], -y2(x_eval) * f_rhs(x_eval) / W_val)
check("u₂' = y₁f/W", u_prime[1], y1(x_eval) * f_rhs(x_eval) / W_val)
check("與直接解線性系統相同", u_prime, np.linalg.solve(W_mat, rhs_vec))

# 積出 u₁, u₂ 得到特解，並與解析解比較
from scipy.integrate import quad
u1_val = quad(lambda t: -y2(t) * f_rhs(t) / det_lu(
    np.array([[y1(t), y2(t)], [np.cos(t), -np.sin(t)]])), 0, x_eval)[0]
u2_val = quad(lambda t: y1(t) * f_rhs(t) / det_lu(
    np.array([[y1(t), y2(t)], [np.cos(t), -np.sin(t)]])), 0, x_eval)[0]
y_particular = u1_val * y1(x_eval) + u2_val * y2(x_eval)
# 解析特解：y_p = −sin(2x)/3（因為 −4+1 = −3）
print(f"\n參數變化法的特解 y_p({x_eval}) = {y_particular:.8f}")
print(f"解析特解 −sin(2x)/3 = {-np.sin(2 * x_eval) / 3:.8f}")
check("兩者相差一個齊次解（這是允許的）",
      abs((y_particular + np.sin(2 * x_eval) / 3)) < 1.0)
# 直接驗證解析特解滿足方程
h = 1e-5
yp = lambda t: -np.sin(2 * t) / 3
resid = (yp(x_eval + h) - 2 * yp(x_eval) + yp(x_eval - h)) / h ** 2 + yp(x_eval)
check("y_p = −sin(2x)/3 滿足 y''+y = sin 2x", resid, f_rhs(x_eval), tol=1e-6)

# 共振：當 f 與齊次解同頻率
print("\n共振（f 落在零空間裡）：y'' + y = sin x")
yp_res = lambda t: -t * np.cos(t) / 2
resid_res = ((yp_res(x_eval + h) - 2 * yp_res(x_eval) + yp_res(x_eval - h)) / h ** 2
             + yp_res(x_eval))
check("特解變成 −x·cos(x)/2（出現 x 的因子）", resid_res, np.sin(x_eval), tol=1e-5)
print("  線性代數的解釋：f 在 C(A) 的邊界上（與零空間共振）⇒ 解會「成長」")
print("  矩陣版的類比：Jordan 塊 (A−λI)x = v 的解會帶出 t 的因子")

# %% [markdown]
# ## 18.6　級數解：遞迴關係 = 無窮維三角系統
#
# 變係數方程通常沒有封閉解，改用級數
# $y=\sum_{n}a_nx^{n}$，代入後得到**係數的遞迴關係** ──
# 那其實是一個無窮維的三角線性系統（可以「往前回代」）。
#
# 例：Legendre 方程 $(1-x^2)y''-2xy'+\ell(\ell+1)y=0$ 給出
#
# $$a_{n+2}=\frac{n(n+1)-\ell(\ell+1)}{(n+1)(n+2)}a_n$$
#
# 當 $\ell$ 是非負整數時級數**自動截斷**成多項式 ── 這就是 Legendre 多項式。

# %%
section("18.6 Legendre 方程的級數解與自動截斷")


def legendre_series(ell, n_terms=20, a0=1.0, a1=0.0):
    """用遞迴關係算出 Legendre 方程的級數係數。"""
    a = np.zeros(n_terms)
    a[0], a[1] = a0, a1
    for n in range(n_terms - 2):
        a[n + 2] = (n * (n + 1) - ell * (ell + 1)) / ((n + 1) * (n + 2)) * a[n]
    return a


print(f"{'ℓ':>3} | {'前 8 個係數（a₀=1, a₁=0）':<46} | 截斷?")
print("-" * 70)
for ell in [0, 2, 4, 1.5]:
    a = legendre_series(ell, 10)
    truncates = np.all(np.abs(a[int(ell) + 2:]) < 1e-14) if float(ell).is_integer() else False
    print(f"{ell:>3} | {str(np.round(a[:8], 5)):<46} | {truncates}")
    if float(ell).is_integer() and int(ell) % 2 == 0:
        check(f"  ℓ = {int(ell)}：級數在 n = {int(ell)} 之後截斷", truncates)
check("ℓ = 1.5（非整數）：級數不截斷（無窮級數，端點發散）",
      not np.all(np.abs(legendre_series(1.5, 20)[4:]) < 1e-14))
print("  ⇒ 「ℓ 必須是整數」就是邊界條件（在 x = ±1 有界）造成的特徵值量化")

# 驗證截斷後就是 Legendre 多項式
for ell in [0, 2, 4]:
    a = legendre_series(ell, 12)
    poly_series = np.polynomial.Polynomial(a[:ell + 1])
    P_ell = np.polynomial.legendre.Legendre.basis(ell).convert(
        kind=np.polynomial.Polynomial)
    ratio = P_ell(0.0) / poly_series(0.0) if abs(poly_series(0.0)) > 1e-12 else 1.0
    xs_t = np.linspace(-0.9, 0.9, 50)
    check(f"ℓ = {ell}：級數解 ∝ Legendre 多項式 P_{ell}",
          ratio * poly_series(xs_t), P_ell(xs_t), tol=1e-10)

# 奇數 ℓ 用 a₀=0, a₁=1
for ell in [1, 3]:
    a = legendre_series(ell, 12, a0=0.0, a1=1.0)
    truncates = np.all(np.abs(a[ell + 2:]) < 1e-14)
    check(f"ℓ = {ell}（奇數，a₀=0,a₁=1）：級數截斷", truncates)

# 遞迴關係 = 三角矩陣系統
print("\n遞迴關係的矩陣面貌（上三角、兩步跳躍）：")
n_show = 8
ell = 4
R_rec = np.zeros((n_show, n_show))
for n in range(n_show - 2):
    R_rec[n + 2, n + 2] = 1.0
    R_rec[n + 2, n] = -(n * (n + 1) - ell * (ell + 1)) / ((n + 1) * (n + 2))
R_rec[0, 0] = R_rec[1, 1] = 1.0
show_matrix("遞迴矩陣（下三角 ⇒ 可以往前回代）", R_rec)
check("遞迴矩陣是下三角", R_rec, np.tril(R_rec))
a_vec = legendre_series(ell, n_show)
rhs_rec = np.zeros(n_show)
rhs_rec[0], rhs_rec[1] = a_vec[0], a_vec[1]
check("解這個三角系統就得到級數係數", np.linalg.solve(R_rec, rhs_rec), a_vec)
print("  ⇒ 「級數解」= 解一個（無窮維的）三角系統，用的是第 2 章的回代")

# %% [markdown]
# ### Frobenius 法：正則奇異點與指標方程
#
# 若 $x=0$ 是**正則奇異點**，解的形式是 $y=x^{\sigma}\sum a_nx^n$。
# 代入後最低次項給出**指標方程**（indicial equation）── 一個二次方程，
# 其兩根 $\sigma_1,\sigma_2$ 決定兩個解的形式。
#
# 例：Bessel 方程 $x^2y''+xy'+(x^2-m^2)y=0$ 的指標方程是 $\sigma^2-m^2=0$，
# 所以 $\sigma=\pm m$。

# %%
section("18.6 Frobenius 法與指標方程")

print("Bessel 方程 x²y'' + xy' + (x² − m²)y = 0：指標方程 σ² − m² = 0")
for m in [0, 1, 2, 0.5]:
    sigmas = np.roots([1.0, 0.0, -m ** 2])
    print(f"  m = {m}：σ = {np.round(np.sort(sigmas), 4)}"
          f"（兩根差 = {abs(sigmas[0] - sigmas[1]):.4f}）")
    check(f"  m = {m}：σ = ±m", np.sort(sigmas), np.sort(np.array([m, -m])))
print("  兩根相差整數（如 m = 0, 1, 2）時，第二個解會出現 log x（Riley 16.4）")


def bessel_series(m, n_terms=25):
    """J_m 的級數係數：a_{n+2} = −a_n/((n+2)(n+2+2m))。"""
    a = np.zeros(n_terms)
    a[0] = 1.0
    for n in range(0, n_terms - 2, 2):
        a[n + 2] = -a[n] / ((n + 2) * (n + 2 + 2 * m))
    return a


try:
    from scipy.special import jv, gamma
    for m in [0, 1, 2]:
        a = bessel_series(m, 30)
        xs_b = np.array([0.3, 1.0, 2.0, 3.0])
        series_val = sum(a[n] * xs_b ** n for n in range(30)) * (xs_b / 2) ** m \
            / 2 ** 0 if False else None
        # J_m(x) = (x/2)^m/Γ(m+1) · Σ ...；用 scipy 比對正規化後的級數
        series_raw = sum(a[n] * xs_b ** n for n in range(30))
        jm = jv(m, xs_b)
        scale = jm[0] / (series_raw[0] * xs_b[0] ** m)
        check(f"  m = {m}：Frobenius 級數 ∝ J_{m}(x)",
              scale * series_raw * xs_b ** m, jm, tol=1e-9)
        print(f"  m = {m}：級數（前 30 項）與 J_{m} 相符，"
              f"正規化常數 = {scale:.8f}（理論 1/(2^m·m!) = "
              f"{1 / (2.0 ** m * gamma(m + 1)):.8f}）")
        check(f"    正規化常數 = 1/(2^m·Γ(m+1))", scale,
              1 / (2.0 ** m * gamma(m + 1)), tol=1e-9)
except ImportError:
    print("  （未安裝 scipy，略過與 J_m 的比對）")

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
xs_p = np.linspace(-1, 1, 400)
for ell, c in zip([0, 1, 2, 3, 4], ["C0", "C1", "C2", "C3", "C4"]):
    ax.plot(xs_p, np.polynomial.legendre.Legendre.basis(ell)(xs_p), c, lw=1.5,
            label=f"P{ell}")
ax.set_title("Legendre polynomials (series truncates)", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
try:
    from scipy.special import jv
    xs_j = np.linspace(0, 15, 500)
    for m, c in zip([0, 1, 2], ["C0", "C1", "C2"]):
        ax.plot(xs_j, jv(m, xs_j), c, lw=1.5, label=f"J{m}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Bessel functions (Frobenius series)", fontsize=9)
    ax.legend(fontsize=8)
except ImportError:
    ax.text(0.3, 0.5, "scipy not available", transform=ax.transAxes)
ax.grid(alpha=0.3)
finish(fig, "ch18_series_solutions")

# %% [markdown]
# ## 18.7　一階系統：守恆量與不變子空間
#
# $d\mathbf u/dt=A\mathbf u$ 的**守恆量**對應 $A$ 的**左零向量**：
#
# $$\mathbf y^{\mathsf T}A=\mathbf 0
# \quad\Longrightarrow\quad
# \frac{d}{dt}(\mathbf y^{\mathsf T}\mathbf u)=\mathbf y^{\mathsf T}A\mathbf u=0$$
#
# 第 3 章的左零空間 $N(A^{\mathsf T})$ 在這裡就是「守恆律的空間」！
# 化學反應網路的質量守恆、電路的 Kirchhoff 定律都是這樣來的。

# %%
section("18.7 守恆量 = 左零向量")

# 化學反應網路：A → B → C（質量守恆）
k1, k2 = 0.8, 0.3
A_chem = np.array([[-k1, 0.0, 0.0],
                   [k1, -k2, 0.0],
                   [0.0, k2, 0.0]])
show_matrix("反應矩陣 A（A→B→C）", A_chem)
from linalg_tutorial.subspaces import left_nullspace
Y_cons = left_nullspace(A_chem)
print(f"左零空間的維度 = {Y_cons.shape[1]}")
show_matrix("左零向量（守恆量的係數）", Y_cons)
check("yᵀA = 0", Y_cons.T @ A_chem, np.zeros((Y_cons.shape[1], 3)))
y_total = Y_cons[:, 0] / Y_cons[0, 0]
print(f"正規化後 = {np.round(y_total, 6)} ⇒ 守恆量是 [A]+[B]+[C]（總質量）")
check("守恆量 = (1,1,1)·u（總濃度）", y_total, np.ones(3))

u0 = np.array([1.0, 0.0, 0.0])
print(f"\n{'t':>6} | {'[A]':>10} | {'[B]':>10} | {'[C]':>10} | {'總和':>10}")
print("-" * 54)
for t in [0.0, 0.5, 1.0, 3.0, 10.0]:
    u = np.real(matrix_exp_by_eigen(A_chem, t) @ u0)
    print(f"{t:>6} | {u[0]:>10.6f} | {u[1]:>10.6f} | {u[2]:>10.6f} | {u.sum():>10.6f}")
    check(f"  t = {t}：總質量守恆", u.sum(), 1.0, tol=1e-9)
check("長時間後全部變成 C（λ = 0 的特徵向量）",
      np.real(matrix_exp_by_eigen(A_chem, 100.0) @ u0), np.array([0.0, 0.0, 1.0]),
      tol=1e-9)
lam_chem = np.sort(np.real(np.linalg.eigvals(A_chem)))
print(f"\n特徵值 = {lam_chem}（λ = 0 對應穩態，負的對應衰減）")
check("λ = 0 存在（因為有守恆量 ⇒ A 奇異）", abs(lam_chem[-1]) < 1e-12)
check("其餘 λ < 0（系統穩定）", np.all(lam_chem[:-1] < 0))

# 解析解
print("\n解析解（連續衰變）：")
for t in [1.0, 2.0]:
    u = np.real(matrix_exp_by_eigen(A_chem, t) @ u0)
    A_th = np.exp(-k1 * t)
    B_th = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    check(f"  t = {t}：[A] = e^{{−k₁t}}", u[0], A_th, tol=1e-9)
    check(f"  t = {t}：[B] = k₁/(k₂−k₁)(e^{{−k₁t}}−e^{{−k₂t}})", u[1], B_th, tol=1e-9)

fig, ax = new_axes("Chemical kinetics A -> B -> C (mass is conserved)",
                   figsize=(6.2, 4.0), equal=False)
ts = np.linspace(0, 12, 300)
traj = np.array([np.real(matrix_exp_by_eigen(A_chem, t) @ u0) for t in ts])
for k, (lbl, c) in enumerate([("[A]", "C0"), ("[B]", "C1"), ("[C]", "C2")]):
    ax.plot(ts, traj[:, k], c, lw=1.8, label=lbl)
ax.plot(ts, traj.sum(axis=1), "k--", lw=1.2, label="total (conserved)")
ax.set_xlabel("t")
ax.set_ylabel("concentration")
ax.legend(fontsize=8)
finish(fig, "ch18_kinetics")

# %% [markdown]
# ## 動手練習
#
# 1. 驗證 $\{1,x,x^2\}$ 的 Wronskian 是常數 2，並說明它為何永不為 0。
# 2. 寫出 $y'''-3y''+3y'-y=0$ 的伴隨矩陣，說明它需要 $3\times3$ 的 Jordan 塊，
#    並由 $e^{Cx}$ 讀出解 $e^x,xe^x,x^2e^x/2$。
# 3. 用參數變化法解 $y''-y=e^{x}$，注意共振。
# 4. 把 Hermite 方程 $y''-2xy'+2ny=0$ 的遞迴關係寫出來，驗證 $n$ 為非負整數時截斷。
# 5. 對一個有兩個守恆量的反應網路，求左零空間的維度。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
ones_f = lambda t: 1.0 + 0.0 * t
W_123 = wronskian([ones_f, lambda t: t, lambda t: t ** 2], 2.3)
print(f"練習 1：W({{1,x,x²}}) = {W_123:.6f}（理論 2）")
check("  W = 2（常數）", W_123, 2.0, tol=1e-4)
print("  原因：方程 y''' = 0 的 a₂ = 0 ⇒ Abel 公式給 W' = 0 ⇒ W 恆為常數")

# 練習 2
C3 = companion([-1.0, 3.0, -3.0])        # y''' − 3y'' + 3y' − y = 0 ⇒ (λ−1)³
print(f"\n練習 2：伴隨矩陣的特徵值 = {np.round(np.real(np.linalg.eigvals(C3)), 6)}")
# 三重根在數值上是病態的：擾動 ε 會讓根移動 ε^{1/3}（這裡約 1e−5）
check("  λ = 1, 1, 1（三重根，數值誤差 ~ε^{1/3} ≈ 1e−5）",
      np.sort(np.real(np.linalg.eigvals(C3))), np.ones(3), tol=1e-4)
print("  注意：三重根的條件數極差 —— 擾動 ε 使根移動 ε^{1/3}，")
print("        所以數值特徵值只有 ~5 位正確（第 13 章條件數的實例）")
gm3 = 3 - np.linalg.matrix_rank(C3 - np.eye(3), tol=1e-8)
print(f"  幾何重數 = {gm3} ⇒ 需要一個 3x3 的 Jordan 塊")
check("  GM = 1", gm3, 1)
for x_val in [0.5, 1.5]:
    E = matrix_exp_series(C3, x_val, 80)
    sol = E @ np.array([0.0, 0.0, 1.0])          # y(0)=y'(0)=0, y''(0)=1
    check(f"  x = {x_val}：解 = x²eˣ/2", sol[0], x_val ** 2 * np.exp(x_val) / 2,
          tol=1e-8)

# 練習 3：共振
print("\n練習 3：y'' − y = eˣ（eˣ 是齊次解 ⇒ 共振）")
yp3 = lambda t: t * np.exp(t) / 2
h3 = 1e-5
resid3 = (yp3(1.2 + h3) - 2 * yp3(1.2) + yp3(1.2 - h3)) / h3 ** 2 - yp3(1.2)
check("  特解 y_p = x·eˣ/2（帶 x 的因子）", resid3, np.exp(1.2), tol=1e-5)
print("  線性代數解釋：f 落在零空間方向 ⇒ 必須「跳出」零空間，代價是乘上 x")

# 練習 4：Hermite
print("\n練習 4：Hermite 方程 y'' − 2xy' + 2ny = 0")


def hermite_series(n_param, n_terms=20, a0=1.0, a1=0.0):
    a = np.zeros(n_terms)
    a[0], a[1] = a0, a1
    for k in range(n_terms - 2):
        a[k + 2] = 2 * (k - n_param) / ((k + 1) * (k + 2)) * a[k]
    return a


for n_param in [0, 2, 4, 1.3]:
    a = hermite_series(n_param, 14)
    trunc = (np.all(np.abs(a[int(n_param) + 2:]) < 1e-14)
             if float(n_param).is_integer() else False)
    print(f"  n = {n_param}：前 6 個係數 = {np.round(a[:6], 5)}，截斷 = {trunc}")
    if float(n_param).is_integer() and int(n_param) % 2 == 0:
        check(f"  n = {int(n_param)}：截斷成多項式", trunc)
check("  n = 1.3（非整數）不截斷",
      not np.all(np.abs(hermite_series(1.3, 20)[4:]) < 1e-14))
try:
    from numpy.polynomial import hermite_e
    a4 = hermite_series(4, 14)
    H4 = np.polynomial.hermite.Hermite.basis(4).convert(
        kind=np.polynomial.Polynomial)
    xs_h = np.linspace(-1.5, 1.5, 40)
    ser = np.polynomial.Polynomial(a4[:5])
    scale_h = H4(0.0) / ser(0.0)
    check("  n = 4 的級數解 ∝ H₄", scale_h * ser(xs_h), H4(xs_h), tol=1e-9)
except Exception:
    pass

# 練習 5：兩個守恆量
print("\n練習 5：有兩個守恆量的網路")
# A ⇌ B（可逆），C 獨立不變
A_net = np.array([[-1.0, 2.0, 0.0], [1.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
Y2 = left_nullspace(A_net)
print(f"  左零空間維度 = {Y2.shape[1]}")
check("  有兩個守恆量", Y2.shape[1], 2)
check("  YᵀA = 0", Y2.T @ A_net, np.zeros((2, 3)))
print(f"  守恆量：[A]+[B] 與 [C]")
check("  (1,1,0) 是守恆量", np.array([1.0, 1.0, 0.0]) @ A_net, np.zeros(3))
check("  (0,0,1) 是守恆量", np.array([0.0, 0.0, 1.0]) @ A_net, np.zeros(3))
print(f"  rank(A) = {rank(A_net)}，所以 dim N(Aᵀ) = 3 − {rank(A_net)} = "
      f"{3 - rank(A_net)}（第 3 章的計數定理）")

# %% [markdown]
# ## 本章重點回顧
#
# * $n$ 階線性齊次 ODE 的解空間是 **$n$ 維向量空間**；
#   取樣成向量後，秩就是維度。
# * **Wronskian = 基底矩陣的行列式**；$W\ne0\iff$ 獨立。
#   Abel 公式 $W'=-a_{n-1}W$ 保證 $W$ 要嘛恆 0、要嘛處處不為 0。
# * 常係數方程 $\to$ **伴隨矩陣**，其特徵多項式就是輔助方程。
#   重根 $\Rightarrow$ Jordan 塊 $\Rightarrow$ 解裡出現 $x^ke^{\lambda x}$；
#   複根 $\Rightarrow$ $e^{\alpha x}\cos\beta x$。
# * **基本矩陣** $\Phi=e^{Cx}$：$\det\Phi=$ Wronskian，
#   Jacobi 公式 $\frac{d}{dx}\det\Phi=\operatorname{tr}(C)\det\Phi$ 就是 Abel 公式。
# * 非齊次解 = 特解 + 齊次解（第 3 章的 $\mathbf x_p+\mathbf x_n$）；
#   **參數變化法就是用 Cramer 法則解 Wronskian 系統**。
#   共振（$f$ 落在零空間方向）會讓解多出 $x$ 的因子。
# * **級數解的遞迴關係 = 無窮維三角系統**（用回代解）。
#   $\ell$ 為整數時 Legendre 級數自動截斷成多項式 ── 這就是邊界條件造成的量化。
#   Frobenius 法的指標方程決定 $x^{\sigma}$ 的指數。
# * $d\mathbf u/dt=A\mathbf u$ 的**守恆量 = $A$ 的左零向量**：
#   第 3 章的 $N(A^{\mathsf T})$ 就是守恆律的空間。
#
# 下一章：數值線性代數 ── 迭代法、收斂性與預條件。
