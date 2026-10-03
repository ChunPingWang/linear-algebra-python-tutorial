# %% [markdown]
# # 第 8 章　線性微分方程與矩陣指數 $e^{At}$
#
# > 對應 Strang 6.5（解線性微分方程），並銜接 Riley 15.1（常係數線性 ODE）、
# > 9（法模態）、27.6（ODE 數值解）、20（擴散方程與波方程）。
#
# 一句話總結本章：
#
# $$\frac{d\mathbf u}{dt}=A\mathbf u,\quad \mathbf u(0)\ \text{已知}
# \qquad\Longrightarrow\qquad
# \mathbf u(t)=e^{At}\mathbf u(0)
# =\sum_i c_i e^{\lambda_i t}\mathbf x_i$$
#
# 離散情形的 $A^k$ 換成連續情形的 $e^{At}$；$\lambda^k$ 換成 $e^{\lambda t}$。
#
# | 問題 | 解 | 穩定條件 |
# |---|---|---|
# | $\mathbf u_{k+1}=A\mathbf u_k$ | $\mathbf u_k=\sum c_i\lambda_i^k\mathbf x_i$ | 所有 $|\lambda_i|<1$ |
# | $d\mathbf u/dt=A\mathbf u$ | $\mathbf u(t)=\sum c_ie^{\lambda_it}\mathbf x_i$ | 所有 $\operatorname{Re}\lambda_i<0$ |
#
# ```bash
# python chapters/ch08_differential_equations.py
# python tools/build_notebooks.py ch08
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

from linalg_tutorial.eigen import (
    diagonalize,
    matrix_exp_by_eigen,
    matrix_exp_series,
)
from linalg_tutorial.elimination import second_difference_matrix
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 8.1　把微分方程交給特徵向量
#
# 代入 $\mathbf u=e^{\lambda t}\mathbf x$：
#
# $$\frac{d}{dt}\left(e^{\lambda t}\mathbf x\right)=\lambda e^{\lambda t}\mathbf x
# \quad\text{要等於}\quad A\left(e^{\lambda t}\mathbf x\right)
# \quad\Longrightarrow\quad A\mathbf x=\lambda\mathbf x$$
#
# 所以**每個特徵對 $(\lambda,\mathbf x)$ 給一個純指數解**。三步驟（與第 6 章完全平行）：
#
# 1. 把 $\mathbf u(0)$ 寫成特徵向量的組合 $\mathbf u(0)=X\mathbf c$
# 2. 每個特徵向量乘上成長因子 $e^{\lambda_it}$
# 3. 加回來

# %%
section("8.1 三步驟解 du/dt = Au")

A = np.array([[0.0, 1.0], [1.0, 0.0]])
u0 = np.array([4.0, 2.0])
lam, X = np.linalg.eigh(A)
print(f"A = [[0,1],[1,0]]：λ = {lam}（即 −1 與 1），特徵向量 = (1,−1)/√2 與 (1,1)/√2")

c = np.linalg.solve(X, u0)
print(f"\n① 分解：u(0) = {u0} = {np.round(c, 4)} 乘上特徵向量")


def u_exact(t):
    """u(t) = Σ cᵢ e^{λᵢt} xᵢ"""
    return sum(c[i] * np.exp(lam[i] * t) * X[:, i] for i in range(2))


for t in [0.0, 0.5, 1.0, 2.0]:
    print(f"  t = {t:<4} u(t) = {np.round(u_exact(t), 6)}"
          f"   e^{{At}}u(0) = {np.round(matrix_exp_by_eigen(A, t) @ u0, 6)}")
    check(f"  兩種算法相同", u_exact(t), matrix_exp_by_eigen(A, t) @ u0)

# 驗證它真的滿足微分方程（數值微分）
h = 1e-6
for t in [0.3, 1.1]:
    du_dt = (u_exact(t + h) - u_exact(t - h)) / (2 * h)
    check(f"t = {t}：du/dt = Au", du_dt, A @ u_exact(t), tol=1e-6)

# y + z 與 y − z 解耦：Strang 的手算技巧
print("\n手算技巧：y' = z, z' = y ⇒ (y+z)' = (y+z)，(y−z)' = −(y−z)")
print("  → y+z 像 e^t 成長、y−z 像 e^{−t} 衰減，這正是兩個特徵方向")

# %% [markdown]
# ## 8.2　矩陣指數 $e^{At}$
#
# $$e^{At}=I+At+\frac{(At)^2}{2!}+\frac{(At)^3}{3!}+\cdots$$
#
# 三個事實：
#
# * $\dfrac{d}{dt}e^{At}=Ae^{At}$ ⇒ $\mathbf u(t)=e^{At}\mathbf u(0)$ 解微分方程
# * 若 $A=X\Lambda X^{-1}$ 則 $e^{At}=Xe^{\Lambda t}X^{-1}$（對角線放 $e^{\lambda_it}$）
# * 級數定義**不需要可對角化** ── 這是它比特徵分解更一般的地方
#
# 性質：$(e^{At})^{-1}=e^{-At}$、$\lambda(e^{At})=e^{\lambda t}$、
# $A$ 反對稱 $\Rightarrow e^{At}$ 正交。
#
# ⚠️ 注意：$e^{A}e^{B}\ne e^{A+B}$（除非 $AB=BA$）。

# %%
section("8.2 矩陣指數的三種算法")

A = np.array([[1.0, 1.0], [0.0, 2.0]])
t = 0.8
E_series = matrix_exp_series(A, t, terms=80)
E_eigen = matrix_exp_by_eigen(A, t)
show_matrix("e^{At}（級數）", E_series)
show_matrix("e^{At}（特徵分解）", E_eigen)
check("級數 = 特徵分解", E_series, E_eigen, tol=1e-10)

try:
    from scipy.linalg import expm
    check("與 scipy.linalg.expm 一致", E_series, expm(A * t), tol=1e-10)
except ImportError:
    print("  （未安裝 scipy，略過與 expm 的比較）")

print(f"\n理論值（三角矩陣）：[[e^t, e^{{2t}}−e^t], [0, e^{{2t}}]]")
theory = np.array([[np.exp(t), np.exp(2 * t) - np.exp(t)], [0.0, np.exp(2 * t)]])
check("與手算公式一致", E_series, theory, tol=1e-10)

check("(e^{At})⁻¹ = e^{−At}", np.linalg.inv(E_series), matrix_exp_series(A, -t, 80),
      tol=1e-9)
check("e^{At} 的特徵值 = e^{λt}",
      np.sort(np.real(np.linalg.eigvals(E_series))),
      np.sort(np.exp(np.real(np.linalg.eigvals(A)) * t)), tol=1e-9)
check("e^{A(s+t)} = e^{As}e^{At}", matrix_exp_series(A, 1.3, 90),
      matrix_exp_series(A, 0.5, 90) @ matrix_exp_series(A, 0.8, 90), tol=1e-8)

B = np.array([[0.0, 1.0], [1.0, 0.0]])
print(f"\nAB ≠ BA ⇒ e^A e^B ≠ e^{{A+B}}：")
print(f"  ‖e^A e^B − e^{{A+B}}‖ = "
      f"{np.linalg.norm(matrix_exp_series(A) @ matrix_exp_series(B) - matrix_exp_series(A + B)):.4f}")
check("此例確實不相等",
      not np.allclose(matrix_exp_series(A) @ matrix_exp_series(B),
                      matrix_exp_series(A + B)))

# 反對稱矩陣 → 旋轉
Askew = np.array([[0.0, 1.0], [-1.0, 0.0]])
for t_ in [0.0, 0.5, 1.0, np.pi / 2]:
    E = matrix_exp_series(Askew, t_, 80)
    check(f"A 反對稱：e^{{At}} 正交（t = {t_:.3f}）", E.T @ E, np.eye(2), tol=1e-10)
show_matrix("e^{At}（A 反對稱 ⇒ 旋轉矩陣 [[cos t, sin t], [−sin t, cos t]]）",
            matrix_exp_series(Askew, 1.0, 80))
check("= [[cos1, sin1], [−sin1, cos1]]", matrix_exp_series(Askew, 1.0, 80),
      np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]]), tol=1e-10)

# 不可對角化也能算
Jord = np.array([[1.0, 1.0], [0.0, 1.0]])
check("Jordan 塊不可對角化", diagonalize(Jord) is None)
E_j = matrix_exp_series(Jord, 2.0, 80)
show_matrix("e^{Jt}（t = 2）：對角線 e^t，右上角 t·e^t", E_j)
check("= [[e², 2e²], [0, e²]]", E_j,
      np.array([[np.exp(2.0), 2 * np.exp(2.0)], [0.0, np.exp(2.0)]]), tol=1e-10)
print("  重根 λ 的第二個解是 t·e^{λt}x —— 這正是 Jordan 形式的來源")

# %% [markdown]
# ## 8.3　穩定性：$\mathbf u(t)\to\mathbf 0$ 的條件
#
# $$\mathbf u(t)\to\mathbf 0\iff
# \text{所有 }\operatorname{Re}\lambda_i<0$$
#
# 因為 $|e^{\lambda t}|=e^{(\operatorname{Re}\lambda)t}$，虛部只負責振盪。
#
# $2\times2$ 的完整判準（只看 trace 與 det！）：
#
# $$T=\operatorname{trace}A=\lambda_1+\lambda_2<0
# \quad\text{且}\quad
# D=\det A=\lambda_1\lambda_2>0$$

# %%
section("8.3 2x2 的穩定性判準")

examples = {
    "穩定（兩個負實根）": np.array([[-3.0, 1.0], [0.0, -2.0]]),
    "穩定螺旋（複根，實部 < 0）": np.array([[-1.0, -4.0], [1.0, -1.0]]),
    "中性（純虛根，等振幅振盪）": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "不穩定（有正實根）": np.array([[1.0, 2.0], [0.0, -3.0]]),
    "鞍點（一正一負）": np.array([[2.0, 0.0], [0.0, -1.0]]),
}
print(f"{'類型':<28} | {'trace':>7} | {'det':>7} | {'λ':<28} | 穩定?")
print("-" * 92)
for name, M in examples.items():
    lam = np.linalg.eigvals(M)
    T, D = np.trace(M), np.linalg.det(M)
    stable = bool(np.all(np.real(lam) < 0))
    print(f"{name:<28} | {T:>7.2f} | {D:>7.2f} | {str(np.round(lam, 3)):<28} | {stable}")
    check(f"  {name}：T<0 且 D>0 ⟺ 穩定", (T < 0 and D > 0) == stable)

# 數值驗證：長時間行為
print("\n長時間行為（t = 20 時的 ‖u(t)‖，起點 u(0) = (1,1)）：")
u0 = np.array([1.0, 1.0])
for name, M in examples.items():
    norm20 = np.linalg.norm(matrix_exp_by_eigen(M, 20.0) @ u0)
    trend = "→ 0" if norm20 < 1e-6 else ("有界振盪" if norm20 < 10 else "→ ∞")
    print(f"  {name:<28} ‖u(20)‖ = {norm20:>12.4g}  {trend}")

# 相圖
fig = plt.figure(figsize=(11.0, 3.4))
for k, (name, M) in enumerate(list(examples.items())[:3]):
    ax = fig.add_subplot(1, 3, k + 1)
    gx, gy = np.meshgrid(np.linspace(-2, 2, 17), np.linspace(-2, 2, 17))
    du = M[0, 0] * gx + M[0, 1] * gy
    dv = M[1, 0] * gx + M[1, 1] * gy
    ax.streamplot(gx, gy, du, dv, density=1.0, color="C7", linewidth=0.7,
                  arrowsize=0.8)
    ts = np.linspace(0, 6, 400)
    for start in [np.array([1.5, 1.5]), np.array([-1.5, 1.0]), np.array([0.5, -1.8])]:
        traj = np.array([matrix_exp_by_eigen(M, tt) @ start for tt in ts])
        ax.plot(np.real(traj[:, 0]), np.real(traj[:, 1]), "C0", lw=1.4)
    ax.set_title(["stable node", "stable spiral", "center (neutral)"][k], fontsize=9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
finish(fig, "ch08_phase_portraits")

# %% [markdown]
# ## 8.4　二階方程 → 一階系統（伴隨矩陣）
#
# 力學的基本方程 $my''+by'+ky=0$。令 $\mathbf u=(y,y')$：
#
# $$\frac{d}{dt}\begin{bmatrix}y\\y'\end{bmatrix}
# =\begin{bmatrix}0&1\\-k/m&-b/m\end{bmatrix}
# \begin{bmatrix}y\\y'\end{bmatrix}$$
#
# 這個**伴隨矩陣**的特徵方程正好是 $m\lambda^2+b\lambda+k=0$ ──
# 微分方程課本裡「代入 $y=e^{\lambda t}$」的那一步，在線性代數裡就是 $\det(A-\lambda I)=0$。
#
# 阻尼的三種情形：
#
# | $b^2$ vs $4mk$ | 根 | 物理 |
# |---|---|---|
# | $b^2<4mk$ | 複根 | 欠阻尼（振盪衰減）|
# | $b^2=4mk$ | 重根 | 臨界阻尼（$te^{\lambda t}$）|
# | $b^2>4mk$ | 兩個實根 | 過阻尼（不振盪）|

# %%
section("8.4 二階方程與伴隨矩陣")


def companion(m, b, k):
    """my'' + by' + ky = 0 的伴隨矩陣。"""
    return np.array([[0.0, 1.0], [-k / m, -b / m]])


m, k = 1.0, 4.0
print(f"m = {m}, k = {k}（自然頻率 ω₀ = {np.sqrt(k / m):.4g}）\n")
for b, tag in [(0.0, "無阻尼"), (1.0, "欠阻尼"), (4.0, "臨界阻尼"), (10.0, "過阻尼")]:
    A = companion(m, b, k)
    lam = np.linalg.eigvals(A)
    roots = np.roots([m, b, k])
    disc = b ** 2 - 4 * m * k
    kind = ("無阻尼（純虛根）" if b == 0 else "欠阻尼（複根）" if disc < 0
            else "臨界阻尼（重根）" if abs(disc) < 1e-12 else "過阻尼（兩實根）")
    print(f"  b = {b:<5} λ = {np.round(lam, 4)}   判別式 = {disc:+.1f}  → {kind}")
    check(f"    伴隨矩陣的 λ = mλ²+bλ+k 的根",
          np.sort_complex(lam), np.sort_complex(roots), tol=1e-8)

# 解出 y(t) 並與解析解比較（欠阻尼）
b = 1.0
A = companion(m, b, k)
u0 = np.array([1.0, 0.0])              # y(0) = 1, y'(0) = 0
omega_d = np.sqrt(k / m - (b / (2 * m)) ** 2)
zeta = b / (2 * np.sqrt(m * k))
print(f"\n欠阻尼解析解：y(t) = e^{{−bt/2m}}[cos(ω_d t) + (b/2mω_d)sin(ω_d t)]")
print(f"  阻尼比 ζ = {zeta:.4f}，阻尼頻率 ω_d = {omega_d:.4f}")
for t in [0.0, 0.5, 1.5, 3.0]:
    y_mat = (matrix_exp_by_eigen(A, t) @ u0)[0]
    y_ana = np.exp(-b * t / (2 * m)) * (np.cos(omega_d * t)
                                        + b / (2 * m * omega_d) * np.sin(omega_d * t))
    check(f"  t = {t}：e^{{At}}u(0) 的第一分量 = 解析解", np.real(y_mat), y_ana, tol=1e-9)

fig, ax = new_axes("Damped oscillator via matrix exponential",
                   figsize=(6.2, 4.0), equal=False)
ts = np.linspace(0, 10, 500)
for b_, c in zip([0.0, 1.0, 4.0, 10.0], ["C0", "C1", "C2", "C3"]):
    A_ = companion(m, b_, k)
    ys = [np.real((matrix_exp_series(A_, tt, 60) @ u0)[0]) for tt in ts]
    ax.plot(ts, ys, c, lw=1.6, label=f"b = {b_}")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.legend(fontsize=8)
finish(fig, "ch08_damping")

# %% [markdown]
# ## 8.5　差分方法：三種方式繞圓，三種命運
#
# 用 $y''=-y$（答案是 $\cos t$，相圖上是單位圓）來測試三種差分法。
# 把它寫成 $\mathbf U_{n+1}=A\mathbf U_n$，命運完全由 $|\lambda(A)|$ 決定：
#
# | 方法 | 遞推 | $|\lambda|$ | 行為 |
# |---|---|---|---|
# | 前向（顯式）| $Y_{n+1}=Y_n+\Delta t\,Z_n$ | $\sqrt{1+\Delta t^2}>1$ | **向外螺旋** |
# | 後向（隱式）| $Y_{n+1}-\Delta t\,Z_{n+1}=Y_n$ | $<1$ | **向內螺旋** |
# | 中央（leapfrog）| $Y_{n+1}-2Y_n+Y_{n-1}=-\Delta t^2Y_n$ | $=1$（$\Delta t<2$）| **留在圓上** |
#
# 這是計算科學的核心教訓：同一個微分方程可以離散成穩定或不穩定的差分方程。

# %%
section("8.5 前向/後向/中央差分的穩定性")

dt = 2 * np.pi / 32
A_fwd = np.array([[1.0, dt], [-dt, 1.0]])
A_bwd = np.linalg.inv(np.array([[1.0, -dt], [dt, 1.0]]))
print(f"Δt = 2π/32 = {dt:.6f}")
for name, M in [("前向（顯式）", A_fwd), ("後向（隱式）", A_bwd)]:
    lam = np.linalg.eigvals(M)
    print(f"  {name}：|λ| = {np.abs(lam)[0]:.8f}  "
          f"（理論 {'√(1+Δt²) = %.8f' % np.sqrt(1 + dt ** 2) if name.startswith('前') else '1/√(1+Δt²) = %.8f' % (1 / np.sqrt(1 + dt ** 2))}）")
check("前向：|λ| > 1 ⇒ 不穩定（向外螺旋）", np.abs(np.linalg.eigvals(A_fwd))[0] > 1)
check("後向：|λ| < 1 ⇒ 過度耗散（向內螺旋）", np.abs(np.linalg.eigvals(A_bwd))[0] < 1)
check("前向 |λ| = √(1+Δt²)", np.abs(np.linalg.eigvals(A_fwd))[0], np.sqrt(1 + dt ** 2))

# leapfrog：Y_{n+1} = 2Y_n − Y_{n-1} − Δt²Y_n，寫成 2x2 系統
A_leap = np.array([[2.0 - dt ** 2, -1.0], [1.0, 0.0]])
lam_leap = np.linalg.eigvals(A_leap)
print(f"  中央（leapfrog）：|λ| = {np.abs(lam_leap)}")
check("leapfrog：|λ| = 1 ⇒ 中性穩定（Δt < 2）", np.abs(lam_leap), np.ones(2), tol=1e-10)

# 跑 32 步看半徑
print(f"\n{'方法':<16} | {'32 步後的半徑':>14} | {'應為 1':>8}")
print("-" * 44)
radii = {}
for name, M in [("前向", A_fwd), ("後向", A_bwd)]:
    U = np.array([1.0, 0.0])
    for _ in range(32):
        U = M @ U
    radii[name] = np.linalg.norm(U)
    print(f"{name:<16} | {np.linalg.norm(U):>14.6f} | {1.0:>8.1f}")
Y = [1.0, np.cos(dt)]
for _ in range(32):
    Y.append((2 - dt ** 2) * Y[-1] - Y[-2])
r_leap = np.sqrt(Y[-1] ** 2 + ((Y[-1] - Y[-2]) / dt) ** 2)
print(f"{'中央 (leapfrog)':<16} | {r_leap:>14.6f} | {1.0:>8.1f}")
check("前向半徑 > 1", radii["前向"] > 1.0)
check("後向半徑 < 1", radii["後向"] < 1.0)
check("leapfrog 半徑 ≈ 1（留在圓附近，誤差 < 2%）", abs(r_leap - 1.0) < 0.02)
print("  （leapfrog 的 |λ| 剛好是 1，但守恆量是一個略為傾斜的橢圓，"
      "所以用 Y² + Z² 量測會有 O(Δt²) 的偏差）")

fig, ax = new_axes("Three difference methods on the unit circle", figsize=(5.4, 5.2))
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(th), np.sin(th), "k--", lw=1, label="exact circle")
for name, M, c in [("forward (explicit)", A_fwd, "C3"), ("backward (implicit)", A_bwd, "C0")]:
    U = np.array([1.0, 0.0])
    pts = [U]
    for _ in range(32):
        U = M @ U
        pts.append(U)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], c + "o-", ms=3, lw=1, label=name)
Y = [1.0, np.cos(dt)]
for _ in range(32):
    Y.append((2 - dt ** 2) * Y[-1] - Y[-2])
Z = [(Y[i + 1] - Y[i]) / dt for i in range(len(Y) - 1)]
ax.plot(Y[:-1], Z, "C2o-", ms=3, lw=1, label="centered (leapfrog)")
ax.legend(fontsize=7)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
finish(fig, "ch08_difference_methods")

# %% [markdown]
# ## 8.6　從 ODE 到 PDE：熱傳導與波方程
#
# 把空間也離散化（第 2 章的 $K$ 矩陣），偏微分方程就變成 $d\mathbf u/dt=-K\mathbf u$：
#
# $$\frac{\partial u}{\partial t}=\frac{\partial^2u}{\partial x^2}
# \ \longrightarrow\ \frac{d\mathbf u}{dt}=-\frac{1}{h^2}K\mathbf u
# \qquad(\text{熱傳導：}\ \lambda<0\Rightarrow\text{指數衰減})$$
#
# $$\frac{\partial^2u}{\partial t^2}=\frac{\partial^2u}{\partial x^2}
# \ \longrightarrow\ \frac{d^2\mathbf u}{dt^2}=-\frac{1}{h^2}K\mathbf u
# \qquad(\text{波：}\ \omega^2=\lambda>0\Rightarrow\text{振盪})$$
#
# $K$ 正定（第 7 章）$\Rightarrow$ 熱傳導必定穩定、波必定振盪。
# **特徵向量 = 正弦模態、特徵值 = 衰減率或頻率平方**。

# %%
section("8.6 熱傳導方程：每個正弦模態各自衰減")

n = 20
h = 1.0 / (n + 1)
x = np.linspace(h, 1 - h, n)
K = second_difference_matrix(n) / h ** 2
lamK, QK = np.linalg.eigh(K)
print(f"−K 的特徵值（衰減率）前 5 個最慢：{np.round(-lamK[:5], 2)}")
print(f"理論值 −k²π²（k = 1..5）：{np.round(-(np.arange(1, 6) * np.pi) ** 2, 2)}")
check("最慢衰減率 ≈ −π²", -lamK[0], -np.pi ** 2, tol=0.1)

u0 = np.where(np.abs(x - 0.5) < 0.1, 1.0, 0.0)        # 初始的方形熱源
coef = QK.T @ u0


def heat_solution(t):
    """u(t) = Σ cₖ e^{−λₖt} qₖ：每個模態獨立衰減。"""
    return QK @ (coef * np.exp(-lamK * t))


for t in [0.0, 0.001, 0.01, 0.05]:
    u = heat_solution(t)
    print(f"  t = {t:<6} 最高溫 = {u.max():.4f}，總熱量 = {u.sum() * h:.4f}")
check("t = 0 時還原初始條件", heat_solution(0.0), u0, tol=1e-10)
check("解滿足 du/dt = −Ku（數值微分）",
      (heat_solution(0.01 + 1e-7) - heat_solution(0.01 - 1e-7)) / 2e-7,
      -K @ heat_solution(0.01), tol=1e-3)
check("長時間後趨近 0（固定邊界）", np.linalg.norm(heat_solution(10.0)) < 1e-10)

fig = plt.figure(figsize=(10.5, 3.6))
ax = fig.add_subplot(1, 2, 1)
for t, c in zip([0.0, 0.002, 0.01, 0.03, 0.08], ["C0", "C1", "C2", "C3", "C4"]):
    ax.plot(x, heat_solution(t), c, lw=1.6, label=f"t = {t}")
ax.set_title("Heat equation: du/dt = -K u", fontsize=9)
ax.set_xlabel("x")
ax.set_ylabel("temperature")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
omega = np.sqrt(lamK)
wave = lambda t: QK @ (coef * np.cos(omega * t))
for t, c in zip([0.0, 0.1, 0.25, 0.5], ["C0", "C1", "C2", "C3"]):
    ax.plot(x, wave(t), c, lw=1.6, label=f"t = {t}")
ax.set_title("Wave equation: d2u/dt2 = -K u", fontsize=9)
ax.set_xlabel("x")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
finish(fig, "ch08_heat_wave")

print("\n波方程：ω² = λ ⇒ 模態以 cos(ωt) 振盪，能量守恆（不衰減）")
E0 = wave(0.0) @ (K @ wave(0.0))
for t in [0.0, 0.1, 0.3]:
    v = -QK @ (coef * omega * np.sin(omega * t))      # du/dt
    E = v @ v + wave(t) @ (K @ wave(t))
    print(f"  t = {t:<5} 總能量（動能 + 位能）= {E:.6f}")
check("波方程能量守恆", wave(0.0) @ (K @ wave(0.0)) + 0.0,
      (-QK @ (coef * omega * np.sin(omega * 0.3))) @
      (-QK @ (coef * omega * np.sin(omega * 0.3)))
      + wave(0.3) @ (K @ wave(0.3)), tol=1e-6)

# %% [markdown]
# ## 動手練習
#
# 1. 解 $d\mathbf u/dt=A\mathbf u$，$A=\begin{bmatrix}1&1&1\\0&2&1\\0&0&3\end{bmatrix}$，
#    $\mathbf u(0)=(9,7,4)$。（提示：特徵值就在對角線上）
# 2. 證明 $A$ 反對稱 $\Rightarrow\|\mathbf u(t)\|$ 不變（能量守恆）。
# 3. 對 $y''+2y'+5y=0$ 寫出伴隨矩陣，判斷阻尼類型，並畫出 $y(t)$。
# 4. 把 leapfrog 的 $\Delta t$ 加大到 $>2$，觀察 $|\lambda|$ 何時超過 1（CFL 條件的雛形）。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A1 = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])
u0 = np.array([9.0, 7.0, 4.0])
lam1, X1 = np.linalg.eig(A1)
c1 = np.linalg.solve(X1, u0)
print(f"練習 1：λ = {np.sort(np.real(lam1))}（= 1, 2, 3）")
print(f"  u(0) 的特徵向量係數 c = {np.round(np.real(c1), 4)}")
for t in [0.0, 0.5, 1.0]:
    u_eig = np.real(X1 @ (np.exp(lam1 * t) * c1))
    check(f"  t = {t}：Σcᵢe^{{λᵢt}}xᵢ = e^{{At}}u(0)", u_eig,
          matrix_exp_by_eigen(A1, t) @ u0, tol=1e-8)
print(f"  u(1) = {np.round(np.real(X1 @ (np.exp(lam1 * 1.0) * c1)), 4)}")

# 練習 2
print("\n練習 2：A 反對稱 ⇒ d/dt‖u‖² = 2uᵀ(Au) = 0（因為 uᵀAu = 0）")
Ask = np.array([[0.0, 2.0, -1.0], [-2.0, 0.0, 3.0], [1.0, -3.0, 0.0]])
u0 = np.array([1.0, -2.0, 0.5])
for t in [0.0, 0.7, 2.5]:
    u = matrix_exp_series(Ask, t, 100) @ u0
    check(f"  t = {t}：‖u(t)‖ = ‖u(0)‖", np.linalg.norm(u), np.linalg.norm(u0),
          tol=1e-9)
rng = np.random.default_rng(0)
v = rng.standard_normal(3)
check("  任意 v：vᵀAv = 0", v @ Ask @ v, 0.0)

# 練習 3
A3 = companion(1.0, 2.0, 5.0)
lam3 = np.linalg.eigvals(A3)
print(f"\n練習 3：y''+2y'+5y=0 → λ = {np.round(lam3, 4)}（−1 ± 2i）")
print(f"  判別式 = 4 − 20 = −16 < 0 ⇒ 欠阻尼；實部 −1 < 0 ⇒ 穩定")
check("  λ = −1 ± 2i", np.sort_complex(lam3), np.sort_complex(np.array([-1 + 2j, -1 - 2j])),
      tol=1e-10)
y3 = [np.real((matrix_exp_series(A3, tt, 70) @ np.array([1.0, 0.0]))[0])
      for tt in [0.0, 1.0, 2.0, 4.0]]
print(f"  y(0,1,2,4) = {np.round(y3, 5)}（振盪衰減）")
check("  y(4) 幾乎衰減到 0", abs(y3[-1]) < 0.05)

# 練習 4：CFL
print("\n練習 4：leapfrog 的穩定界限")
print(f"{'Δt':>8} | {'|λ|':>12} | 穩定?")
print("-" * 32)
for dt_ in [0.5, 1.0, 1.9, 2.0, 2.1, 3.0]:
    M = np.array([[2.0 - dt_ ** 2, -1.0], [1.0, 0.0]])
    mag = np.max(np.abs(np.linalg.eigvals(M)))
    print(f"{dt_:>8.2f} | {mag:>12.6f} | {mag <= 1 + 1e-9}")
check("Δt < 2 時 |λ| = 1", np.max(np.abs(np.linalg.eigvals(
    np.array([[2.0 - 1.9 ** 2, -1.0], [1.0, 0.0]])))), 1.0, tol=1e-10)
check("Δt > 2 時 |λ| > 1（不穩定）", np.max(np.abs(np.linalg.eigvals(
    np.array([[2.0 - 2.1 ** 2, -1.0], [1.0, 0.0]])))) > 1.0)
print("  Δt < 2 就是這個問題的 CFL 條件（第 19、20 章會再見到）")

# %% [markdown]
# ## 本章重點回顧
#
# * $d\mathbf u/dt=A\mathbf u$ 的每個特徵對給一個純指數解 $e^{\lambda t}\mathbf x$；
#   完整解是它們的組合，三步驟與 $A^k$ 完全平行。
# * $e^{At}=\sum(At)^n/n!=Xe^{\Lambda t}X^{-1}$。級數定義**不需要可對角化**，
#   重根時會自然冒出 $te^{\lambda t}$。
# * **穩定** $\iff$ 所有 $\operatorname{Re}\lambda<0$；$2\times2$ 只要檢查
#   $\operatorname{trace}<0$ 且 $\det>0$。
# * 二階方程經伴隨矩陣變成一階系統，特徵方程就是課本的 $m\lambda^2+b\lambda+k=0$。
# * 同一個微分方程可以離散成不穩定（前向）、過度耗散（後向）或中性（leapfrog）的差分方程，
#   全看 $|\lambda|$ 與 1 的關係。
# * 把空間離散化後，熱傳導與波方程都變成 $K$ 的特徵問題：
#   正弦模態 + 指數衰減／正弦振盪。
#
# 下一章：SVD ── 把「對稱矩陣才有的好性質」推廣到**任意**矩陣。
