# %% [markdown]
# # 第 11 章　最佳化中的線性代數
#
# > 對應 Strang 第 9 章（9.1 多變數函數的極小化、9.2 反向傳播與隨機梯度下降、
# > 9.3 約束、Lagrange 乘子與最小範數、9.4 線性規劃、賽局理論與對偶），
# > 以及 Riley 5.8–5.9（多變數的穩定點與約束極值）、22（變分法）。
#
# 本章把微積分與線性代數接起來。核心模型是**二次函數**
#
# $$Q(\mathbf x)=\tfrac12\mathbf x^{\mathsf T}S\mathbf x-\mathbf b^{\mathsf T}\mathbf x,
# \qquad \nabla Q=S\mathbf x-\mathbf b,
# \qquad \nabla Q=\mathbf 0\iff S\mathbf x=\mathbf b$$
#
# 「求極小」與「解線性方程」是同一件事（當 $S$ 對稱正定）。
#
# | 方法 | 用到的資訊 | 每步成本 | 收斂速度 |
# |---|---|---|---|
# | 牛頓法 | $\nabla F$ 與 Hessian $S$ | 高（解 $n\times n$ 系統）| 平方收斂 |
# | 梯度下降 | 只要 $\nabla F$ | 低 | $\left(\frac{1-b}{1+b}\right)$，$b=1/\kappa$ |
# | 動量法 | 只要 $\nabla F$ | 低 | $\left(\frac{1-\sqrt b}{1+\sqrt b}\right)$ ← **大躍進** |
# | 隨機梯度下降 | 一次只看一筆資料 | 極低 | 深度學習的主力 |
#
# ```bash
# python chapters/ch11_optimization.py
# python tools/build_notebooks.py ch11
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

from linalg_tutorial.eigen import is_positive_definite
from linalg_tutorial.orthogonal import least_squares_normal, min_norm_solution
from linalg_tutorial.svd_tools import condition_number
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 11.1　梯度、Hessian 與牛頓法
#
# 多變數的 Taylor 展開（只取到二階）：
#
# $$F(\mathbf x+\Delta\mathbf x)\approx F(\mathbf x)
# +(\Delta\mathbf x)^{\mathsf T}\nabla F
# +\tfrac12(\Delta\mathbf x)^{\mathsf T}S(\Delta\mathbf x)$$
#
# * $\nabla F=(\partial F/\partial x_1,\dots,\partial F/\partial x_n)$ ── **梯度**
# * $S_{ij}=\partial^2F/\partial x_i\partial x_j$ ── **Hessian**（對稱！）
#
# 把 $\nabla F(\mathbf x+\Delta\mathbf x)=\nabla F+S\,\Delta\mathbf x=\mathbf 0$ 解出來：
#
# $$\boxed{\Delta\mathbf x=-S^{-1}\nabla F}\qquad\text{（牛頓法一步）}$$
#
# 二次函數時牛頓法**一步到位**；一般函數則是**平方收斂**（正確位數每步加倍）。

# %%
section("11.1 二次模型：求極小 = 解 Sx = b")

S = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b = np.array([1.0, 2.0, -1.0])
check("S 對稱正定 ⇒ Q 是碗狀（有唯一極小）", is_positive_definite(S))

Q = lambda x: 0.5 * x @ S @ x - b @ x
gradQ = lambda x: S @ x - b
x_star = np.linalg.solve(S, b)
print(f"極小點 x* = S⁻¹b = {x_star}")
print(f"Q(x*) = {Q(x_star):.6f}")
check("∇Q(x*) = 0", gradQ(x_star), np.zeros(3))

rng = np.random.default_rng(0)
worse = all(Q(x_star + 0.3 * rng.standard_normal(3)) > Q(x_star) for _ in range(500))
check("任何擾動都讓 Q 變大 ⇒ 真的是極小", worse)
print(f"Q(x*) 的封閉式 = −½bᵀS⁻¹b = {-0.5 * b @ x_star:.6f}")
check("Q(x*) = −½bᵀS⁻¹b", Q(x_star), -0.5 * b @ x_star)

# 牛頓法一步到位
x0 = np.array([5.0, -3.0, 4.0])
x1 = x0 - np.linalg.solve(S, gradQ(x0))
print(f"\n牛頓法從 x₀ = {x0} 出發：")
print(f"  一步之後 x₁ = {x1}")
check("二次函數：牛頓法一步到達極小點", x1, x_star)

# 數值驗證梯度與 Hessian
def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def numeric_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                       - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


xt = np.array([1.0, -0.5, 2.0])
check("∇Q 的解析式 = 數值梯度", gradQ(xt), numeric_grad(Q, xt), tol=1e-6)
check("Hessian = S（二次函數的 Hessian 是常數）", numeric_hessian(Q, xt), S, tol=1e-4)

# %% [markdown]
# ### 牛頓法的平方收斂：算 $\sqrt{4}$
#
# 解 $F'(x)=x^2-4=0$，牛頓法變成
# $x_{k+1}=\frac12\left(x_k+\frac4{x_k}\right)$。
# 誤差每一步**平方**：$|x_{k+1}-2|\le\frac14|x_k-2|^2$。

# %%
section("11.1 牛頓法的平方收斂")

x = 4.0
print(f"{'k':>3} | {'x_k':>22} | {'誤差 |x_k − 2|':>22} | {'誤差比 e_k/e_{k−1}²':>20}")
print("-" * 76)
errs = []
for k in range(6):
    err = abs(x - 2.0)
    ratio = f"{err / errs[-1] ** 2:.4f}" if errs and errs[-1] > 0 else "—"
    print(f"{k:>3} | {x:>22.18f} | {err:>22.3e} | {ratio:>20}")
    errs.append(err)
    if err > 0:
        x = 0.5 * (x + 4.0 / x)
check("誤差平方收斂（e_k ≈ e_{k−1}²/4）", errs[3] <= errs[2] ** 2 / 4 * 1.01 + 1e-18)
print("  小數點後正確位數：1 → 2 → 4 → 8 → 16 ...（每步加倍）")

# 一般函數的牛頓法
F = lambda v: np.exp(v[0] + 0.3 * v[1]) + v[0] ** 2 + 2 * v[1] ** 2
gradF = lambda v: np.array([np.exp(v[0] + 0.3 * v[1]) + 2 * v[0],
                            0.3 * np.exp(v[0] + 0.3 * v[1]) + 4 * v[1]])


def hessF(v):
    e = np.exp(v[0] + 0.3 * v[1])
    return np.array([[e + 2, 0.3 * e], [0.3 * e, 0.09 * e + 4]])


xk = np.array([2.0, -2.0])
print(f"\n非二次函數 F = e^(x+0.3y) + x² + 2y² 的牛頓法：")
for k in range(6):
    g = np.linalg.norm(gradF(xk))
    print(f"  k = {k}：x = {np.round(xk, 10)}，‖∇F‖ = {g:.3e}")
    if g < 1e-14:
        break
    xk = xk - np.linalg.solve(hessF(xk), gradF(xk))
check("收斂到 ∇F = 0", np.linalg.norm(gradF(xk)) < 1e-10)
check("Hessian 在極小點正定", is_positive_definite(hessF(xk)))
check("解析 Hessian = 數值 Hessian", hessF(xk), numeric_hessian(F, xk), tol=1e-4)

# %% [markdown]
# ## 11.2　梯度下降、鋸齒路徑與動量
#
# $$\mathbf x_{k+1}=\mathbf x_k-s\,\nabla F(\mathbf x_k)$$
#
# Strang 的「超級模型」$F=\frac12(x^2+by^2)$（$0<b\le1$）有**精確解**：
# 從 $(x_0,y_0)=(b,1)$ 出發，用最佳步長時
#
# $$F(\mathbf x_k)=\left(\frac{1-b}{1+b}\right)^{2k}F(\mathbf x_0)$$
#
# $b=\lambda_{\min}/\lambda_{\max}=1/\kappa$ 很小時，比值趨近 1 ── **進展幾乎停滯**，
# 路徑在狹長山谷裡來回鋸齒。
#
# **動量法**（Polyak）：像一顆重球滾下山，記住上一步的方向
#
# $$\mathbf z_k=\nabla F(\mathbf x_k)+\beta\mathbf z_{k-1},\qquad
# \mathbf x_{k+1}=\mathbf x_k-s\mathbf z_k$$
#
# 最佳參數與收斂率（$b$ 變成 $\sqrt b$ ── 這就是「加速」）：
#
# $$s=\left(\frac{2}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\right)^2,\quad
# \beta=\left(\frac{\sqrt{\lambda_{\max}}-\sqrt{\lambda_{\min}}}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\right)^2,
# \quad\text{率}=\frac{1-\sqrt b}{1+\sqrt b}$$

# %%
section("11.2 鋸齒路徑：b = λ_min/λ_max 決定一切")


def gd_exact_line_search(S, x0, steps):
    """對 F = ½xᵀSx 做梯度下降，每步用精確線搜尋 s = gᵀg/(gᵀSg)。"""
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        g = S @ x
        if np.linalg.norm(g) < 1e-300:
            break
        s = (g @ g) / (g @ S @ g)
        x = x - s * g
        xs.append(x.copy())
    return np.array(xs)


print(f"{'b = 1/κ':>10} | {'理論率 (1−b)/(1+b)':>20} | {'實測率':>12} | "
      f"{'20 步後 F/F₀':>14}")
print("-" * 64)
for b in [1.0, 0.5, 0.1, 0.01]:
    Sb = np.diag([1.0, b])
    x0 = np.array([b, 1.0])
    xs = gd_exact_line_search(Sb, x0, 20)
    F = lambda v: 0.5 * v @ Sb @ v
    theory = (1 - b) / (1 + b)
    measured = np.sqrt(F(xs[-1]) / F(xs[0])) ** (1 / 20) if F(xs[-1]) > 0 else 0.0
    print(f"{b:>10} | {theory:>20.6f} | {measured:>12.6f} | "
          f"{F(xs[-1]) / F(xs[0]):>14.3e}")
    if b < 1:
        check(f"  b = {b}：實測率 = (1−b)/(1+b)", measured, theory, tol=1e-6)
    else:
        check("  b = 1（圓形碗）：一步到位", np.linalg.norm(xs[1]) < 1e-14)

# 每一步都轉 90°
Sb = np.diag([1.0, 0.1])
xs = gd_exact_line_search(Sb, np.array([0.1, 1.0]), 12)
dirs = np.diff(xs, axis=0)
angles = [np.degrees(np.arccos(np.clip(
    dirs[i] @ dirs[i + 1] / (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[i + 1])),
    -1, 1))) for i in range(len(dirs) - 1)]
print(f"\n相鄰步伐的夾角 = {np.round(angles[:5], 4)} ...")
check("精確線搜尋 ⇒ 每步正好轉 90°（鋸齒的數學原因）",
      np.allclose(angles, 90.0, atol=1e-6))


def momentum_descent(S, x0, steps, s, beta):
    """帶動量的梯度下降。"""
    xs = [x0.copy()]
    x, z = x0.copy(), np.zeros_like(x0)
    for _ in range(steps):
        z = S @ x + beta * z
        x = x - s * z
        xs.append(x.copy())
    return np.array(xs)


section("11.2 動量法：b → √b 的加速")

print(f"{'b':>8} | {'普通下降率':>14} | {'動量法率':>14} | {'10 步的誤差比':>26}")
print("-" * 72)
for b in [0.1, 0.01, 0.001]:
    Sb = np.diag([1.0, b])
    lmax, lmin = 1.0, b
    s_opt = (2 / (np.sqrt(lmax) + np.sqrt(lmin))) ** 2
    beta_opt = ((np.sqrt(lmax) - np.sqrt(lmin)) / (np.sqrt(lmax) + np.sqrt(lmin))) ** 2
    plain = (1 - b) / (1 + b)
    accel = (1 - np.sqrt(b)) / (1 + np.sqrt(b))
    xs_m = momentum_descent(Sb, np.array([b, 1.0]), 10, s_opt, beta_opt)
    F = lambda v: 0.5 * v @ Sb @ v
    print(f"{b:>8} | {plain:>14.6f} | {accel:>14.6f} | "
          f"普通 {plain ** 10:>8.4f} vs 動量 {accel ** 10:>8.4f}")
    check(f"  b = {b}：動量法的理論率更小", accel < plain)
    check(f"  b = {b}：動量法 10 步確實比理論率界住",
          np.sqrt(F(xs_m[-1]) / F(xs_m[0])) <= accel ** 10 * 20 + 1e-12)

print("\n條件數 κ = λ_max/λ_min = 1/b 控制一切：")
for kappa in [10, 100, 10000]:
    b = 1 / kappa
    print(f"  κ = {kappa:>6}：普通下降需要 ~{np.log(0.01) / np.log((1 - b) / (1 + b)):>8.0f} 步，"
          f"動量法只需 ~{np.log(0.01) / np.log((1 - np.sqrt(b)) / (1 + np.sqrt(b))):>6.0f} 步"
          f"（降到 1% 誤差）")

fig = plt.figure(figsize=(10.8, 4.0))
b = 0.05
Sb = np.diag([1.0, b])
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
Fg = 0.5 * (gx ** 2 + b * gy ** 2)
for k, (title, path) in enumerate([
        ("gradient descent (zig-zag)", gd_exact_line_search(Sb, np.array([b, 1.0]), 30)),
        ("with momentum (accelerated)",
         momentum_descent(Sb, np.array([b, 1.0]), 30,
                          (2 / (1 + np.sqrt(b))) ** 2,
                          ((1 - np.sqrt(b)) / (1 + np.sqrt(b))) ** 2))]):
    ax = fig.add_subplot(1, 2, k + 1)
    ax.contour(gx, gy, Fg, levels=np.logspace(-4, -0.5, 12), cmap="Blues", alpha=0.7)
    ax.plot(path[:, 0], path[:, 1], "C3.-", ms=4, lw=1.1)
    ax.plot(0, 0, "k*", ms=12)
    ax.set_title(f"{title}\nF = (x^2 + {b}y^2)/2", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
finish(fig, "ch11_zigzag_momentum")

# %% [markdown]
# ## 11.3　隨機梯度下降與反向傳播
#
# 深度學習的損失函數是**很多項的總和**
# $L(\mathbf x)=\sum_{i=1}^{N}\ell_i(\mathbf x)$，$N$ 可能上百萬。
# 算完整梯度太貴 ⇒ 每步**隨機抽一小批**（mini-batch）來估計梯度。
#
# 最小平方就是最好的示範：$\min\sum_i(\mathbf a_i^{\mathsf T}\mathbf x-b_i)^2$。
# 每次只用一列 $\mathbf a_i$ 做投影，就是 **Kaczmarz 法**。
#
# **反向傳播** = 鏈鎖律的矩陣版本。對兩層網路
# $\mathbf y=W_2\,\sigma(W_1\mathbf v)$，梯度由後往前算：
#
# $$\frac{\partial L}{\partial W_2}=\boldsymbol\delta_2\mathbf h^{\mathsf T},\qquad
# \boldsymbol\delta_1=(W_2^{\mathsf T}\boldsymbol\delta_2)\odot\sigma'(\mathbf z_1),\qquad
# \frac{\partial L}{\partial W_1}=\boldsymbol\delta_1\mathbf v^{\mathsf T}$$

# %%
section("11.3 隨機梯度下降 vs 完整梯度（最小平方）")

rng = np.random.default_rng(1)
N, n = 200, 10
A = rng.standard_normal((N, n))
x_true = rng.standard_normal(n)
b_data = A @ x_true + 0.1 * rng.standard_normal(N)
x_ls = least_squares_normal(A, b_data)
loss = lambda x: np.sum((A @ x - b_data) ** 2) / N


def full_gradient_descent(steps, s):
    x = np.zeros(n)
    hist = [loss(x)]
    for _ in range(steps):
        g = 2 * A.T @ (A @ x - b_data) / N
        x = x - s * g
        hist.append(loss(x))
    return x, hist


def sgd(steps, s0, batch=1, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    hist = [loss(x)]
    for k in range(steps):
        idx = r.integers(0, N, batch)
        g = 2 * A[idx].T @ (A[idx] @ x - b_data[idx]) / batch
        x = x - s0 / (1 + 0.01 * k) * g          # 逐步縮小學習率
        hist.append(loss(x))
    return x, hist


x_fg, hist_fg = full_gradient_descent(300, 0.02)
x_sgd, hist_sgd = sgd(300, 0.02, batch=1)
x_mb, hist_mb = sgd(300, 0.02, batch=16)
print(f"最小平方最佳解的損失 = {loss(x_ls):.6f}")
print(f"完整梯度 300 步      ：損失 {loss(x_fg):.6f}，"
      f"‖x − x_LS‖ = {np.linalg.norm(x_fg - x_ls):.4f}")
print(f"SGD（batch=1）300 步 ：損失 {loss(x_sgd):.6f}，"
      f"‖x − x_LS‖ = {np.linalg.norm(x_sgd - x_ls):.4f}")
print(f"SGD（batch=16）300 步：損失 {loss(x_mb):.6f}，"
      f"‖x − x_LS‖ = {np.linalg.norm(x_mb - x_ls):.4f}")
print(f"\n每步成本：完整梯度要看 {N} 筆資料，batch=1 只看 1 筆（便宜 {N} 倍）")
check("三種方法都接近最小平方解", np.linalg.norm(x_mb - x_ls) < 0.2)
check("SGD 的損失明顯下降", hist_sgd[-1] < hist_sgd[0] / 10)
check("完整梯度單調下降（凸函數 + 合適步長）",
      all(hist_fg[i + 1] <= hist_fg[i] + 1e-12 for i in range(len(hist_fg) - 1)))

# Kaczmarz：一次只投影到一個方程
def kaczmarz(steps, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    hist = [loss(x)]
    for _ in range(steps):
        i = r.integers(0, N)
        a = A[i]
        x = x + (b_data[i] - a @ x) / (a @ a) * a       # 投影到超平面 aᵀx = bᵢ
        hist.append(loss(x))
    return x, hist


x_kz, hist_kz = kaczmarz(600)
print(f"Kaczmarz（隨機投影）600 步：損失 {loss(x_kz):.6f}")
check("Kaczmarz 也收斂到最小平方附近", np.linalg.norm(x_kz - x_ls) < 0.3)

fig, ax = new_axes("Full gradient vs stochastic gradient", figsize=(6.2, 4.2),
                   equal=False)
for hist, name in [(hist_fg, "full gradient (300 x N data)"),
                   (hist_mb, "mini-batch 16"),
                   (hist_sgd, "SGD batch 1"),
                   (hist_kz[:301], "Kaczmarz")]:
    ax.semilogy(np.array(hist) - loss(x_ls) + 1e-12, lw=1.4, label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("loss - optimal loss")
ax.legend(fontsize=8)
finish(fig, "ch11_sgd")

# %%
section("11.3 反向傳播 = 鏈鎖律的矩陣形式")

rng = np.random.default_rng(3)
n_in, n_hid, n_out = 4, 5, 2
W1 = rng.standard_normal((n_hid, n_in)) * 0.5
W2 = rng.standard_normal((n_out, n_hid)) * 0.5
v = rng.standard_normal(n_in)
target = rng.standard_normal(n_out)

sigma = np.tanh
sigma_prime = lambda z: 1 - np.tanh(z) ** 2


def forward(W1, W2, v):
    z1 = W1 @ v
    h = sigma(z1)
    y = W2 @ h
    return z1, h, y


def loss_net(W1, W2):
    _, _, y = forward(W1, W2, v)
    return 0.5 * np.sum((y - target) ** 2)


def backprop(W1, W2):
    """由後往前算梯度（這就是反向傳播）。"""
    z1, h, y = forward(W1, W2, v)
    delta2 = y - target                        # ∂L/∂y
    dW2 = np.outer(delta2, h)                  # ∂L/∂W₂ = δ₂hᵀ
    delta1 = (W2.T @ delta2) * sigma_prime(z1)  # 把誤差「傳回」第一層
    dW1 = np.outer(delta1, v)                  # ∂L/∂W₁ = δ₁vᵀ
    return dW1, dW2


dW1, dW2 = backprop(W1, W2)

# 與數值梯度逐項比對（梯度檢查，實務上訓練網路一定要做這一步）
def numeric_grad_matrix(f, W, h=1e-6):
    G = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            G[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return G


dW1_num = numeric_grad_matrix(lambda M: loss_net(M, W2), W1)
dW2_num = numeric_grad_matrix(lambda M: loss_net(W1, M), W2)
check("∂L/∂W₁：反向傳播 = 數值梯度", dW1, dW1_num, tol=1e-7)
check("∂L/∂W₂：反向傳播 = 數值梯度", dW2, dW2_num, tol=1e-7)
print(f"參數個數 = {W1.size + W2.size}；")
print(f"數值梯度需要 {2 * (W1.size + W2.size)} 次前向計算，")
print(f"反向傳播只要 1 次前向 + 1 次後向 —— 這就是深度學習可行的原因")

# 真的訓練幾步看損失下降
W1t, W2t = W1.copy(), W2.copy()
print(f"\n用梯度下降訓練（學習率 0.3）：")
for k in range(0, 41, 10):
    print(f"  第 {k:>2} 步：損失 = {loss_net(W1t, W2t):.8f}")
    for _ in range(10 if k < 40 else 0):
        g1, g2 = backprop(W1t, W2t)
        W1t -= 0.3 * g1
        W2t -= 0.3 * g2
check("訓練讓損失下降", loss_net(W1t, W2t) < loss_net(W1, W2) / 10)

# %% [markdown]
# ## 11.4　約束、最小範數與 Lagrange 乘子
#
# 在約束 $3x+4y=1$ 下求「最小的」$(x,y)$ ── **答案完全取決於用哪個範數**：
#
# | 範數 | 單位球的形狀 | 最佳解 $\mathbf v^*$ | 特徵 |
# |---|---|---|---|
# | $\ell^1=|x|+|y|$ | 菱形 | $(0,\tfrac14)$ | **稀疏！**（尖角先碰到直線）|
# | $\ell^2=\sqrt{x^2+y^2}$ | 圓 | $(\tfrac{3}{25},\tfrac{4}{25})$ | 最小平方 |
# | $\ell^\infty=\max$ | 正方形 | $(\tfrac17,\tfrac17)$ | 各分量平均 |
#
# $\ell^1$ 給出稀疏解，是壓縮感知（compressed sensing）與 LASSO 的全部基礎。
#
# **Lagrange 乘子**：把約束乘上 $\lambda$ 併入目標函數
#
# $$L(\mathbf x,\lambda)=F(\mathbf x)-\lambda^{\mathsf T}(A\mathbf x-\mathbf b),
# \qquad \nabla_{\mathbf x}L=0,\ \nabla_{\lambda}L=0$$
#
# 對 $F=\|\mathbf x\|^2$ 得到**鞍點系統**
#
# $$\begin{bmatrix}2I&-A^{\mathsf T}\\A&0\end{bmatrix}
# \begin{bmatrix}\mathbf x\\\lambda\end{bmatrix}
# =\begin{bmatrix}\mathbf 0\\\mathbf b\end{bmatrix}
# \qquad\Longrightarrow\qquad
# \mathbf x^*=A^{\mathsf T}(AA^{\mathsf T})^{-1}\mathbf b=A^{+}\mathbf b$$

# %%
section("11.4 三種範數下的最小解：ℓ¹ 給出稀疏解")

a = np.array([3.0, 4.0])


def min_norm_on_line(p):
    """在 3x + 4y = 1 上最小化 ℓᵖ 範數（用細網格掃描，p = 1, 2, inf）。"""
    ts = np.linspace(-2, 2, 2000001)
    pts = np.column_stack([ts, (1 - 3 * ts) / 4])
    if p == 1:
        vals = np.abs(pts).sum(axis=1)
    elif p == 2:
        vals = np.linalg.norm(pts, axis=1)
    else:
        vals = np.abs(pts).max(axis=1)
    return pts[int(np.argmin(vals))], vals.min()


for p, exact, name in [(1, np.array([0.0, 0.25]), "ℓ¹（菱形）"),
                       (2, np.array([3 / 25, 4 / 25]), "ℓ²（圓）"),
                       (np.inf, np.array([1 / 7, 1 / 7]), "ℓ∞（正方形）")]:
    v_star, val = min_norm_on_line(p)
    print(f"  {name:<14} v* = {np.round(v_star, 6)}，範數 = {val:.6f}"
          f"（理論 {np.round(exact, 6)}）")
    check(f"  {name} 的解正確", v_star, exact, tol=1e-4)
    check(f"  {name} 的解在直線上", a @ v_star, 1.0, tol=1e-5)
print("  ℓ¹ 的解只有一個非零分量 ⇒ 稀疏（LASSO / compressed sensing 的原理）")
check("ℓ¹ 解的非零分量個數 = 1",
      int(np.sum(np.abs(min_norm_on_line(1)[0]) > 1e-6)), 1)

fig, ax = new_axes("Minimum norm on the line 3x + 4y = 1", figsize=(5.6, 5.2))
ts = np.linspace(-0.3, 0.45, 50)
ax.plot(ts, (1 - 3 * ts) / 4, "k-", lw=2, label="3x + 4y = 1")
th = np.linspace(0, 2 * np.pi, 400)
for p, col, r_ in [(1, "C0", 0.25), (2, "C2", 0.2), (np.inf, "C3", 1 / 7)]:
    if p == 1:
        pts = r_ * np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]).T
    elif p == 2:
        pts = r_ * np.array([np.cos(th), np.sin(th)])
    else:
        pts = r_ * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]).T
    ax.plot(pts[0], pts[1], col, lw=1.6,
            label={1: "l1 ball (diamond)", 2: "l2 ball (circle)"}.get(p, "linf ball (square)"))
    v_star, _ = min_norm_on_line(p)
    ax.plot(*v_star, col + "o", ms=8)
ax.legend(fontsize=7)
ax.set_xlim(-0.3, 0.45)
ax.set_ylim(-0.3, 0.45)
finish(fig, "ch11_min_norm")

# %%
section("11.4 Lagrange 乘子與鞍點系統")

A = np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 1.0]])
b = np.array([6.0, 1.0])
m, n = A.shape

# 鞍點（KKT）系統
KKT = np.block([[2 * np.eye(n), -A.T], [A, np.zeros((m, m))]])
rhs = np.concatenate([np.zeros(n), b])
sol = np.linalg.solve(KKT, rhs)
x_lag, lam = sol[:n], sol[n:]
show_matrix("鞍點矩陣 [[2I, −Aᵀ], [A, 0]]", KKT)
print(f"解：x* = {np.round(x_lag, 6)}，λ = {np.round(lam, 6)}")
check("約束滿足 Ax = b", A @ x_lag, b)
check("x* = Aᵀ(AAᵀ)⁻¹b（最小範數解）",
      x_lag, A.T @ np.linalg.inv(A @ A.T) @ b)
check("x* = A⁺b（偽逆！與第 4 章一致）", x_lag, min_norm_solution(A, b))
check("x* 在列空間中（零空間成分為 0）",
      x_lag, A.T @ np.linalg.lstsq(A.T, x_lag, rcond=None)[0], tol=1e-9)

rng = np.random.default_rng(5)
from linalg_tutorial.subspaces import nullspace
Nn = nullspace(A)
worse = all(np.linalg.norm(x_lag + Nn @ rng.standard_normal(Nn.shape[1]))
            >= np.linalg.norm(x_lag) - 1e-12 for _ in range(500))
check("任何滿足約束的其他解都更長", worse)

# Lagrange 乘子的意義：成本對 b 的導數
print(f"\nLagrange 乘子的意義：λ = ∂(最小成本)/∂b")
cost = lambda bb: np.linalg.norm(min_norm_solution(A, bb)) ** 2
h = 1e-6
for i in range(m):
    e = np.zeros(m)
    e[i] = h
    dcost = (cost(b + e) - cost(b - e)) / (2 * h)
    print(f"  ∂cost/∂b{i + 1} = {dcost:.8f}，λ{i + 1} = {lam[i]:.8f}")
    check(f"  λ{i + 1} = ∂cost/∂b{i + 1}", lam[i], dcost, tol=1e-5)
print("  ⇒ 乘子是「影子價格」：放鬆約束一單位能省多少成本（經濟學的核心概念）")

# 等式約束下的二次規劃
S = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
c = np.array([1.0, 0.0, -2.0])
KKT2 = np.block([[S, -A.T], [A, np.zeros((m, m))]])
sol2 = np.linalg.solve(KKT2, np.concatenate([c, b]))
x_qp, lam_qp = sol2[:n], sol2[n:]
print(f"\n一般二次規劃 min ½xᵀSx − cᵀx s.t. Ax = b：")
print(f"  x* = {np.round(x_qp, 6)}")
check("  約束成立", A @ x_qp, b)
check("  ∇L = Sx − c − Aᵀλ = 0", S @ x_qp - c - A.T @ lam_qp, np.zeros(n))
Qf = lambda x: 0.5 * x @ S @ x - c @ x
worse = all(Qf(x_qp + Nn @ (0.3 * rng.standard_normal(Nn.shape[1]))) > Qf(x_qp) - 1e-12
            for _ in range(500))
check("  在約束集合上真的是極小", worse)

# %% [markdown]
# ## 11.5　線性規劃、對偶與賽局
#
# $$\text{原問題：}\min\ \mathbf c^{\mathsf T}\mathbf x
# \quad\text{s.t.}\quad A\mathbf x=\mathbf b,\ \mathbf x\ge\mathbf 0$$
#
# $$\text{對偶問題：}\max\ \mathbf y^{\mathsf T}\mathbf b
# \quad\text{s.t.}\quad A^{\mathsf T}\mathbf y\le\mathbf c$$
#
# **弱對偶**（一行證明）：
# $\mathbf y^{\mathsf T}\mathbf b=\mathbf y^{\mathsf T}A\mathbf x
# =(A^{\mathsf T}\mathbf y)^{\mathsf T}\mathbf x\le\mathbf c^{\mathsf T}\mathbf x$
#
# **強對偶定理**：兩者的最佳值**相等**。
#
# 因為成本是線性的，最小值一定出現在可行區域的**角點**上 ──
# 這就是 Dantzig 的單形法（simplex method）在做的事。
#
# 最漂亮的對偶實例：**最大流 = 最小割**。

# %%
section("11.5 線性規劃：最小值在角點上")

c = np.array([5.0, 3.0, 8.0])
A_lp = np.array([[1.0, 1.0, 2.0]])
b_lp = np.array([4.0])
print("Strang 的例子：博士/朋友/電腦每小時收費 5, 3, 8 元，")
print("每小時各解 1, 1, 2 題，要解 4 題。min cᵀx s.t. x₁+x₂+2x₃ = 4, x ≥ 0")

corners = []
for j in range(3):
    x = np.zeros(3)
    x[j] = b_lp[0] / A_lp[0, j]
    corners.append(x)
print(f"\n{'角點':<22} | {'成本 cᵀx':>10}")
print("-" * 36)
for x in corners:
    print(f"{str(np.round(x, 3)):<22} | {c @ x:>10.2f}")
best = corners[int(np.argmin([c @ x for x in corners]))]
print(f"\n最佳角點 = {best}，最小成本 = {c @ best:.2f}（全部交給朋友做）")
check("最佳解在角點上", c @ best, 12.0)
rng = np.random.default_rng(0)
feasible_worse = True
for _ in range(2000):
    w = rng.dirichlet(np.ones(3))                   # 隨機可行點
    x = w / (A_lp[0] @ w) * b_lp[0]
    feasible_worse = feasible_worse and (c @ x >= c @ best - 1e-12)
check("沒有內部點比角點更便宜", feasible_worse)

# 對偶問題
y_dual = np.array([3.0])                            # 猜 y = 3
print(f"\n對偶問題：max yᵀb s.t. Aᵀy ≤ c，即 y ≤ 5, y ≤ 3, 2y ≤ 8 ⇒ y ≤ 3")
print(f"  最佳 y* = 3，對偶目標 yᵀb = {y_dual[0] * b_lp[0]:.2f}")
check("強對偶：max yᵀb = min cᵀx", y_dual @ b_lp, c @ best)
for y_try in [0.0, 1.0, 2.0, 3.0]:
    if np.all(A_lp.T @ np.array([y_try]) <= c + 1e-12):
        check(f"  弱對偶 y = {y_try}：yᵀb ≤ cᵀx", y_try * b_lp[0] <= c @ best + 1e-12)

# 用 scipy 驗證（若可用）
try:
    from scipy.optimize import linprog
    res = linprog(c, A_eq=A_lp, b_eq=b_lp, bounds=[(0, None)] * 3)
    check("與 scipy.optimize.linprog 一致", res.fun, c @ best, tol=1e-6)
    print(f"  scipy 的解 = {np.round(res.x, 6)}")
except ImportError:
    print("  （未安裝 scipy，略過 linprog 驗證）")

# %%
section("11.5 最大流 = 最小割（對偶的最美實例）")

# 圖：0 = source, 4 = sink
cap = {(0, 1): 7.0, (0, 2): 2.0, (0, 3): 8.0,
       (1, 2): 6.0, (1, 4): 4.0, (2, 4): 5.0, (3, 2): 4.0, (3, 4): 5.0}
nodes = sorted({u for u, _ in cap} | {v for _, v in cap})
source, sink = 0, 4


def max_flow(cap, source, sink):
    """Edmonds–Karp：用 BFS 找增廣路徑，直到找不到為止。"""
    residual = {k: v for k, v in cap.items()}
    for (u, v) in list(cap):
        residual.setdefault((v, u), 0.0)
    total = 0.0
    while True:
        parent, queue = {source: None}, [source]
        while queue and sink not in parent:
            u = queue.pop(0)
            for (a, v), c in residual.items():
                if a == u and c > 1e-12 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path, v = [], sink
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        bottleneck = min(residual[e] for e in path)
        for (u, v) in path:
            residual[(u, v)] -= bottleneck
            residual[(v, u)] += bottleneck
        total += bottleneck
    reachable = set(parent)                    # 最後一次 BFS 能到達的點 = 最小割的一側
    return total, reachable, residual


flow, reachable, residual = max_flow(cap, source, sink)
print(f"圖的邊與容量：{cap}")
print(f"\n最大流 = {flow:.1f}")
cut_edges = [(u, v) for (u, v) in cap if u in reachable and v not in reachable]
cut_cap = sum(cap[e] for e in cut_edges)
print(f"最小割的邊 = {cut_edges}，容量 = {cut_cap:.1f}")
check("最大流 = 最小割（強對偶）", flow, cut_cap)

# 窮舉所有割驗證
from itertools import combinations
others = [v for v in nodes if v not in (source, sink)]
min_cut = np.inf
for r in range(len(others) + 1):
    for subset in combinations(others, r):
        Sset = {source} | set(subset)
        capacity = sum(c for (u, v), c in cap.items() if u in Sset and v not in Sset)
        min_cut = min(min_cut, capacity)
print(f"窮舉所有 {2 ** len(others)} 種割，最小容量 = {min_cut:.1f}")
check("窮舉的最小割 = 最大流", min_cut, flow)
check("Kirchhoff 守恆：每個中間節點流入 = 流出",
      all(abs(sum(cap.get((u, v), 0.0) - residual[(u, v)]
                  for v in nodes if (u, v) in cap)
              - sum(cap.get((w, u), 0.0) - residual[(w, u)]
                    for w in nodes if (w, u) in cap)) < 1e-9
          for u in others))

# 二人零和賽局：minimax = maximin
section("11.5 二人零和賽局：鞍點 = minimax")

payoff = np.array([[3.0, -1.0, 2.0], [-2.0, 4.0, 1.0], [1.0, 0.0, 3.0]])
show_matrix("報償矩陣 A（R 選列付給 C，C 選欄）", payoff)
row_min = payoff.min(axis=1)
col_max = payoff.max(axis=0)
print(f"R 的保證（每列最小）= {row_min} → maximin = {row_min.max():.1f}")
print(f"C 的風險（每欄最大）= {col_max} → minimax = {col_max.min():.1f}")
if abs(row_min.max() - col_max.min()) < 1e-12:
    print("  純策略就有鞍點")
else:
    print("  maximin < minimax ⇒ 純策略沒有鞍點，需要混合策略（正是線性規劃）")
check("maximin ≤ minimax（弱對偶）", row_min.max() <= col_max.min() + 1e-12)

# 用 LP 求混合策略的值（若有 scipy）
try:
    from scipy.optimize import linprog
    # max v s.t. Aᵀx ≥ v·1, Σx = 1, x ≥ 0（變數 [x; v]，目標 min −v）
    n_s = payoff.shape[0]
    c_obj = np.zeros(n_s + 1)
    c_obj[-1] = -1.0
    A_ub = np.hstack([-payoff.T, np.ones((n_s, 1))])
    b_ub = np.zeros(n_s)
    A_eq = np.zeros((1, n_s + 1))
    A_eq[0, :n_s] = 1.0
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * n_s + [(None, None)])
    value = -res.fun
    strat = res.x[:n_s]
    print(f"\n混合策略的賽局值 = {value:.6f}")
    print(f"  R 的最佳混合策略 = {np.round(strat, 6)}")
    check("賽局值落在 [maximin, minimax] 之間",
          row_min.max() - 1e-9 <= value <= col_max.min() + 1e-9)
    check("策略是機率分布", float(strat.sum()), 1.0, tol=1e-8)
except ImportError:
    print("  （未安裝 scipy，略過混合策略的 LP）")

# %% [markdown]
# ## 動手練習
#
# 1. 對 $F(x,y)=\frac12(x^2+100y^2)$，比較梯度下降與動量法到達 $10^{-6}$ 精度所需步數。
# 2. 自己推導：為什麼精確線搜尋會讓相鄰兩步正交？
# 3. 用 Lagrange 乘子解「在 $x+y+z=3$ 下最小化 $x^2+2y^2+3z^2$」，並驗證乘子等於成本對約束的導數。
# 4. 寫一個 $\ell^1$ 最小化（LASSO）的簡單版本，觀察解的稀疏度隨懲罰係數的變化。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
b = 0.01
Sb = np.diag([1.0, 100.0]) / 100.0        # 等價於 ½(x² + 100y²) 的 Hessian 比例
S1 = np.diag([1.0, 100.0])
lmax, lmin = 100.0, 1.0
s_gd = 2 / (lmax + lmin)
s_mom = (2 / (np.sqrt(lmax) + np.sqrt(lmin))) ** 2
beta = ((np.sqrt(lmax) - np.sqrt(lmin)) / (np.sqrt(lmax) + np.sqrt(lmin))) ** 2
x0 = np.array([1.0, 1.0])
F1 = lambda v: 0.5 * v @ S1 @ v


def steps_to_tol(update, tol=1e-6, cap=100000):
    x, z = x0.copy(), np.zeros(2)
    for k in range(1, cap + 1):
        x, z = update(x, z)
        if F1(x) / F1(x0) < tol:
            return k
    return cap


k_gd = steps_to_tol(lambda x, z: (x - s_gd * (S1 @ x), z))
k_mom = steps_to_tol(lambda x, z: ((lambda zz: (x - s_mom * zz, zz))(S1 @ x + beta * z)))
print(f"練習 1：κ = {lmax / lmin:.0f}")
print(f"  梯度下降需要 {k_gd} 步，動量法需要 {k_mom} 步（快 {k_gd / k_mom:.1f} 倍）")
check("  動量法明顯更快", k_mom < k_gd / 2)

# 練習 2
print("\n練習 2：精確線搜尋 s* 使 d/ds F(x − s g) = 0 ⇒ −gᵀ∇F(x_new) = 0")
print("  也就是新的梯度 ⟂ 舊的方向 g ⇒ 下一步方向與這一步正交 ⇒ 鋸齒")
Sd = np.diag([1.0, 0.1])
x = np.array([0.1, 1.0])
g = Sd @ x
s = (g @ g) / (g @ Sd @ g)
x_new = x - s * g
check("  新梯度 ⟂ 舊步伐方向", (Sd @ x_new) @ g, 0.0)

# 練習 3
A3 = np.array([[1.0, 1.0, 1.0]])
b3 = np.array([3.0])
S3 = np.diag([2.0, 4.0, 6.0])              # ½xᵀSx = x² + 2y² + 3z²
KKT3 = np.block([[S3, -A3.T], [A3, np.zeros((1, 1))]])
sol3 = np.linalg.solve(KKT3, np.concatenate([np.zeros(3), b3]))
x3, lam3 = sol3[:3], sol3[3:]
print(f"\n練習 3：x* = {np.round(x3, 6)}，λ = {lam3[0]:.6f}")
check("  約束 x+y+z = 3 成立", A3 @ x3, b3)
check("  Sx = Aᵀλ（梯度條件）", S3 @ x3, A3.T @ lam3)
cost3 = lambda bb: (lambda xx: xx[0] ** 2 + 2 * xx[1] ** 2 + 3 * xx[2] ** 2)(
    np.linalg.solve(KKT3, np.concatenate([np.zeros(3), bb]))[:3])
d = (cost3(b3 + 1e-6) - cost3(b3 - 1e-6)) / 2e-6
print(f"  ∂cost/∂b = {d:.6f}，λ = {lam3[0]:.6f}")
check("  λ = ∂cost/∂b", lam3[0], d, tol=1e-5)
print(f"  （理論解 x* = (λ/2, λ/4, λ/6)，由 Σx = 3 定出 λ = {lam3[0]:.4f}）")

# 練習 4：LASSO by ISTA
print("\n練習 4：ℓ¹ 懲罰（LASSO）的稀疏度")
rng = np.random.default_rng(9)
m_l, n_l = 60, 40
A_l = rng.standard_normal((m_l, n_l))
x_sparse = np.zeros(n_l)
x_sparse[[3, 11, 27]] = [2.0, -3.0, 1.5]           # 只有 3 個非零
b_l = A_l @ x_sparse + 0.05 * rng.standard_normal(m_l)
L = np.linalg.norm(A_l, 2) ** 2


def ista(alpha, iters=4000):
    """迭代軟門檻法：min ½‖Ax−b‖² + α‖x‖₁。"""
    x = np.zeros(n_l)
    for _ in range(iters):
        x = x - (A_l.T @ (A_l @ x - b_l)) / L
        x = np.sign(x) * np.maximum(np.abs(x) - alpha / L, 0.0)   # 軟門檻
    return x


print(f"{'α':>8} | {'非零分量數':>12} | {'‖x−x_true‖':>14} | {'殘差':>10}")
print("-" * 52)
for alpha in [0.0, 0.5, 2.0, 10.0, 50.0]:
    xl = ista(alpha)
    nz = int(np.sum(np.abs(xl) > 1e-6))
    print(f"{alpha:>8.1f} | {nz:>12} | {np.linalg.norm(xl - x_sparse):>14.4f} "
          f"| {np.linalg.norm(A_l @ xl - b_l):>10.4f}")
check("  α 越大解越稀疏", int(np.sum(np.abs(ista(10.0)) > 1e-6))
      <= int(np.sum(np.abs(ista(0.5)) > 1e-6)))
x_mid = ista(2.0)
check("  適當的 α 能找回真正的 3 個非零位置",
      set(np.argsort(-np.abs(x_mid))[:3].tolist()) == {3, 11, 27})

# %% [markdown]
# ## 本章重點回顧
#
# * 二次模型 $Q=\frac12\mathbf x^{\mathsf T}S\mathbf x-\mathbf b^{\mathsf T}\mathbf x$：
#   $\nabla Q=S\mathbf x-\mathbf b$，所以**求極小 = 解 $S\mathbf x=\mathbf b$**。
# * 牛頓法 $\Delta\mathbf x=-S^{-1}\nabla F$：二次函數一步到位，一般函數平方收斂，
#   但每步要算 Hessian。
# * 梯度下降的收斂率由**條件數**決定：$\left(\frac{1-b}{1+b}\right)$，$b=1/\kappa$。
#   精確線搜尋使相鄰步伐正交 ⇒ 狹長山谷裡鋸齒前進。
# * **動量法把 $b$ 換成 $\sqrt b$**：$\kappa=10^4$ 時從數千步降到數十步。
# * 隨機梯度下降每步只看一小批資料；反向傳播用一次前向 + 一次後向就得到所有梯度
#   （別忘了做梯度檢查）。
# * 約束問題的答案取決於範數：$\ell^1$ 給**稀疏**解（LASSO、壓縮感知）、
#   $\ell^2$ 給最小平方、$\ell^\infty$ 給平均。
# * Lagrange 乘子把約束併進目標，得到**鞍點系統**
#   $\begin{bmatrix}S&-A^{\mathsf T}\\A&0\end{bmatrix}$；
#   乘子 $\lambda=\partial(\text{最小成本})/\partial\mathbf b$ 就是經濟學的「影子價格」。
# * 線性規劃的最小值在**角點**；原問題與對偶問題的最佳值相等（強對偶），
#   最美的實例就是**最大流 = 最小割**。
#
# 下一章：從資料學習 ── 分段線性函數、神經網路與共變異數矩陣。
