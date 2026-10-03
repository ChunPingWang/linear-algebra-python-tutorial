# %% [markdown]
# # 第 12 章　從資料學習：分段線性函數與共變異數矩陣
#
# > 對應 Strang 第 10 章（10.1 分段線性學習函數、10.2 建構與實驗、
# > 10.3 平均、變異數與共變異數），以及 Riley 30（機率）、31.2（樣本統計）。
#
# 深度學習的學習函數是**連續分段線性**（CPL）函數的合成鏈：
#
# $$F(\mathbf v)=F_L(F_{L-1}(\cdots F_2(F_1(\mathbf v))))，\qquad
# F_k(\mathbf v)=\text{ReLU}(A_k\mathbf v+\mathbf b_k)$$
#
# * **線性**（$A_k\mathbf v+\mathbf b_k$）：簡單、可計算
# * **分段**（ReLU）：提供非線性 ── 真實資料絕對需要
# * **連續**：模擬未知但合理的規則
#
# 本章後半回到統計的線性代數核心：**共變異數矩陣**
#
# $$V=\mathbb E\left[(\mathbf x-\mathbf m)(\mathbf x-\mathbf m)^{\mathsf T}\right]
# \quad\text{對稱半正定}，\qquad
# \operatorname{Cov}(A\mathbf x)=A\,V\,A^{\mathsf T}$$
#
# ```bash
# python chapters/ch12_learning_from_data.py
# python tools/build_notebooks.py ch12
# ```

# %%
import pathlib
import sys
from math import comb

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.eigen import is_positive_definite, is_symmetric
from linalg_tutorial.elimination import cholesky
from linalg_tutorial.svd_tools import pca
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 12.1　為什麼線性不夠？分段線性為什麼夠？
#
# 若 $F$ 只是線性的 $A\mathbf v$，那麼「兩個 0 相加」就該得到另一個 0 ──
# 但影像不能相加。辨識數字、臉孔的規則離線性很遠。
#
# ReLU $=\max(y,0)$ 是最簡單的非線性：**兩段線性**。
# 把它作用在 $A_1\mathbf v+\mathbf b_1$ 的每個分量上，
# $p_1$ 個超平面就把 $\mathbb R^{p_0}$ 切成
#
# $$r(p_1,p_0)=\binom{p_1}{0}+\binom{p_1}{1}+\dots+\binom{p_1}{p_0}\ \text{塊}$$
#
# 這個數字衡量網路的「表達力」（expressivity）。

# %%
section("12.1 ReLU 與分段線性")

relu = lambda y: np.maximum(y, 0.0)
rng = np.random.default_rng(0)
ys = rng.standard_normal(6)
print(f"y        = {ys}")
print(f"ReLU(y)  = {relu(ys)}")
check("ReLU 不是線性的（ReLU(−y) ≠ −ReLU(y)）",
      not np.allclose(relu(-ys), -relu(ys)))
check("ReLU 是正齊次的：ReLU(cy) = c·ReLU(y)（c > 0）",
      relu(3.0 * ys), 3.0 * relu(ys))
check("ReLU 連續（左右極限相同）", relu(np.array([-1e-12])), relu(np.array([0.0])),
      tol=1e-10)
print("  ⇒ ReLU 是「分段線性 + 連續」，但不是線性：正好是我們要的最小非線性")

# 線性函數辦不到的事：XOR
print("\nXOR 問題：線性函數無法分開")
V_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
w_xor = np.array([0.0, 1.0, 1.0, 0.0])
A_lin = np.column_stack([np.ones(4), V_xor])
coef, *_ = np.linalg.lstsq(A_lin, w_xor, rcond=None)
pred = A_lin @ coef
print(f"  最佳線性擬合的預測 = {np.round(pred, 4)}（目標 {w_xor}）")
print(f"  殘差 = {np.linalg.norm(pred - w_xor):.4f} ≠ 0 ⇒ 線性絕對做不到")
check("線性模型無法擬合 XOR", np.linalg.norm(pred - w_xor) > 0.5)

# 一個隱藏層 + ReLU 就可以（手算權重）
A1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
A2 = np.array([[1.0, -2.0]])
F = lambda v: A2 @ relu(A1 @ v + b1)
out = np.array([F(v)[0] for v in V_xor])
print(f"\n  兩層 ReLU 網路的輸出 = {out}（目標 {w_xor}）")
check("ReLU 網路精確解出 XOR", out, w_xor)
print("  手算權重：h₁ = ReLU(x+y)，h₂ = ReLU(x+y−1)，輸出 = h₁ − 2h₂")

# 切割計數公式
section("12.1 表達力：p₁ 個超平面把 R^p₀ 切成幾塊？")

def count_regions_exact(A, b):
    """精確計算 p₁ 個超平面把 R^p₀ 切成幾塊。

    每一塊對應一個「啟動模式」s ∈ {±1}^p₁。該模式存在 ⟺ 線性不等式系統
    sᵢ(aᵢ·x + bᵢ) > 0（對所有 i）有解。用線性規劃判定可行性即可。
    """
    from itertools import product
    from scipy.optimize import linprog

    p1, p0 = A.shape
    count = 0
    for signs in product([1.0, -1.0], repeat=p1):
        sv = np.array(signs)
        # 變數 [x; t]，求 max t s.t. sᵢ(aᵢx + bᵢ) ≥ t, 0 ≤ t ≤ 1
        c_obj = np.zeros(p0 + 1)
        c_obj[-1] = -1.0
        A_ub = np.hstack([-(sv[:, None] * A), np.ones((p1, 1))])
        b_ub = sv * b
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * p0 + [(0, 1)])
        if res.success and -res.fun > 1e-9:
            count += 1
    return count


print(f"{'p₀ (輸入維度)':>14} | {'p₁ (神經元數)':>14} | {'理論塊數 r':>12} | "
      f"{'LP 精確計數':>12}")
print("-" * 62)
for p0, p1 in [(1, 3), (2, 3), (2, 5), (3, 5), (2, 10)]:
    r_theory = sum(comb(p1, i) for i in range(p0 + 1))
    rg = np.random.default_rng(p0 * 100 + p1)
    A = rg.standard_normal((p1, p0))
    b = rg.standard_normal(p1)
    exact = count_regions_exact(A, b)
    print(f"{p0:>14} | {p1:>14} | {r_theory:>12} | {exact:>12}")
    check(f"  p₀={p0}, p₁={p1}：精確計數 = 理論值", exact, r_theory)
print("  （啟動模式 = 每個神經元是否被激發；一種可行的模式 = 一塊線性區域）")
print("  注意：隨機取樣只會「漏算」遠處的小區域，所以這裡用 LP 判定可行性才精確")

# 深度 vs 寬度
print("\n深度的威力：合成讓區域數指數成長")
print(f"{'結構':<28} | {'參數量':>8} | {'區域數（實測）':>14}")
print("-" * 56)
for layers, tag in [([2, 6], "1 個隱藏層 (6)"),
                    ([2, 3, 3], "2 個隱藏層 (3,3)"),
                    ([2, 2, 2, 2], "3 個隱藏層 (2,2,2)")]:
    rg = np.random.default_rng(7)
    Ws = [rg.standard_normal((layers[i + 1], layers[i])) for i in range(len(layers) - 1)]
    bs = [rg.standard_normal(layers[i + 1]) for i in range(len(layers) - 1)]
    n_params = sum(W.size for W in Ws) + sum(b.size for b in bs)
    pts = rg.uniform(-3, 3, size=(300000, 2))
    h = pts
    pattern = []
    for W, b in zip(Ws, bs):
        z = h @ W.T + b
        pattern.append(z > 0)
        h = relu(z)
    full = np.hstack(pattern)
    print(f"{tag:<28} | {n_params:>8} | {len(np.unique(full, axis=0)):>14}")
print("  相同參數量下，深度網路切出的區域通常比淺層網路多得多")

# %% [markdown]
# ## 12.2　訓練一個分段線性分類器
#
# 用純 NumPy（加上第 11 章的反向傳播）訓練一個兩隱藏層的 ReLU 網路，
# 把「兩個交錯半月形」分開。重點不是精度，而是看清：
#
# * 決策邊界是由**直線段**拼成的（CPL 的直接後果）
# * 損失由隨機梯度下降下降
# * 參數比資料多時仍能**泛化**到未見資料

# %%
section("12.2 訓練 ReLU 網路：決策邊界由直線段組成")


def make_moons(n, noise=0.2, seed=0):
    """產生兩個交錯的半月形資料（經典的非線性分類測試）。"""
    rg = np.random.default_rng(seed)
    n_half = n // 2
    t1 = rg.uniform(0, np.pi, n_half)
    t2 = rg.uniform(0, np.pi, n - n_half)
    X = np.vstack([np.column_stack([np.cos(t1), np.sin(t1)]),
                   np.column_stack([1 - np.cos(t2), -0.4 - np.sin(t2)])])
    y = np.concatenate([np.zeros(n_half), np.ones(n - n_half)])
    return X + noise * rg.standard_normal(X.shape), y


X_tr, y_tr = make_moons(400, seed=1)
X_te, y_te = make_moons(400, seed=2)

sizes = [2, 16, 16, 1]
rg = np.random.default_rng(0)
Ws = [rg.standard_normal((sizes[i + 1], sizes[i])) * np.sqrt(2.0 / sizes[i])
      for i in range(len(sizes) - 1)]
bs = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
n_params = sum(W.size for W in Ws) + sum(b.size for b in bs)
print(f"網路結構 {sizes}，參數量 = {n_params}，訓練資料 = {len(X_tr)} 筆")


def forward(Ws, bs, X):
    """回傳每一層的 z 與啟動值（最後一層不加 ReLU）。"""
    acts, zs = [X], []
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = h @ W.T + b
        zs.append(z)
        h = relu(z) if i < len(Ws) - 1 else z
        acts.append(h)
    return zs, acts


def loss_and_grads(Ws, bs, X, y):
    """平方損失 + 反向傳播（全部用矩陣運算）。"""
    zs, acts = forward(Ws, bs, X)
    out = acts[-1][:, 0]
    N = len(y)
    L = np.sum((out - y) ** 2) / N
    delta = (2.0 * (out - y) / N).reshape(-1, 1)
    dWs, dbs = [None] * len(Ws), [None] * len(Ws)
    for i in range(len(Ws) - 1, -1, -1):
        dWs[i] = delta.T @ acts[i]
        dbs[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ Ws[i]) * (zs[i - 1] > 0)
    return L, dWs, dbs


accuracy = lambda Ws, bs, X, y: np.mean((forward(Ws, bs, X)[1][-1][:, 0] > 0.5) == (y > 0.5))

# 梯度檢查（務必做！）
L0, dWs, dbs = loss_and_grads(Ws, bs, X_tr, y_tr)
i, j, k = 1, 3, 5
eps = 1e-6
Wp = [W.copy() for W in Ws]
Wp[i][j, k] += eps
Wm = [W.copy() for W in Ws]
Wm[i][j, k] -= eps
num = (loss_and_grads(Wp, bs, X_tr, y_tr)[0] - loss_and_grads(Wm, bs, X_tr, y_tr)[0]) / (2 * eps)
check("反向傳播的梯度 = 數值梯度（梯度檢查）", dWs[i][j, k], num, tol=1e-6)

print(f"\n{'epoch':>7} | {'訓練損失':>12} | {'訓練準確率':>12} | {'測試準確率':>12}")
print("-" * 52)
lr, batch = 0.5, 32
rg2 = np.random.default_rng(3)
for epoch in range(301):
    if epoch % 60 == 0:
        L, _, _ = loss_and_grads(Ws, bs, X_tr, y_tr)
        print(f"{epoch:>7} | {L:>12.6f} | {accuracy(Ws, bs, X_tr, y_tr):>11.1%} "
              f"| {accuracy(Ws, bs, X_te, y_te):>11.1%}")
    idx = rg2.permutation(len(X_tr))
    for s in range(0, len(X_tr), batch):
        bi = idx[s:s + batch]
        _, dW, db = loss_and_grads(Ws, bs, X_tr[bi], y_tr[bi])
        for i in range(len(Ws)):
            Ws[i] -= lr * dW[i]
            bs[i] -= lr * db[i]

check("訓練準確率 > 95%", accuracy(Ws, bs, X_tr, y_tr) > 0.95)
check("測試準確率 > 90%（泛化！參數比資料多仍然有效）",
      accuracy(Ws, bs, X_te, y_te) > 0.90)
print(f"\n參數量 {n_params} 相對 {len(X_tr)} 筆資料並不算少，")
print("但隨機梯度下降選出的權重仍然泛化良好 —— 這正是深度學習的核心謎題")

# 決策邊界與線性區域
gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 400), np.linspace(-1.6, 1.8, 400))
grid = np.column_stack([gx.ravel(), gy.ravel()])
zs_g, acts_g = forward(Ws, bs, grid)
out_g = acts_g[-1][:, 0].reshape(gx.shape)
patterns = np.hstack([z > 0 for z in zs_g[:-1]])
n_regions = len(np.unique(patterns, axis=0))
print(f"\n訓練後的網路在這塊區域切出 {n_regions} 個線性區域")
check("線性區域數 > 1（確實是分段線性）", n_regions > 1)

# 在單一區域內，網路確實是線性的
region_id = np.unique(patterns, axis=0)[n_regions // 2]
mask = np.all(patterns == region_id, axis=1)
if mask.sum() > 50:
    pts_r = grid[mask][:50]
    vals_r = out_g.ravel()[mask][:50]
    Afit = np.column_stack([np.ones(len(pts_r)), pts_r])
    cfit, *_ = np.linalg.lstsq(Afit, vals_r, rcond=None)
    check("在同一個線性區域內，網路輸出完全是線性的",
          Afit @ cfit, vals_r, tol=1e-8)

fig = plt.figure(figsize=(10.8, 4.2))
ax = fig.add_subplot(1, 2, 1)
ax.contourf(gx, gy, out_g, levels=[-10, 0.5, 10], colors=["#cfe3f7", "#f7d5cf"], alpha=0.9)
ax.contour(gx, gy, out_g, levels=[0.5], colors="k", linewidths=1.6)
ax.scatter(*X_tr[y_tr == 0].T, s=9, c="C0", label="class 0")
ax.scatter(*X_tr[y_tr == 1].T, s=9, c="C3", label="class 1")
ax.set_title("Decision boundary is piecewise linear", fontsize=9)
ax.legend(fontsize=7)
ax.set_aspect("equal")

ax = fig.add_subplot(1, 2, 2)
codes = np.unique(patterns, axis=0, return_inverse=True)[1].reshape(gx.shape)
ax.imshow(codes % 20, origin="lower", cmap="tab20",
          extent=[-1.8, 2.8, -1.6, 1.8], aspect="equal", interpolation="nearest")
ax.contour(gx, gy, out_g, levels=[0.5], colors="k", linewidths=1.6)
ax.set_title(f"{n_regions} linear regions (ReLU activation patterns)", fontsize=9)
finish(fig, "ch12_relu_regions")

# %% [markdown]
# ## 12.3　平均、變異數與共變異數矩陣
#
# | | 樣本（已發生）| 期望（機率）|
# |---|---|---|
# | 平均 | $m=\frac1N\sum x_i$ | $\mathbb E[x]=\sum p_ix_i$ |
# | 變異數 | $S^2=\frac{1}{N-1}\sum(x_i-m)^2$ | $\sigma^2=\sum p_i(x_i-m)^2$ |
#
# 樣本變異數除以 $N-1$（不是 $N$）：一個自由度已經被樣本平均用掉了。
#
# 恆等式：$\sigma^2=\mathbb E[x^2]-(\mathbb E[x])^2$。
#
# 多變數時，變異數升級成**共變異數矩陣**
#
# $$V_{ij}=\mathbb E\left[(x_i-m_i)(x_j-m_j)\right]$$
#
# 三個必記的性質：
#
# 1. $V$ **對稱**
# 2. $V$ **半正定**（$\mathbf c^{\mathsf T}V\mathbf c=\operatorname{Var}(\mathbf c^{\mathsf T}\mathbf x)\ge0$）
# 3. $\operatorname{Cov}(A\mathbf x)=A\,V\,A^{\mathsf T}$ ← 線性代數與統計的接口

# %%
section("12.3 樣本統計 vs 期望值")

ages = np.array([18.0, 17.0, 18.0, 19.0, 17.0])
m_sample = ages.mean()
S2 = np.sum((ages - m_sample) ** 2) / (len(ages) - 1)
print(f"樣本 {ages}：平均 m = {m_sample}，樣本變異數 S² = {S2}")
check("樣本平均 = 17.8", m_sample, 17.8)
check("樣本變異數 = 0.7（除以 N−1）", S2, 0.7)
check("與 numpy 的 ddof=1 一致", S2, float(np.var(ages, ddof=1)))

x_vals = np.array([17.0, 18.0, 19.0])
p_vals = np.array([0.2, 0.5, 0.3])
m_exp = p_vals @ x_vals
var_exp = p_vals @ (x_vals - m_exp) ** 2
print(f"\n機率 {p_vals} 下：期望 E[x] = {m_exp}，變異數 σ² = {var_exp:.4f}，σ = {np.sqrt(var_exp):.4f}")
check("E[x] = 18.1", m_exp, 18.1)
check("σ² = 0.49", var_exp, 0.49)
check("σ² = E[x²] − (E[x])²", var_exp, p_vals @ x_vals ** 2 - m_exp ** 2)

# 大數法則：樣本平均 → 期望值
print("\n大數法則（投擲公正硬幣）：")
rg = np.random.default_rng(0)
for N in [10, 100, 10000, 1000000]:
    flips = rg.integers(0, 2, N)
    print(f"  N = {N:>8}：樣本平均 = {flips.mean():.6f}（期望 0.5，"
          f"誤差 {abs(flips.mean() - 0.5):.6f} ≈ σ/√N = {0.5 / np.sqrt(N):.6f}）")
check("N 很大時樣本平均 → 0.5", abs(rg.integers(0, 2, 1000000).mean() - 0.5) < 0.005)

# %%
section("12.3 共變異數矩陣的三個性質")

# Strang 的年齡-身高例子
A_data = np.array([[3.0, -4.0, 7.0, 1.0, -4.0, -3.0],
                   [7.0, -6.0, 8.0, -1.0, -1.0, -7.0]])     # 已置中，2 列 x 6 樣本
n_samp = A_data.shape[1]
V = A_data @ A_data.T / (n_samp - 1)
show_matrix("共變異數矩陣 V = AAᵀ/(n−1)", V)
print(f"  對角線 = 變異數 {np.diag(V)}")
print(f"  非對角 = 共變異數 {V[0, 1]:.1f} > 0 ⇒ 年齡大身高也高")
check("① V 對稱", is_symmetric(V))
check("② V 半正定（λ ≥ 0）", np.all(np.linalg.eigvalsh(V) > -1e-12))
check("   V 正定（此例資料不退化）", is_positive_definite(V))
rg = np.random.default_rng(0)
ok = True
for _ in range(500):
    c = rg.standard_normal(2)
    var_proj = np.var(c @ A_data, ddof=1)
    ok = ok and abs(c @ V @ c - var_proj) < 1e-9
check("   cᵀVc = Var(cᵀx) ≥ 0（半正定的統計意義）", ok)

# ③ Cov(Ax) = A V Aᵀ
M = np.array([[2.0, 1.0], [-1.0, 3.0], [0.5, 0.0]])
Y = M @ A_data
V_Y = Y @ Y.T / (n_samp - 1)
check("③ Cov(Mx) = M·V·Mᵀ", V_Y, M @ V @ M.T)
print("  ⇒ 任何線性組合的共變異數都能用矩陣乘法算出（統計與線性代數的接口）")

# 不相關 ⟺ 共變異數矩陣對角 ⟺ PCA 的目標
lam, Q = np.linalg.eigh(V)
Z = Q.T @ A_data                      # 轉到主成分座標
V_Z = Z @ Z.T / (n_samp - 1)
show_matrix("主成分座標下的共變異數（對角！）", V_Z)
check("PCA 把共變異數矩陣對角化 ⇒ 成分互不相關", V_Z, np.diag(np.diag(V_Z)), tol=1e-10)
check("對角線 = V 的特徵值", np.sort(np.diag(V_Z)), np.sort(lam))
check("總變異數 trace 不變（正交變換）", np.trace(V), np.trace(V_Z))
print(f"  總變異數 = trace V = {np.trace(V):.1f} = Σλ = {np.sum(lam):.1f}")

# 白化（whitening）：V^{-1/2} 把資料變成單位共變異數
W_white = Q @ np.diag(1 / np.sqrt(lam)) @ Q.T
X_white = W_white @ A_data
check("白化後共變異數 = I", X_white @ X_white.T / (n_samp - 1), np.eye(2), tol=1e-10)
print("  白化矩陣 V^{−1/2} 讓所有方向的變異數都變成 1（馬氏距離的基礎）")

# %% [markdown]
# ### 多維常態分布與 Cholesky
#
# 要產生指定共變異數 $V$ 的隨機向量：取 $V=R^{\mathsf T}R$（Cholesky），
# 則 $\mathbf x=R^{\mathsf T}\mathbf z$（$\mathbf z$ 為獨立標準常態）滿足
#
# $$\operatorname{Cov}(\mathbf x)=R^{\mathsf T}\,I\,R=V$$
#
# 機率密度的等高線是橢圓
# $(\mathbf x-\mathbf m)^{\mathsf T}V^{-1}(\mathbf x-\mathbf m)=c$ ──
# 主軸沿 $V$ 的特徵向量，半軸長 $\propto\sqrt{\lambda_i}$（正是第 7 章的能量橢圓）。

# %%
section("12.3 用 Cholesky 產生指定共變異數的資料")

V_target = np.array([[4.0, 2.4], [2.4, 2.25]])
R = cholesky(V_target)
print(f"目標共變異數 V =\n{V_target}")
print(f"Cholesky R =\n{np.round(R, 6)}")
check("V = RᵀR", R.T @ R, V_target)

rg = np.random.default_rng(7)
Z = rg.standard_normal((2, 200000))
X_gen = R.T @ Z
V_emp = np.cov(X_gen)
print(f"\n200000 筆樣本的實測共變異數 =\n{np.round(V_emp, 4)}")
check("實測共變異數 ≈ 目標（相對誤差 < 2%）",
      np.linalg.norm(V_emp - V_target) / np.linalg.norm(V_target) < 0.02)
corr = V_target[0, 1] / np.sqrt(V_target[0, 0] * V_target[1, 1])
print(f"相關係數 ρ = {corr:.4f}")
check("|ρ| ≤ 1（Schwarz 不等式的統計版）", abs(corr) <= 1.0)

# PCA 找回主軸
res = pca(X_gen.T, n_components=2)
lam_t, Q_t = np.linalg.eigh(V_target)
print(f"PCA 的解釋變異 = {np.round(res['explained_variance'], 4)}")
print(f"V 的特徵值     = {np.round(lam_t[::-1], 4)}")
check("PCA 找回 V 的特徵值", np.sort(res["explained_variance"]), np.sort(lam_t),
      tol=0.05)

# 中央極限定理
section("12.3 中央極限定理：任何分布的平均都趨向常態")

rg = np.random.default_rng(1)
print(f"{'原始分布':<16} | {'N':>6} | {'樣本平均的 skew':>16} | {'kurtosis − 3':>14}")
print("-" * 60)
for name, sampler in [("均勻 U(0,1)", lambda n: rg.uniform(0, 1, n)),
                      ("指數 Exp(1)", lambda n: rg.exponential(1.0, n)),
                      ("Bernoulli(0.2)", lambda n: (rg.random(n) < 0.2).astype(float))]:
    for N in [1, 30]:
        means = np.array([sampler(N).mean() for _ in range(40000)])
        z = (means - means.mean()) / means.std()
        skew = np.mean(z ** 3)
        kurt = np.mean(z ** 4) - 3
        print(f"{name:<16} | {N:>6} | {skew:>16.4f} | {kurt:>14.4f}")
    check(f"  {name}：N = 30 的 skew 已縮小到 N = 1 的 1/4 以下", abs(skew) < 0.6)
print("  常態分布的 skew = 0、kurtosis − 3 = 0 ⇒ N 增大時樣本平均趨近常態")
print("  理論：樣本平均的偏態 = (原分布偏態)/√N，例如指數分布 2/√30 = "
      f"{2 / np.sqrt(30):.4f}")
rg_s = np.random.default_rng(42)
for N in [1, 10, 30, 100]:
    means = np.array([rg_s.exponential(1.0, N).mean() for _ in range(40000)])
    z = (means - means.mean()) / means.std()
    print(f"    指數分布 N = {N:>3}：實測 skew = {np.mean(z ** 3):>7.4f}，"
          f"理論 2/√N = {2 / np.sqrt(N):>7.4f}")
check("偏態按 1/√N 衰減（N = 100 時 < 0.3）", abs(np.mean(
    ((lambda mm: (mm - mm.mean()) / mm.std())(
        np.array([rg_s.exponential(1.0, 100).mean() for _ in range(40000)]))) ** 3)) < 0.3)

fig = plt.figure(figsize=(10.8, 3.6))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_gen[0, :3000], X_gen[1, :3000], s=4, alpha=0.3)
th = np.linspace(0, 2 * np.pi, 200)
for c in [1.0, 2.0, 3.0]:
    ell = Q_t @ np.diag(np.sqrt(lam_t)) @ np.array([np.cos(th), np.sin(th)]) * c
    ax.plot(ell[0], ell[1], "C3", lw=1.3)
for k in range(2):
    ax.annotate("", xy=Q_t[:, k] * np.sqrt(lam_t[k]) * 2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
ax.set_title("Gaussian data: contours are ellipses of V", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
for N, c in [(1, "C0"), (2, "C1"), (5, "C2"), (30, "C3")]:
    means = np.array([rg.exponential(1.0, N).mean() for _ in range(60000)])
    z = (means - means.mean()) / means.std()
    ax.hist(z, bins=90, density=True, histtype="step", color=c, lw=1.4, label=f"N = {N}")
grid_z = np.linspace(-4, 4, 300)
ax.plot(grid_z, np.exp(-grid_z ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.4,
        label="standard normal")
ax.set_title("Central limit theorem (exponential samples)", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
finish(fig, "ch12_covariance_clt")

# %% [markdown]
# ## 動手練習
#
# 1. 手算一組權重，讓兩層 ReLU 網路實作 $|x|$（提示：$|x|=\text{ReLU}(x)+\text{ReLU}(-x)$）。
# 2. 驗證 $r(p_1,p_0)=\sum_{i=0}^{p_0}\binom{p_1}{i}$：在 $\mathbb R^2$ 中隨機畫 $p_1$ 條直線，數區域。
# 3. 證明 $\operatorname{Cov}(A\mathbf x)=A\,V\,A^{\mathsf T}$，並用它算出
#    $\operatorname{Var}(x_1+x_2)=V_{11}+2V_{12}+V_{22}$。
# 4. 用 $V^{-1/2}$ 白化資料後計算馬氏距離
#    $d^2=(\mathbf x-\mathbf m)^{\mathsf T}V^{-1}(\mathbf x-\mathbf m)$，
#    確認它服從卡方分布。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1：|x|
A1_abs = np.array([[1.0], [-1.0]])
b1_abs = np.array([0.0, 0.0])
A2_abs = np.array([[1.0, 1.0]])
F_abs = lambda x: (A2_abs @ relu(A1_abs @ np.array([x]) + b1_abs)).item()
xs = np.linspace(-3, 3, 13)
print(f"練習 1：網路輸出 = {np.round([F_abs(x) for x in xs], 4)}")
print(f"        |x|      = {np.round(np.abs(xs), 4)}")
check("  兩層 ReLU 網路精確實作 |x|",
      np.array([F_abs(x) for x in xs]), np.abs(xs))

# 練習 2
print("\n練習 2：R² 中 p₁ 條直線切出的區域數")
for p1 in [1, 2, 3, 5, 8]:
    theory = 1 + p1 + comb(p1, 2)
    rg3 = np.random.default_rng(p1)
    A = rg3.standard_normal((p1, 2))
    b = rg3.standard_normal(p1)
    exact = count_regions_exact(A, b)
    print(f"  p₁ = {p1}：理論 1 + p₁ + C(p₁,2) = {theory}，LP 精確計數 {exact}")
    check(f"  p₁ = {p1} 相符", exact, theory)

# 練習 3
print("\n練習 3：")
c_sum = np.array([1.0, 1.0])
check("  Var(x₁+x₂) = V₁₁ + 2V₁₂ + V₂₂",
      c_sum @ V @ c_sum, V[0, 0] + 2 * V[0, 1] + V[1, 1])
check("  = 直接計算 Var(x₁+x₂)", c_sum @ V @ c_sum,
      float(np.var(A_data.sum(axis=0), ddof=1)))
print(f"  Var(x₁+x₂) = {c_sum @ V @ c_sum:.4f}"
      f"（若獨立則只會是 {V[0,0] + V[1,1]:.4f}，差額就是 2V₁₂ = {2*V[0,1]:.4f}）")

# 練習 4：馬氏距離 ~ 卡方
print("\n練習 4：馬氏距離的分布")
rg4 = np.random.default_rng(11)
V4 = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, -0.4], [0.5, -0.4, 1.5]])
R4 = cholesky(V4)
X4 = (R4.T @ rg4.standard_normal((3, 200000)))
Vinv = np.linalg.inv(V4)
d2 = np.einsum("ij,jk,ik->i", X4.T, Vinv, X4.T)
print(f"  馬氏距離平方的平均 = {d2.mean():.4f}（卡方(3) 的期望 = 3）")
print(f"  變異數 = {d2.var():.4f}（卡方(3) 的變異數 = 6）")
check("  E[d²] = 自由度 3", d2.mean(), 3.0, tol=0.05)
check("  Var[d²] = 2·自由度 = 6", d2.var(), 6.0, tol=0.3)
W4 = np.linalg.inv(R4.T)
check("  白化後共變異數 = I", np.cov(W4 @ X4), np.eye(3), tol=0.02)

# %% [markdown]
# ## 本章重點回顧
#
# * 線性函數對真實資料太弱（XOR 就打敗它）；
#   **連續分段線性**（ReLU 的合成鏈）既有足夠的非線性又能計算。
# * $p_1$ 個 ReLU 神經元把 $\mathbb R^{p_0}$ 切成
#   $\sum_{i\le p_0}\binom{p_1}{i}$ 塊，每塊上網路是**嚴格線性**的；
#   深度讓區域數指數成長。
# * 訓練 = 隨機梯度下降 + 反向傳播；決策邊界就是由直線段拼成的。
#   參數多於資料仍能泛化，是深度學習的核心謎題。
# * 樣本變異數除以 $N-1$；$\sigma^2=\mathbb E[x^2]-(\mathbb E[x])^2$。
# * **共變異數矩陣** $V$ 對稱、半正定，且
#   $\operatorname{Cov}(A\mathbf x)=AVA^{\mathsf T}$ ── 統計與線性代數的接口。
#   $\mathbf c^{\mathsf T}V\mathbf c=\operatorname{Var}(\mathbf c^{\mathsf T}\mathbf x)\ge0$
#   就是半正定的統計意義。
# * PCA 把 $V$ 對角化（成分互不相關）；$V^{-1/2}$ 做白化；
#   Cholesky $V=R^{\mathsf T}R$ 可產生指定共變異數的資料；
#   機率密度等高線就是第 7 章的能量橢圓。
# * 中央極限定理讓常態分布無所不在，而多維常態完全由 $\mathbf m$ 與 $V$ 決定。
#
# 下一章：Strang 附錄精選 ── 秩的不等式、五大分解、Markov 與 Perron–Frobenius、
# 張量與計算機圖學。
