# %% [markdown]
# # 第 1 章　向量、矩陣與欄空間
#
# > 對應 Strang《Introduction to Linear Algebra》6/e 第 1 章
# > （1.1 線性組合、1.2 點積與角度、1.3 矩陣與欄空間、1.4 矩陣乘法 AB 與 CR）
# > 以及 Riley《Mathematical Methods for Physics and Engineering》7.1–7.6、8.1–8.4。
#
# 這一章的四個主角：
#
# | 概念 | 符號 | 一句話 |
# |---|---|---|
# | 線性組合 | $c\mathbf v + d\mathbf w$ | 線性代數唯一的基本動作 |
# | 點積 | $\mathbf v\cdot\mathbf w$ | 把幾何（長度、角度）帶進代數 |
# | 欄空間 | $C(A)$ | 所有 $A\mathbf x$ 的集合 |
# | 秩與 $A=CR$ | $r$ | 矩陣裡真正獨立的資訊量 |
#
# **執行方式**
#
# ```bash
# python chapters/ch01_vectors_and_matrices.py      # 當成一般程式跑
# python tools/build_notebooks.py ch01              # 產生對應的 notebook
# ```

# %%
# --- 讓本檔案不論從哪個目錄執行、或在 notebook 裡執行，都能 import linalg_tutorial ---
import pathlib
import sys

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.subspaces import cr_factor, column_space, rank
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import draw_vector, finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1.1　向量與線性組合
#
# 微積分從數字 $x$ 與函數 $f(x)$ 開始；線性代數從向量 $\mathbf v,\mathbf w$ 與它們的
# **線性組合** 開始。只有兩個基本動作：
#
# $$\text{純量乘法：}\quad c\mathbf v,\qquad\qquad \text{加法：}\quad \mathbf v+\mathbf w$$
#
# 合起來就是線性組合
#
# $$c\mathbf v + d\mathbf w \qquad (c, d \text{ 是任意實數})$$
#
# Strang 在第 1 章一開頭就問兩個問題，整本書都在回答它們：
#
# 1. **所有** 的 $c\mathbf v + d\mathbf w$ 長什麼樣子？是一條直線、一個平面，還是整個空間？
# 2. 要得到某個指定的 $\mathbf b$，該取哪一組 $c, d$？（這就是解 $A\mathbf x=\mathbf b$）

# %%
section("1.1 線性組合：兩個基本動作")

v = np.array([2.0, 1.0])
w = np.array([-1.0, 2.0])

show_matrix("v", v)
show_matrix("w", w)
print(f"\n3v        = {3 * v}          (純量乘法：方向不變、長度變 3 倍)")
print(f"v + w     = {v + w}          (加法：平行四邊形法則)")
print(f"2v - 3w   = {2 * v - 3 * w}          (線性組合 c=2, d=-3)")
print(f"0v + 0w   = {0 * v + 0 * w}          (零向量永遠在裡面)")

# numpy 的向量運算本身就是「逐元素」的線性運算
check("c(v+w) = cv + cw（分配律）", 5 * (v + w), 5 * v + 5 * w)
check("(c+d)v = cv + dv", (2 + 7) * v, 2 * v + 7 * v)

# %% [markdown]
# ### 線性組合填出什麼形狀？
#
# 若 $\mathbf v,\mathbf w$ 在 $\mathbb R^2$ 中**不共線**，則 $c\mathbf v+d\mathbf w$ 填滿整個平面；
# 若共線（$\mathbf w = k\mathbf v$，稱為**線性相依**），所有組合只落在一條直線上。
#
# 下面用大量隨機的 $(c,d)$ 把這兩種情形畫出來──這是「欄空間」概念的第一張圖。

# %%
rng = np.random.default_rng(0)
cd = rng.uniform(-2, 2, size=(400, 2))

fig, axes = new_axes("", figsize=(9.6, 4.6), equal=False)
fig.clf()
for k, (a, b, name) in enumerate([
    (v, w, "independent: combinations fill the plane"),
    (v, 3 * v, "dependent (w = 3v): combinations fill a line"),
]):
    ax = fig.add_subplot(1, 2, k + 1)
    pts = cd @ np.vstack([a, b])          # 每一列都是 c*a + d*b
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.35, color="C7")
    draw_vector(ax, a, "v", "C0")
    draw_vector(ax, b, "w", "C3")
    ax.set_title(name, fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.grid(alpha=0.3)
finish(fig, "ch01_linear_combinations")

# %% [markdown]
# ### 解兩個方程：消去法的雛形
#
# 要找 $c, d$ 使 $c\mathbf v + d\mathbf w = \mathbf b$，寫成三種等價的樣貌
# （Strang 稱為 column way / row way / matrix way）：
#
# $$c\begin{bmatrix}2\\1\end{bmatrix}+d\begin{bmatrix}-1\\2\end{bmatrix}
# =\begin{bmatrix}3\\8\end{bmatrix}
# \iff
# \begin{aligned}2c-d&=3\\c+2d&=8\end{aligned}
# \iff
# \begin{bmatrix}2&-1\\1&2\end{bmatrix}\begin{bmatrix}c\\d\end{bmatrix}
# =\begin{bmatrix}3\\8\end{bmatrix}$$
#
# 手算的規則很簡單：**把某列乘上一個數，再從另一列減掉**，目標是製造 0。

# %%
section("1.1 手算消去 vs 矩陣解法")

A = np.column_stack([v, w])              # 把 v, w 當成 A 的兩個欄
b = np.array([3.0, 8.0])
show_matrix("A = [v w]", A)

# 手算：第 2 列 - (1/2) x 第 1 列，消掉 c
m = A[1, 0] / A[0, 0]
row2 = A[1, :] - m * A[0, :]
rhs2 = b[1] - m * b[0]
print(f"\n消去乘數 m = {m}")
print(f"新的第 2 列：{row2} | 右側 {rhs2:.4g}   →   d = {rhs2 / row2[1]:.4g}")
d_sol = rhs2 / row2[1]
c_sol = (b[0] - A[0, 1] * d_sol) / A[0, 0]          # 回代
print(f"回代得 c = {c_sol:.4g}")

x_hand = np.array([c_sol, d_sol])
check("手算解滿足 Ax = b", A @ x_hand, b)
check("與 numpy.linalg.solve 一致", x_hand, np.linalg.solve(A, b))

# 相依的情形：無解或無窮多解
A_dep = np.column_stack([v, 3 * v])
print("\n相依矩陣 A_dep = [v  3v]：")
print(f"  rank(A_dep) = {rank(A_dep)}（只有 1 個獨立欄）")
print("  b = (3, 8) 不在那條直線上 → 方程無解")
check("b 不是 v 的倍數", not np.allclose(np.cross(np.append(v, 0), np.append(b, 0)), 0))

# %% [markdown]
# ## 1.2　點積：長度與角度
#
# $$\mathbf v\cdot\mathbf w = v_1w_1+\dots+v_nw_n = \mathbf v^{\mathsf T}\mathbf w$$
#
# 點積一口氣給出三件事：
#
# * **長度**：$\|\mathbf v\|^2=\mathbf v\cdot\mathbf v$（$n$ 維的畢氏定理）
# * **垂直**：$\mathbf v\cdot\mathbf w=0 \iff$ 夾角 $90^\circ$
# * **角度**：$\displaystyle\cos\theta=\frac{\mathbf v\cdot\mathbf w}{\|\mathbf v\|\,\|\mathbf w\|}$
#
# 由 $|\cos\theta|\le 1$ 立刻得到兩個最重要的不等式：
#
# $$\text{Schwarz：}\ |\mathbf v\cdot\mathbf w|\le\|\mathbf v\|\|\mathbf w\|,
# \qquad
# \text{三角：}\ \|\mathbf v+\mathbf w\|\le\|\mathbf v\|+\|\mathbf w\|$$

# %%
section("1.2 點積、長度、角度")


def dot(a, b):
    """點積的教學版實作：逐項相乘再相加。"""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape:
        raise ValueError("只有同維度的向量才能做點積")
    return float(sum(ai * bi for ai, bi in zip(a, b)))


def norm(a):
    """長度 = 根號下自己跟自己的點積。"""
    return float(np.sqrt(dot(a, a)))


def angle_deg(a, b):
    cos = dot(a, b) / (norm(a) * norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


p, q = np.array([1.0, 3.0, 2.0]), np.array([4.0, -4.0, 4.0])
print(f"p·q        = {dot(p, q):.4g}  → 垂直！")
print(f"‖p‖        = {norm(p):.6f}  (= √14)")
print(f"夾角        = {angle_deg(p, q):.2f}°")

check("自製 dot 與 numpy 一致", dot(p, q), float(p @ q))
check("自製 norm 與 numpy 一致", norm(p), float(np.linalg.norm(p)))
check("Schwarz 不等式", abs(dot(p, q)) <= norm(p) * norm(q) + 1e-12)
check("三角不等式", norm(p + q) <= norm(p) + norm(q) + 1e-12)
check("p·q = 0 ⇒ ‖p+q‖² = ‖p‖² + ‖q‖²（畢氏）",
      norm(p + q) ** 2, norm(p) ** 2 + norm(q) ** 2)

u = np.array([1.0, 1.0]) / np.sqrt(2)
print(f"\n單位向量 u = {u}，‖u‖ = {norm(u):.1f}；與 x 軸夾角 {angle_deg(u, [1, 0]):.0f}°")

# %% [markdown]
# ### 高維度的意外：隨機向量幾乎都互相垂直
#
# 點積不只是計算工具，它還揭露高維幾何的直覺陷阱。在 $\mathbb R^n$ 中隨機取兩個向量，
# $\cos\theta$ 的典型大小是 $O(1/\sqrt n)$ ──維度越高，兩個向量越接近垂直。
# 這個現象是第 7 章 SVD、第 10 章機器學習裡「高維資料為何稀疏」的根源。

# %%
section("1.2 高維度：隨機向量的夾角")

rng = np.random.default_rng(1)
print(f"{'維度 n':>8} | {'平均 |cosθ|':>12} | {'平均夾角':>10}")
print("-" * 38)
dims, means = [], []
for n in [2, 3, 10, 100, 1000, 10000]:
    X = rng.standard_normal((500, n))
    Y = rng.standard_normal((500, n))
    cos = np.einsum("ij,ij->i", X, Y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    dims.append(n)
    means.append(np.mean(np.abs(cos)))
    print(f"{n:>8} | {means[-1]:>12.4f} | {np.degrees(np.arccos(np.mean(np.abs(cos)))):>9.2f}°")

check("平均 |cosθ| 大致與 1/√n 同階",
      abs(means[-1] * np.sqrt(dims[-1]) - means[0] * np.sqrt(dims[0])) < 1.0)

fig, ax = new_axes("Random vectors become orthogonal in high dimensions",
                   figsize=(5.6, 4.0), equal=False)
ax.loglog(dims, means, "o-", label="mean |cos θ| (measured)")
ax.loglog(dims, [np.sqrt(2 / (np.pi * n)) for n in dims], "--", label=r"$\sqrt{2/\pi n}$ (theory)")
ax.set_xlabel("dimension n")
ax.set_ylabel(r"mean $|\cos\theta|$")
ax.legend()
finish(fig, "ch01_high_dim_angles")

# %% [markdown]
# ## 1.3　矩陣與它的欄空間
#
# 一個 $m\times n$ 矩陣就是把 $n$ 個 $m$ 維欄向量並排：
#
# $$A=\begin{bmatrix}\mathbf a_1&\mathbf a_2&\cdots&\mathbf a_n\end{bmatrix}$$
#
# $A\mathbf x$ 有兩種讀法，**同樣的乘法、不同的理解層次**：
#
# * **列觀點**：$(A\mathbf x)_i = (\text{第 } i \text{ 列})\cdot\mathbf x$ ──適合手算
# * **欄觀點**：$A\mathbf x = x_1\mathbf a_1+\dots+x_n\mathbf a_n$ ──適合理解
#
# 所有 $A\mathbf x$ 構成的集合叫 **欄空間** $C(A)$：
#
# $$C(A)=\{A\mathbf x : \mathbf x\in\mathbb R^n\}=\text{span}\{\mathbf a_1,\dots,\mathbf a_n\}$$

# %%
section("1.3 Ax 的列觀點與欄觀點")

# Strang 愛用的差分矩陣：(Ax)_k = x_k - x_{k-1}，離散版的微分
A = np.array([
    [-1.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 1.0],
])
x = np.array([1.0, 4.0, 9.0, 16.0])          # 平方數
show_matrix("A（3x4 差分矩陣）", A)

row_view = np.array([A[i, :] @ x for i in range(A.shape[0])])
col_view = sum(x[j] * A[:, j] for j in range(A.shape[1]))

print(f"\n列觀點（3 個點積）     Ax = {row_view}")
print(f"欄觀點（4 欄的組合）   Ax = {col_view}")
print(f"numpy                 Ax = {A @ x}")
print("→ 差分矩陣把 (1,4,9,16) 變成相鄰差 (3,5,7)：正是導數 2x+1 的離散版")
check("列觀點 = 欄觀點", row_view, col_view)
check("兩者都等於 A @ x", row_view, A @ x)
print(f"\n小乘法次數 = m·n = {A.shape[0] * A.shape[1]}（兩種觀點完全一樣，只是順序不同）")

# %% [markdown]
# ### 獨立、相依與欄空間的四種可能
#
# 由左到右檢查每一欄：**它能不能由前面的欄組合出來？**
# 不能 → 貢獻新方向（獨立）；能 → 不帶來新東西（相依）。
#
# 在 $\mathbb R^3$ 裡，$C(A)$ 只有四種可能：整個 $\mathbb R^3$、一個過原點的平面、
# 一條過原點的直線、或只有原點。

# %%
section("1.3 欄空間 C(A) 的四種可能")

examples = {
    "3 個獨立欄 → C(A) = R³": np.array([[1, 0, 0], [2, 4, 0], [3, 5, 6]], float),
    "欄1+欄2=欄3 → C(A) 是平面": np.array([[1, 2, 3], [1, 4, 5], [6, 0, 6]], float),
    "每欄都是 (1,2,5) 的倍數 → C(A) 是直線": np.array([[1, 3, 4], [2, 6, 8], [5, 15, 20]], float),
    "零矩陣 → C(A) 只有原點": np.zeros((3, 3)),
}
for name, M in examples.items():
    r = rank(M)
    shapes = {3: "整個 R³", 2: "平面", 1: "直線", 0: "單點 {0}"}
    print(f"  rank = {r}  →  C(A) = {shapes[r]:<10}  |  {name}")

# 秩一矩陣：欄都在一條線上，列也都在一條線上
A1 = np.array([[1, 2, 10, 100], [3, 6, 30, 300], [2, 4, 20, 200]], float)
col, row = A1[:, [0]], A1[[0], :]
print("\n秩一矩陣 A = (欄向量)(列向量)：")
show_matrix("A", A1)
check("A = (1,3,2)ᵀ (1,2,10,100)", A1, col @ row)
check("rank(A) = 1", rank(A1) == 1)
print("  ⇒ 欄空間是直線、列空間也是直線：『列秩 = 欄秩』最簡單的情形")

# %% [markdown]
# ## 1.4　矩陣乘法 AB 的四種算法
#
# 同樣的 $mnp$ 次乘法，有四種排列順序。四種都要會讀：
#
# | 觀點 | 公式 | 一次得到 |
# |---|---|---|
# | 點積 | $(AB)_{ij}=(\text{列}_i A)\cdot(\text{欄}_j B)$ | 一個數字 |
# | 欄 | $AB$ 的第 $j$ 欄 $=A\mathbf b_j$ | 一整欄 |
# | 列 | $AB$ 的第 $i$ 列 $=(\text{列}_i A)B$ | 一整列 |
# | 外積 | $AB=\sum_k \mathbf a_k \mathbf b_k^{\mathsf T}$ | 一個秩一矩陣 |
#
# 最後一種（欄 × 列）最少人用，卻最強大：它說明 **每個秩 $r$ 矩陣都是 $r$ 個秩一矩陣的和**，
# 直通第 7 章的 SVD。

# %%
section("1.4 AB 的四種算法給同一個答案")

A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])     # 3x2
B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])    # 2x3
m, n = A.shape
n2, p = B.shape

way_dot = np.array([[A[i, :] @ B[:, j] for j in range(p)] for i in range(m)])
way_col = np.column_stack([A @ B[:, j] for j in range(p)])
way_row = np.vstack([A[i, :] @ B for i in range(m)])
way_outer = sum(np.outer(A[:, k], B[k, :]) for k in range(n))

show_matrix("AB", A @ B)
check("① 點積觀點", way_dot, A @ B)
check("② 欄觀點 A·bⱼ", way_col, A @ B)
check("③ 列觀點 aᵢᵀ·B", way_row, A @ B)
check("④ 外積和 Σ aₖbₖᵀ", way_outer, A @ B)
print(f"\n四種方式都用掉 m·n·p = {m}·{n}·{p} = {m * n * p} 次小乘法")

print("\n外積展開（每一項都是秩一矩陣）：")
for k in range(n):
    piece = np.outer(A[:, k], B[k, :])
    print(f"  第 {k + 1} 項 rank = {rank(piece)}，貢獻\n{piece}")
check("rank(AB) ≤ min(rank A, rank B)", rank(A @ B) <= min(rank(A), rank(B)))

# 不可交換，但結合律成立
P = np.array([[0.0, 1.0], [1.0, 0.0]])        # 交換矩陣
S = np.array([[1.0, 2.0], [3.0, 4.0]])
print("\n左乘 = 列運算，右乘 = 欄運算：")
show_matrix("P·S（換列）", P @ S)
show_matrix("S·P（換欄）", S @ P)
check("一般而言 PS ≠ SP", not np.allclose(P @ S, S @ P))
check("結合律 (PS)P = P(SP)", (P @ S) @ P, P @ (S @ P))

# %% [markdown]
# ### $A = CR$：把矩陣拆成「獨立欄」與「組合說明書」
#
# * $C$：由左到右挑出 $A$ 的前 $r$ 個**獨立欄**（$m\times r$）
# * $R$：每一欄說明「如何用 $C$ 的欄組出 $A$ 的那一欄」（$r\times n$），形如 $[\,I\ \ F\,]$ 的重排
#
# 於是
#
# $$A = CR, \qquad r = \operatorname{rank}(A)$$
#
# 這個分解立刻給出線性代數的**第一個大定理**：
#
# > **列秩 = 欄秩**
#
# 理由只有四句話：$C$ 的 $r$ 欄獨立；$A$ 的每一欄都是它們的組合（$A=CR$）；
# $R$ 的 $r$ 列獨立（因為裡面有 $I$）；$A$ 的每一列都是 $R$ 的列的組合（同一式子按列讀）。

# %%
section("1.4 A = CR 與『列秩 = 欄秩』")

A = np.array([
    [1.0, 3.0, 4.0],
    [2.0, 4.0, 2.0],
    [3.0, 7.0, 6.0],
])
C, R = cr_factor(A)
show_matrix("A", A)
show_matrix("C（A 的獨立欄）", C)
show_matrix("R（組合說明書，含單位矩陣）", R)

check("A = C·R", C @ R, A)
check("rank(A) = C 的欄數", rank(A) == C.shape[1])
check("C 的欄 = C(A) 的基底", np.allclose(column_space(A), C))
print(f"\nA 是 {A.shape[0]}x{A.shape[1]}，但真正的資訊量只有 r = {R.shape[0]}：")
print(f"  C 是 {C.shape[0]}x{C.shape[1]}，R 是 {R.shape[0]}x{R.shape[1]}"
      f"  （{C.size + R.size} 個數字 vs 原本 {A.size} 個）")
print("第 3 欄不是獨立欄，R 的第 3 欄告訴我們怎麼組出它：",
      f"{R[0, 2]:.0f}·(欄1) + {R[1, 2]:.0f}·(欄2)")
check("用 R 的係數真的組出 A 的第 3 欄", C @ R[:, 2], A[:, 2])

# 列秩 = 欄秩（用隨機矩陣再驗證一次）
rng = np.random.default_rng(7)
for trial in range(3):
    M = rng.integers(-3, 4, size=(5, 7)).astype(float)
    M[:, 3] = M[:, 0] + 2 * M[:, 1]           # 刻意製造相依欄
    M[4, :] = M[0, :] - M[2, :]               # 刻意製造相依列
    check(f"隨機矩陣 #{trial + 1}：rank(A) = rank(Aᵀ)", rank(M) == rank(M.T))

# %% [markdown]
# ### 五大分解的第一塊拼圖
#
# Strang 把整本書歸結為五個分解。本章完成了第一個：
#
# | 章 | 分解 | 回答的問題 |
# |---|---|---|
# | **1** | $A=CR$ | 哪些欄是獨立的？秩是多少？ |
# | 2 | $A=LU$ | 怎麼有系統地解 $A\mathbf x=\mathbf b$？ |
# | 4 | $A=QR$ | 怎麼把欄換成正交基底？ |
# | 6 | $A=X\Lambda X^{-1}$ | 哪些方向只被拉伸不被轉向？ |
# | 7 | $A=U\Sigma V^{\mathsf T}$ | 任意矩陣的「最佳」座標是什麼？ |

# %% [markdown]
# ## 動手練習
#
# 1. 令 $\mathbf v=(1,2),\ \mathbf w=(2,4)$。$c\mathbf v+d\mathbf w$ 填出什麼？把 $\mathbf b=(3,6)$
#    與 $\mathbf b=(3,7)$ 分別代入 $A\mathbf x=\mathbf b$，哪一個有解？
# 2. 證明 $\|\mathbf v+\mathbf w\|^2+\|\mathbf v-\mathbf w\|^2=2\|\mathbf v\|^2+2\|\mathbf w\|^2$
#    （平行四邊形恆等式），再用程式對隨機向量驗證。
# 3. 對 $4\times 6$ 的隨機矩陣做 $A=CR$，確認 $C$ 的欄數等於 `np.linalg.matrix_rank` 的結果。
# 4. 用外積觀點計算 $AB$，並觀察只保留前 1 項時誤差多大（這是低秩近似的先驗）。
#
# 下面是第 2 題與第 4 題的參考解答。

# %%
section("練習參考解答")

rng = np.random.default_rng(42)
a, bb = rng.standard_normal(5), rng.standard_normal(5)
lhs = np.linalg.norm(a + bb) ** 2 + np.linalg.norm(a - bb) ** 2
rhs = 2 * np.linalg.norm(a) ** 2 + 2 * np.linalg.norm(bb) ** 2
check("練習 2：平行四邊形恆等式", lhs, rhs)

A = rng.standard_normal((6, 4))
B = rng.standard_normal((4, 6))
full = A @ B
for k in range(1, 5):
    approx = sum(np.outer(A[:, i], B[i, :]) for i in range(k))
    err = np.linalg.norm(full - approx, "fro") / np.linalg.norm(full, "fro")
    print(f"  練習 4：保留前 {k} 個外積 → 相對誤差 {err:.3%}")
check("練習 4：保留全部 4 項時完全重建", sum(np.outer(A[:, i], B[i, :]) for i in range(4)), full)

# %% [markdown]
# ## 本章重點回顧
#
# * 線性代數只有一個基本動作：**線性組合**。$A\mathbf x$ 就是 $A$ 各欄的線性組合。
# * 點積 $\mathbf v\cdot\mathbf w$ 同時給出長度、垂直判定與夾角，並導出 Schwarz 與三角不等式。
# * **欄空間** $C(A)$ 是所有 $A\mathbf x$；它的維度就是**秩** $r$。
# * 矩陣乘法有四種等價算法；其中「欄 × 列」把矩陣寫成 $r$ 個秩一矩陣之和。
# * $A=CR$ 直接證明了**列秩 = 欄秩**，並預告了後續所有分解。
#
# 下一章：有系統地解 $A\mathbf x=\mathbf b$ ── 消去法與 $A=LU$。
