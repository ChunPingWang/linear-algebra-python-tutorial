# %% [markdown]
# # 第 3 章　四個基本子空間
#
# > 對應 Strang 第 3 章（3.1 向量空間、3.2 用消去法求零空間 $A=CR$、3.3 $A\mathbf x=\mathbf b$ 的完整解、
# > 3.4 獨立/基底/維度、3.5 四個子空間的維度），以及 Riley 8.1（向量空間）、8.11（秩）、8.18。
#
# 本章是全書的樞紐。給定任意 $m\times n$ 矩陣 $A$（秩 $r$），有四個子空間：
#
# | 子空間 | 符號 | 在哪個空間 | 維度 | 基底來源 |
# |---|---|---|---|---|
# | 欄空間 | $C(A)$ | $\mathbb R^m$ | $r$ | $A$ 的主元欄 |
# | 列空間 | $C(A^{\mathsf T})$ | $\mathbb R^n$ | $r$ | $R$ 的非零列 |
# | 零空間 | $N(A)$ | $\mathbb R^n$ | $n-r$ | $n-r$ 個特殊解 |
# | 左零空間 | $N(A^{\mathsf T})$ | $\mathbb R^m$ | $m-r$ | $A^{\mathsf T}\mathbf y=\mathbf 0$ 的解 |
#
# **線性代數基本定理（第一部分）**：
# $$r+(n-r)=n,\qquad r+(m-r)=m$$
#
# ```bash
# python chapters/ch03_four_subspaces.py
# python tools/build_notebooks.py ch03
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

from linalg_tutorial.subspaces import (
    complete_solution,
    cr_factor,
    four_subspace_report,
    is_independent,
    left_nullspace,
    nullspace,
    particular_solution,
    rank,
    rref,
)
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 3.1　向量空間與子空間
#
# 向量空間的唯一要求：**對線性組合封閉**。$\mathbf v,\mathbf w$ 在空間裡
# $\Rightarrow$ $c\mathbf v+d\mathbf w$ 也在裡面（所以零向量一定在）。
#
# 「向量」不必是數字欄：矩陣、函數都可以。
#
# | 空間 | 維度 | 一組基底 |
# |---|---|---|
# | $\mathbb R^n$ | $n$ | $\mathbf e_1,\dots,\mathbf e_n$ |
# | 全部 $2\times2$ 矩陣 | 4 | $E_{11},E_{12},E_{21},E_{22}$ |
# | $n\times n$ 對稱矩陣 | $\frac{n^2+n}{2}$ | $E_{ii}$ 與 $E_{ij}+E_{ji}$ |
# | $y''=-y$ 的解 | 2 | $\sin x,\ \cos x$ |
# | 次數 $\le d$ 的多項式 | $d+1$ | $1,x,\dots,x^d$ |

# %%
section("3.1 子空間的封閉性檢查")


def closed_under_combination(gen, samples=200, seed=0):
    """隨機抽兩個元素做線性組合，檢查是否仍滿足集合的條件。"""
    rng = np.random.default_rng(seed)
    make, test = gen
    for _ in range(samples):
        v, w = make(rng), make(rng)
        c, d = rng.standard_normal(2)
        if not test(c * v + d * w):
            return False
    return True


# 是子空間：平面 x+y+z=0
plane = (lambda rng: np.array([1.0, -1.0, 0.0]) * rng.standard_normal()
         + np.array([1.0, 0.0, -1.0]) * rng.standard_normal(),
         lambda v: abs(v.sum()) < 1e-9)
# 不是子空間：第一象限（不含負數倍）
quadrant = (lambda rng: np.abs(rng.standard_normal(2)),
            lambda v: np.all(v >= 0))
check("平面 x+y+z=0 是子空間", closed_under_combination(plane))
check("第一象限不是子空間", not closed_under_combination(quadrant))

# 矩陣空間：對稱矩陣的維度 = n(n+1)/2
for n in [2, 3, 4]:
    basis = []
    for i in range(n):
        for j in range(i, n):
            E = np.zeros((n, n))
            E[i, j] = E[j, i] = 1.0
            basis.append(E.ravel())
    B = np.column_stack(basis)
    check(f"{n}x{n} 對稱矩陣維度 = n(n+1)/2 = {n * (n + 1) // 2}",
          rank(B) == n * (n + 1) // 2)

# 函數空間：y'' = -y 的解空間維度 2（用取樣點把函數變成向量）
t = np.linspace(0, 2 * np.pi, 50)
F = np.column_stack([np.sin(t), np.cos(t), np.sin(t + 0.7)])   # 第三個是前兩個的組合
check("sin, cos 獨立；sin(t+0.7) 是它們的組合 → 維度 2", rank(F) == 2)

# %% [markdown]
# ## 3.2　用消去法求零空間
#
# 消去法不改變零空間：$N(A)=N(R_0)=N(R)$。
# 把 $A$ 一路化簡到 **rref**（reduced row echelon form）：
#
# $$R_0=\begin{bmatrix}I&F\\0&0\end{bmatrix}\!P
# \qquad(\text{主元欄裝 }I\text{，自由欄裝 }F)$$
#
# 讀法：$n-r$ 個**自由變數**各給一個 1、其餘 0，就得到 $n-r$ 個**特殊解**，
# 它們正好是 $\begin{bmatrix}-F\\I\end{bmatrix}$ 的欄（視主元位置重排）。這就是 $N(A)$ 的基底。

# %%
section("3.2 rref 與零空間的基底")

A = np.array([
    [1.0, 7.0, 3.0, 3.0],
    [2.0, 14.0, 6.0, 70.0],
    [2.0, 14.0, 9.0, 97.0],
])
show_matrix("A", A)
R0, piv = rref(A)
show_matrix("R₀ = rref(A)", R0)
print(f"主元欄 = {[p + 1 for p in piv]}（1-based），自由欄 = "
      f"{[c + 1 for c in range(A.shape[1]) if c not in piv]}")
print(f"rank r = {len(piv)}，零空間維度 n − r = {A.shape[1] - len(piv)}")

N = nullspace(A)
show_matrix("N(A) 的基底（特殊解）", N)
check("A·（每個特殊解）= 0", A @ N, np.zeros((A.shape[0], N.shape[1])))
check("特殊解彼此獨立", is_independent(N))
check("N(A) = N(R₀)", R0 @ N, np.zeros((R0.shape[0], N.shape[1])))

C, R = cr_factor(A)
check("A = CR（C 取主元欄，R 去掉零列）", C @ R, A)
print("\n相依欄的秘密：F 告訴我們怎麼用獨立欄組出相依欄")
free_cols = [c for c in range(A.shape[1]) if c not in piv]
for f in free_cols:
    coeffs = R0[:len(piv), f]
    terms = " + ".join(f"{coeffs[i]:g}·(欄{piv[i] + 1})" for i in range(len(piv)))
    print(f"  欄{f + 1} = {terms}")
    check(f"  驗證欄{f + 1}", C @ coeffs, A[:, f])

print("\n重要事實：n > m（欄比列多）時，Ax = 0 一定有非零解")
wide = np.array([[1.0, 2.0, 2.0, 4.0], [3.0, 8.0, 6.0, 16.0]])
check("2x4 矩陣的零空間維度 ≥ 2", nullspace(wide).shape[1] >= 2)

# %% [markdown]
# ## 3.3　$A\mathbf x=\mathbf b$ 的完整解
#
# $$\boxed{\ \mathbf x_{\text{complete}}=\mathbf x_{p}+\mathbf x_{n}\ }$$
#
# * $\mathbf x_p$：**一個**特解（把自由變數全設為 0 最方便，直接從 $\mathbf d$ 讀出）
# * $\mathbf x_n$：$N(A)$ 中的**任意**向量
#
# 可解條件：消去後 $R_0$ 的零列對應的 $\mathbf d$ 也必須是 0，
# 等價於 $\mathbf b\in C(A)$，也等價於 $\mathbf b \perp N(A^{\mathsf T})$。
#
# 四種結局，完全由 $r$ 與 $m,n$ 決定：
#
# | 條件 | 形狀 | 解的個數 |
# |---|---|---|
# | $r=m=n$ | 方陣可逆 | 恰 1 組 |
# | $r=m<n$ | 矮胖（滿列秩）| 無窮多 |
# | $r=n<m$ | 高瘦（滿欄秩）| 0 或 1 組 |
# | $r<m,\ r<n$ | 都不滿 | 0 或無窮多 |

# %%
section("3.3 完整解 x = x_p + x_n")

A2 = np.array([
    [1.0, 3.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 4.0],
    [1.0, 3.0, 1.0, 6.0],
])
b_ok = np.array([1.0, 6.0, 7.0])      # 1 + 6 = 7 → 可解
b_bad = np.array([1.0, 6.0, 8.0])     # 不滿足 b₁+b₂=b₃ → 無解

for b, tag in [(b_ok, "b = (1,6,7)"), (b_bad, "b = (1,6,8)")]:
    info = complete_solution(A2, b)
    print(f"\n{tag}：{info['description']}")
    if info["consistent"]:
        xp = info["particular"]
        print(f"  特解 x_p = {xp}（自由變數設 0）")
        check("  A·x_p = b", A2 @ xp, b)
        Nb = info["nullspace_basis"]
        rng = np.random.default_rng(0)
        for _ in range(3):
            coef = rng.standard_normal(Nb.shape[1])
            x_any = xp + Nb @ coef
            check(f"  x_p + N(A)·{np.round(coef, 2)} 也是解", A2 @ x_any, b)
    else:
        print(f"  可解條件 b₁+b₂−b₃ = {b[0] + b[1] - b[2]:g} ≠ 0")
        y = left_nullspace(A2)[:, 0]
        print(f"  左零空間向量 y = {y}；yᵀb = {y @ b:.4g} ≠ 0 ⇒ b ∉ C(A)")

# 四種結局各舉一例
section("3.3 由秩決定的四種結局")
cases = [
    ("r = m = n（方陣可逆）", np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([3.0, 4.0])),
    ("r = m < n（矮胖，滿列秩）", np.array([[1.0, 1.0, 1.0], [1.0, 2.0, -1.0]]),
     np.array([3.0, 4.0])),
    ("r = n < m（高瘦，滿欄秩）", np.array([[1.0, 1.0], [1.0, 2.0], [-2.0, -3.0]]),
     np.array([1.0, 2.0, -3.0])),
    ("r < m, r < n（都不滿）", np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
     np.array([1.0, 2.0, 3.0])),
]
for tag, M, rhs in cases:
    info = complete_solution(M, rhs)
    m_, n_ = M.shape
    print(f"\n{tag}：{m_}x{n_}，r = {info['rank']}")
    print(f"  {info['description']}")
    if info["consistent"]:
        check("  特解正確", M @ info["particular"], rhs)

# %% [markdown]
# ### 把「一條解線」畫出來
#
# 兩個方程、三個未知數（滿列秩 $r=m=2<n=3$）：兩個平面交於一條直線。
# 這條直線就是 $\mathbf x_p+c\,\mathbf s$。

# %%
M = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, -1.0]])
rhs = np.array([3.0, 4.0])
xp = particular_solution(M, rhs)
s = nullspace(M)[:, 0]
print(f"x_p = {xp}，零空間基底 s = {s}")
check("x_p 是解", M @ xp, rhs)
check("s 在零空間", M @ s, np.zeros(2))

fig, ax = new_axes("Complete solution = particular + nullspace line",
                   figsize=(6.0, 5.0), d3=True)
ts = np.linspace(-2.5, 2.5, 60)
line = xp[:, None] + s[:, None] * ts
ax.plot(line[0], line[1], line[2], "C0", lw=2.2, label=r"$x_p + c\,s$")
ax.scatter(*xp, color="C3", s=55, label=r"$x_p$ (free var = 0)")
ax.quiver(*xp, *s, color="C2", length=1.0, normalize=True, label="nullspace direction")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.legend(fontsize=8)
finish(fig, "ch03_solution_line")

# %% [markdown]
# ## 3.4　獨立、生成、基底、維度
#
# * **獨立**：$A\mathbf x=\mathbf 0$ 只有 $\mathbf x=\mathbf 0$（$N(A)=\{\mathbf 0\}$）
# * **生成（span）**：所有線性組合填滿該空間
# * **基底**：獨立 **且** 生成 → 每個向量有**唯一**的表示法
# * **維度**：任一基底的向量個數（與選哪組基底無關）
#
# 兩個常用事實：
#
# * $\mathbb R^m$ 裡任意 $n>m$ 個向量必相依
# * $\mathbf v_1,\dots,\mathbf v_n$ 是 $\mathbb R^n$ 的基底 $\iff$ 它們是某個可逆矩陣的欄

# %%
section("3.4 基底與唯一表示法")

V = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])   # 可逆 → 是基底
show_matrix("基底矩陣 V（下三角，可逆）", V)
check("V 可逆 ⇒ 欄是 R³ 的基底", abs(np.linalg.det(V)) > 1e-12)

target = np.array([3.0, 5.0, 9.0])
coords = np.linalg.solve(V, target)
print(f"\nb = {target} 在這組基底下的座標 = {coords}")
check("唯一表示：V·coords = b", V @ coords, target)

rng = np.random.default_rng(5)
print("\n維度不變性：隨便換一組基底，向量個數都是 3")
for trial in range(3):
    B = rng.standard_normal((3, 3))
    if abs(np.linalg.det(B)) > 1e-6:
        check(f"  隨機基底 #{trial + 1}：rank = 3", rank(B) == 3)

# n > m 必相依
check("R³ 中任取 4 個向量必相依", rank(rng.standard_normal((3, 4))) < 4)

# 基底的「換底矩陣」：V = W·B，B 可逆 ⇔ V 的欄也是基底
W = np.eye(3)
Bchange = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
check("c ≠ 1 時 B 可逆 ⇒ 新向量仍是基底", rank(W @ Bchange) == 3)
Bdeg = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]])
check("c = 1 時 B 奇異 ⇒ 新向量相依（v1 − v2 + v3 = 0）", rank(W @ Bdeg) == 2)

# %% [markdown]
# ## 3.5　四個子空間的維度：基本定理（第一部分）
#
# $$\dim C(A)=\dim C(A^{\mathsf T})=r,\qquad
# \dim N(A)=n-r,\qquad \dim N(A^{\mathsf T})=m-r$$
#
# 直觀圖像：
#
# ```
#        R^n                               R^m
#   ┌─────────────┐                  ┌─────────────┐
#   │ row space   │ ── A ──────────▶ │ column space│
#   │   dim r     │                  │   dim r     │
#   ├─────────────┤                  ├─────────────┤
#   │ nullspace   │ ── A ──▶ 0       │ left nullsp.│
#   │ dim n − r   │                  │ dim m − r   │
#   └─────────────┘                  └─────────────┘
# ```
#
# 關鍵洞見：**從列空間到欄空間，$A$ 就像一個 $r\times r$ 的可逆矩陣**。
# 是零空間迫使我們在第 4 章定義偽逆 $A^+$。

# %%
section("3.5 四個子空間的完整報告")

A3 = np.array([
    [1.0, 3.0, 5.0, 0.0, 7.0],
    [0.0, 0.0, 0.0, 1.0, 2.0],
    [1.0, 3.0, 5.0, 1.0, 9.0],
])
rep = four_subspace_report(A3)
m_, n_ = rep["shape"]
print(f"A 是 {m_}x{n_}，rank r = {rep['rank']}\n")
for name, dim in rep["dims"].items():
    where = "R^n" if name in ("N(A)", "C(A^T)") else "R^m"
    print(f"  dim {name:<8} = {dim}   （在 {where} 中）")
print(f"\n計數定理：r + (n−r) = {rep['rank']} + {n_ - rep['rank']} = n = {n_}")
print(f"             r + (m−r) = {rep['rank']} + {m_ - rep['rank']} = m = {m_}")

show_matrix("C(A) 基底（A 的主元欄）", rep["column_space"])
show_matrix("C(Aᵀ) 基底（rref 的非零列，以欄表示）", rep["row_space"])
show_matrix("N(A) 基底", rep["nullspace"])
show_matrix("N(Aᵀ) 基底", rep["left_nullspace"])

check("列空間維度 = 欄空間維度", rep["row_space"].shape[1], rep["column_space"].shape[1])
check("A·N(A) = 0", A3 @ rep["nullspace"], np.zeros((m_, n_ - rep["rank"])))
check("N(Aᵀ)ᵀ·A = 0", rep["left_nullspace"].T @ A3,
      np.zeros((m_ - rep["rank"], n_)))
print("\n正交性預告（第 4 章）：")
check("N(A) ⟂ C(Aᵀ)", rep["row_space"].T @ rep["nullspace"],
      np.zeros((rep["rank"], n_ - rep["rank"])))
check("N(Aᵀ) ⟂ C(A)", rep["column_space"].T @ rep["left_nullspace"],
      np.zeros((rep["rank"], m_ - rep["rank"])))

# %% [markdown]
# ### 應用：圖的入射矩陣與 Kirchhoff 定律
#
# 這是四個子空間最漂亮的實例（Strang 3.5 的 Example 2）。
# 連通圖有 $n$ 個節點、$m$ 條邊，入射矩陣 $A$（$m\times n$，每列一個 $+1$ 一個 $-1$）：
#
# | 子空間 | 維度 | 圖論意義 |
# |---|---|---|
# | $N(A)$ | 1 | 常數電位 $(c,c,\dots,c)$ |
# | $C(A^{\mathsf T})$ | $n-1$ | 一棵**生成樹**的邊（無迴路）|
# | $C(A)$ | $n-1$ | 電壓定律：沿任何迴路電位差總和為 0 |
# | $N(A^{\mathsf T})$ | $m-n+1$ | **電流定律**：獨立迴路電流 |
#
# 再把維度加起來，就得到拓樸學第一定理 ── **Euler 公式**：
#
# $$(\text{節點})-(\text{邊})+(\text{獨立迴路})=n-m+(m-n+1)=1$$

# %%
section("3.5 入射矩陣：Kirchhoff 定律與 Euler 公式")


def incidence_matrix(n_nodes, edges):
    """圖的入射矩陣：每條邊 (i, j) 給一列，第 i 欄 −1、第 j 欄 +1。"""
    A = np.zeros((len(edges), n_nodes))
    for k, (i, j) in enumerate(edges):
        A[k, i] = -1.0
        A[k, j] = 1.0
    return A


edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]        # Strang 的 5 邊 4 節點圖
A4 = incidence_matrix(4, edges)
show_matrix("入射矩陣 A（5 邊 x 4 節點）", A4)

r4 = rank(A4)
m4, n4 = A4.shape
print(f"\nrank = {r4} = n − 1 = {n4 - 1}")
check("N(A) 由常數向量 (1,1,1,1) 生成", A4 @ np.ones(4), np.zeros(5))
check("dim N(A) = 1", nullspace(A4).shape[1] == 1)

Y = left_nullspace(A4)
show_matrix("N(Aᵀ) 基底 = 獨立迴路電流", Y)
print(f"dim N(Aᵀ) = m − r = {m4} − {r4} = {Y.shape[1]}（= 獨立迴路數）")
check("電流定律 AᵀY = 0（每個節點流入 = 流出）", A4.T @ Y, np.zeros((n4, Y.shape[1])))

print(f"\nEuler 公式：節點 − 邊 + 迴路 = {n4} − {m4} + {Y.shape[1]} = "
      f"{n4 - m4 + Y.shape[1]}")
check("Euler 公式 = 1", n4 - m4 + Y.shape[1] == 1)

# 電壓定律：Ax 沿迴路總和為 0
x_potential = np.array([0.0, 2.0, 5.0, 7.0])             # 四個節點的電位
v_drop = A4 @ x_potential
print(f"\n節點電位 x = {x_potential}")
print(f"各邊電位差 Ax = {v_drop}")
loop = [0, 2, -1]            # 邊 1(0→1) + 邊 3(1→2) − 邊 2(0→2)
check("電壓定律：迴路 0→1→2→0 電位差總和 = 0",
      v_drop[0] + v_drop[2] - v_drop[1], 0.0)

# 生成樹的列獨立、含迴路的列相依
check("邊 {1,2,3} 形成迴路 → 三列相依", rank(A4[[0, 1, 2], :]) == 2)
check("邊 {1,2,4} 形成樹 → 三列獨立", rank(A4[[0, 1, 3], :]) == 3)

# %% [markdown]
# ### 秩 $r$ = $r$ 個秩一矩陣之和
#
# $A=CR$ 按「欄 × 列」展開，就是
#
# $$A=\mathbf u_1\mathbf v_1^{\mathsf T}+\dots+\mathbf u_r\mathbf v_r^{\mathsf T}$$
#
# 第 7 章的 SVD 會在這個式子上加一個要求：$\mathbf u$ 與 $\mathbf v$ 都正交。

# %%
section("3.5 秩 r = r 個秩一矩陣之和")

A5 = np.array([[1.0, 0.0, 3.0], [1.0, 1.0, 4.0], [4.0, 2.0, 2.0]])
C5, R5 = cr_factor(A5)
acc = np.zeros_like(A5)
print(f"rank(A) = {rank(A5)}")
for k in range(C5.shape[1]):
    piece = np.outer(C5[:, k], R5[k, :])
    acc += piece
    print(f"  加入第 {k + 1} 個秩一矩陣：rank(累積) = {rank(acc)}，"
          f"‖A − 累積‖ = {np.linalg.norm(A5 - acc):.4g}")
check("r 個秩一矩陣加總 = A", acc, A5)
check("rank(AB) ≤ min(rank A, rank B)",
      rank(A5 @ A5.T) <= min(rank(A5), rank(A5.T)))

# %% [markdown]
# ## 動手練習
#
# 1. 在 $5\times 6$ 的零矩陣中放入 4 個 1，使其秩最小／最大。四個子空間維度的總和會變嗎？
# 2. 對你自己畫的一張圖（$\ge 6$ 條邊）建立入射矩陣，驗證 Euler 公式。
# 3. 找出所有使 $A\mathbf x=\mathbf b$ 可解的 $\mathbf b$，並證明該條件等價於 $\mathbf b\perp N(A^{\mathsf T})$。
# 4. 證明 $\operatorname{rank}(AB)\le\min(\operatorname{rank}A,\operatorname{rank}B)$，
#    並找出使等號不成立的例子。
#
# 參考解答：

# %%
section("練習參考解答")

print("練習 1：四個子空間維度總和 = r + (n−r) + r + (m−r) = n + m = 11（恆定）")
Z = np.zeros((5, 6))
Z[0, 0] = Z[0, 1] = Z[0, 2] = Z[0, 3] = 1.0            # 同一列 → rank 1
check("  四個 1 放同一列：rank = 1", rank(Z) == 1)
Z2 = np.zeros((5, 6))
for i in range(4):
    Z2[i, i] = 1.0                                       # 不同列不同欄 → rank 4
check("  四個 1 放不同列不同欄：rank = 4", rank(Z2) == 4)
for M in (Z, Z2):
    r_ = rank(M)
    total = r_ + (M.shape[1] - r_) + r_ + (M.shape[0] - r_)
    check(f"  維度總和 = n + m = 11（此例 rank={r_}）", total == 11)

# 練習 2：較大的圖
edges2 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (3, 4)]
A6 = incidence_matrix(5, edges2)
loops = left_nullspace(A6).shape[1]
print(f"\n練習 2：{A6.shape[0]} 邊、{A6.shape[1]} 節點，獨立迴路 = {loops}")
check("  Euler 公式", A6.shape[1] - A6.shape[0] + loops == 1)

# 練習 3
A7 = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
Yl = left_nullspace(A7)
rng = np.random.default_rng(11)
print("\n練習 3：b 可解 ⇔ yᵀb = 0 對所有 y ∈ N(Aᵀ)")
for _ in range(3):
    xx = rng.standard_normal(2)
    b_in = A7 @ xx
    check(f"  b ∈ C(A) ⇒ Yᵀb = 0", Yl.T @ b_in, np.zeros(Yl.shape[1]))
b_out = np.array([1.0, 0.0, 0.0])
check("  b ∉ C(A) ⇒ Yᵀb ≠ 0", np.linalg.norm(Yl.T @ b_out) > 1e-9)
check("  且此時真的無解", complete_solution(A7, b_out)["consistent"] is False)

# 練習 4
P = np.array([[1.0, 0.0], [0.0, 0.0]])
Q = np.array([[0.0, 0.0], [0.0, 1.0]])
print(f"\n練習 4：rank(P) = {rank(P)}, rank(Q) = {rank(Q)}, "
      f"rank(PQ) = {rank(P @ Q)} → 等號不成立的例子")

# %% [markdown]
# ## 本章重點回顧
#
# * 子空間 = 對線性組合封閉的集合；矩陣與函數也能構成向量空間。
# * $A\to R_0=\text{rref}(A)$：主元欄給 $C(A)$ 的基底，自由欄給 $N(A)$ 的特殊解。
# * $A\mathbf x=\mathbf b$ 的完整解 = 一個特解 + 整個零空間。
# * 基底 = 獨立 + 生成；維度與基底選擇無關。
# * **基本定理（第一部分）**：$\dim C(A)=\dim C(A^{\mathsf T})=r$，
#   $\dim N(A)=n-r$，$\dim N(A^{\mathsf T})=m-r$。
# * 入射矩陣把四個子空間翻譯成 Kirchhoff 電壓／電流定律與 Euler 公式。
#
# 下一章：四個子空間不只維度互補，它們還**互相垂直** ── 正交性、投影與最小平方。
