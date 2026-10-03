# %% [markdown]
# # 第 6 章　特徵值與對角化
#
# > 對應 Strang 6.1（$A\mathbf x=\lambda\mathbf x$）、6.2（對角化、$A^k$、相似矩陣），
# > 附錄 5（Jordan 形式）、附錄 8（Markov 矩陣），以及 Riley 8.13–8.16。
#
# 前五章在解 $A\mathbf x=\mathbf b$（**穩態**）。本章開始處理**變化**：
# $\mathbf u_{k+1}=A\mathbf u_k$ 與 $d\mathbf u/dt=A\mathbf u$。
# 這類問題**不能**用消去法解，要用特徵值。
#
# $$\boxed{A\mathbf x=\lambda\mathbf x}\qquad
# \Longleftrightarrow\qquad \det(A-\lambda I)=0$$
#
# 核心想法：把輸入分解成特徵向量，**每個特徵向量各走自己的路**
# $$A^k(c_1\mathbf x_1+c_2\mathbf x_2)=c_1\lambda_1^k\mathbf x_1+c_2\lambda_2^k\mathbf x_2$$
#
# ```bash
# python chapters/ch06_eigenvalues.py
# python tools/build_notebooks.py ch06
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

from linalg_tutorial.determinant import det_lu
from linalg_tutorial.eigen import (
    diagonalize,
    eigen_2x2,
    inverse_power_iteration,
    jacobi_eigen,
    markov_steady_state,
    matrix_power_by_eigen,
    power_iteration,
    qr_algorithm,
    rayleigh_quotient,
)
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import draw_vector, finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 6.1　$A\mathbf x=\lambda\mathbf x$：特徵值怎麼來
#
# 把 $A\mathbf x=\lambda\mathbf x$ 改寫成 $(A-\lambda I)\mathbf x=\mathbf 0$。
# 要有非零解，$A-\lambda I$ 必須**奇異**：
#
# $$\det(A-\lambda I)=0\quad(\text{特徵多項式，}n\text{ 次}\Rightarrow n\text{ 個 }\lambda)$$
#
# 兩個**必做的檢查**：
#
# $$\lambda_1+\dots+\lambda_n=\operatorname{trace}A,\qquad
# \lambda_1\lambda_2\cdots\lambda_n=\det A$$
#
# 重要性質（特徵向量都不變！）：
#
# | 矩陣 | 特徵值 |
# |---|---|
# | $A^k$ | $\lambda^k$ |
# | $A^{-1}$ | $1/\lambda$ |
# | $A+cI$ | $\lambda+c$ |
# | $A^{\mathsf T}$ | 同 $A$（但特徵向量不同）|

# %%
section("6.1 手算特徵值：Strang 的 Markov 例子")

A = np.array([[0.8, 0.3], [0.2, 0.7]])
show_matrix("A（Markov 矩陣，每欄和為 1）", A)
print(f"trace = {np.trace(A):.4g}，det = {det_lu(A):.4g}")
print("特徵多項式 λ² − 1.5λ + 0.5 = (λ − 1)(λ − 0.5)")

lams, X = eigen_2x2(A)
lams = np.real_if_close(lams)
print(f"\n特徵值 λ = {np.real(lams)}")
check("λ 的和 = trace", float(np.sum(np.real(lams))), float(np.trace(A)))
check("λ 的積 = det", float(np.prod(np.real(lams))), det_lu(A))
check("與 numpy.linalg.eigvals 一致（排序後）",
      np.sort(np.real(lams)), np.sort(np.real(np.linalg.eigvals(A))))

for lam in np.real(lams):
    check(f"det(A − {lam:.4g}I) = 0", abs(det_lu(A - lam * np.eye(2))) < 1e-12)

x1 = np.array([0.6, 0.4])          # λ = 1 的特徵向量（穩態）
x2 = np.array([1.0, -1.0])         # λ = 0.5 的特徵向量（衰減模態）
check("A·(0.6, 0.4) = 1·(0.6, 0.4)（穩態）", A @ x1, 1.0 * x1)
check("A·(1, −1) = 0.5·(1, −1)（衰減）", A @ x2, 0.5 * x2)

print("\n把 A 的第 1 欄分解成特徵向量：")
col1 = A[:, 0]
coef = np.linalg.solve(np.column_stack([x1, x2]), col1)
print(f"  (0.8, 0.2) = {coef[0]:.4g}·x₁ + {coef[1]:.4g}·x₂")
for k in [1, 2, 3, 10, 100]:
    comb = coef[0] * 1.0 ** k * x1 + coef[1] * 0.5 ** k * x2
    print(f"  Aᵏ 的第 1 欄 (k={k:>3}) = {np.round(comb, 6)}"
          f"   （衰減項 0.5^{k} = {0.5 ** k:.3g}）")
check("A¹⁰⁰ 的欄 → 穩態 (0.6, 0.4)",
      matrix_power_by_eigen(A, 100)[:, 0], x1, tol=1e-8)
check("markov_steady_state 也給 (0.6, 0.4)", markov_steady_state(A), x1)

# %% [markdown]
# ### 特殊矩陣的特徵值一眼看出來
#
# | 矩陣 | $\lambda$ | 為什麼 |
# |---|---|---|
# | 投影 $P$ | $1,\dots,1,0,\dots,0$ | 欄空間不動、零空間歸零 |
# | 反射／交換 $E$ | $\pm1$ | 鏡面上不動、垂直方向反向 |
# | 旋轉 $Q$（$90°$）| $\pm i$ | 沒有實的不變方向！ |
# | 三角矩陣 | 對角線元素 | $\det(A-\lambda I)$ 直接展開 |
# | 奇異矩陣 | 含 $\lambda=0$ | 零空間非空 |
# | Markov 矩陣 | 含 $\lambda=1$ | 每欄和為 1 |

# %%
section("6.1 特殊矩陣的特徵值")

specials = {
    "投影 P（投到 (1,1) 方向）": np.array([[0.5, 0.5], [0.5, 0.5]]),
    "交換（反射）E": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "旋轉 90° Q": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "上三角": np.array([[1.0, 9.0], [0.0, 2.0]]),
    "奇異": np.array([[1.0, 2.0], [2.0, 4.0]]),
}
for name, M in specials.items():
    lam = np.linalg.eigvals(M)
    print(f"  {name:<24} λ = {np.round(lam, 4)}"
          f"   trace = {np.trace(M):+.3g}, det = {det_lu(M):+.3g}")
check("投影的 λ 只有 0 和 1",
      np.sort(np.real(np.linalg.eigvals(specials["投影 P（投到 (1,1) 方向）"]))),
      np.array([0.0, 1.0]))
check("旋轉 90° 的 λ = ±i",
      np.sort_complex(np.linalg.eigvals(specials["旋轉 90° Q"])),
      np.sort_complex(np.array([1j, -1j])), tol=1e-8)
print("\n實矩陣的複特徵值一定成共軛對（因為特徵多項式係數是實數）")
check("E = 2P − I ⇒ λ_E = 2λ_P − 1",
      np.sort(np.real(np.linalg.eigvals(specials["交換（反射）E"]))),
      np.sort(2 * np.array([0.0, 1.0]) - 1))

# 特徵值的運算性質
rng = np.random.default_rng(3)
S = rng.standard_normal((4, 4))
S = S + S.T                          # 用對稱矩陣避免複數，方便比較
lam = np.sort(np.linalg.eigvalsh(S))
check("A² 的 λ = λ²", np.sort(np.linalg.eigvalsh(S @ S)), np.sort(lam ** 2), tol=1e-8)
check("A⁻¹ 的 λ = 1/λ", np.sort(np.linalg.eigvalsh(np.linalg.inv(S))),
      np.sort(1 / lam), tol=1e-8)
check("A + 5I 的 λ = λ + 5", np.sort(np.linalg.eigvalsh(S + 5 * np.eye(4))),
      np.sort(lam + 5), tol=1e-8)
check("Aᵀ 與 A 有相同的 λ", np.sort(np.real(np.linalg.eigvals(S.T))), lam, tol=1e-8)

# 常見誤解：λ(AB) ≠ λ(A)λ(B)
Aa = np.array([[0.0, 1.0], [0.0, 0.0]])
Bb = np.array([[0.0, 0.0], [1.0, 0.0]])
print(f"\n反例：A、B 的特徵值都是 0，但")
print(f"  λ(AB) = {np.linalg.eigvals(Aa @ Bb)}，λ(A+B) = {np.linalg.eigvals(Aa + Bb)}")
check("λ(AB) ≠ λ(A)·λ(B) 一般不成立",
      not np.allclose(np.sort(np.real(np.linalg.eigvals(Aa @ Bb))), 0))
print("  正確的說法：A、B 共用全部特徵向量 ⇔ AB = BA（量子力學的可觀測量交換關係）")
check("AB = BA ⇔ 共用特徵向量（此例 AB ≠ BA）", not np.allclose(Aa @ Bb, Bb @ Aa))

# %% [markdown]
# ### Gershgorin 圓盤：不算就估特徵值
#
# 每個特徵值都落在某個以 $a_{ii}$ 為圓心、半徑 $R_i=\sum_{j\ne i}|a_{ij}|$ 的圓盤內。
# 這在數值分析裡非常實用（例如判斷對角優勢矩陣可逆、迭代法是否收斂）。

# %%
section("6.1 Gershgorin 圓盤定理")

G = np.array([
    [5.0, 1.0, 0.0],
    [1.0, 2.0, -0.5],
    [0.0, -0.5, -3.0],
])
lam = np.linalg.eigvals(G)
print(f"實際特徵值 = {np.round(np.real(lam), 4)}\n")
ok = True
for i in range(3):
    R = np.sum(np.abs(G[i, :])) - abs(G[i, i])
    inside = [l for l in lam if abs(l - G[i, i]) <= R + 1e-9]
    print(f"  圓心 {G[i, i]:+.2f}，半徑 {R:.2f} → 包含 {np.round(np.real(inside), 4)}")
for l in lam:
    in_any = any(abs(l - G[i, i]) <= np.sum(np.abs(G[i, :])) - abs(G[i, i]) + 1e-9
                 for i in range(3))
    ok = ok and in_any
check("每個 λ 都落在至少一個圓盤內", ok)

fig, ax = new_axes("Gershgorin disks (complex plane)", figsize=(5.6, 4.6))
for i in range(3):
    R = np.sum(np.abs(G[i, :])) - abs(G[i, i])
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(G[i, i] + R * np.cos(th), R * np.sin(th), "C0", lw=1.4)
    ax.plot([G[i, i]], [0], "C0x", ms=6)
ax.plot(np.real(lam), np.imag(lam), "C3o", ms=8, label="eigenvalues")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.legend()
finish(fig, "ch06_gershgorin")

# %% [markdown]
# ## 6.2　對角化 $A=X\Lambda X^{-1}$
#
# 把 $n$ 個**獨立**特徵向量放進 $X$ 的欄，特徵值放進對角矩陣 $\Lambda$：
#
# $$AX=X\Lambda\quad\Longrightarrow\quad
# X^{-1}AX=\Lambda\quad\Longrightarrow\quad A=X\Lambda X^{-1}$$
#
# 於是**所有的冪次都變簡單**：
#
# $$A^k=X\Lambda^kX^{-1},\qquad
# \mathbf u_k=A^k\mathbf u_0=c_1\lambda_1^k\mathbf x_1+\dots+c_n\lambda_n^k\mathbf x_n$$
#
# 三步驟：① $\mathbf u_0=X\mathbf c$（分解）② 各乘 $\lambda_i^k$（演化）③ 加回來（重組）。

# %%
section("6.2 對角化與 Aᵏ")

A = np.array([[2.0, 4.0], [0.0, 6.0]])
out = diagonalize(A)
lam, X, Xinv = out
show_matrix("A（三角，λ 在對角線）", A)
show_matrix("X（特徵向量）", X)
print(f"Λ = diag{tuple(np.round(np.real(lam), 4))}")
check("AX = XΛ", A @ X, X @ np.diag(lam))
check("X⁻¹AX = Λ", Xinv @ A @ X, np.diag(lam), tol=1e-8)
check("A = XΛX⁻¹", X @ np.diag(lam) @ Xinv, A, tol=1e-8)

for k in [2, 3, 10]:
    check(f"Aᵏ = XΛᵏX⁻¹（k = {k}）",
          matrix_power_by_eigen(A, k), np.linalg.matrix_power(A, k), tol=1e-6)
show_matrix("A³（公式 [2³, 6³−2³; 0, 6³]）", matrix_power_by_eigen(A, 3))

# 三步驟解 u_{k+1} = A u_k
u0 = np.array([1.0, 1.0])
c = Xinv @ u0
print(f"\n三步驟：u₀ = {u0}")
print(f"  ① 分解：c = X⁻¹u₀ = {np.round(np.real(c), 4)}")
for k in [1, 2, 5]:
    uk = sum(np.real(c[i]) * np.real(lam[i]) ** k * np.real(X[:, i]) for i in range(2))
    print(f"  ②③ k = {k}：u_k = Σ cᵢλᵢᵏxᵢ = {np.round(uk, 4)}"
          f"  （直接乘 = {np.round(np.linalg.matrix_power(A, k) @ u0, 4)}）")
    check(f"  相符", uk, np.linalg.matrix_power(A, k) @ u0, tol=1e-8)

# %% [markdown]
# ### Fibonacci：用特徵值跳過 100 步
#
# $$F_{k+2}=F_{k+1}+F_k
# \quad\Longleftrightarrow\quad
# \mathbf u_{k+1}=\begin{bmatrix}1&1\\1&0\end{bmatrix}\mathbf u_k,\quad
# \mathbf u_k=\begin{bmatrix}F_{k+1}\\F_k\end{bmatrix}$$
#
# $\lambda^2=\lambda+1$ 給出黃金比例
# $\lambda_1=\frac{1+\sqrt5}{2}\approx1.618$ 與 $\lambda_2=\frac{1-\sqrt5}{2}\approx-0.618$，
#
# $$F_k=\frac{\lambda_1^k-\lambda_2^k}{\sqrt5}$$

# %%
section("6.2 Fibonacci 與黃金比例")

F = np.array([[1.0, 1.0], [1.0, 0.0]])
lamF = np.linalg.eigvalsh(F)        # F 對稱 ⇒ 特徵值必為實數
lam1, lam2 = float(np.max(lamF)), float(np.min(lamF))
print(f"λ₁ = {lam1:.10f}（黃金比例 φ），λ₂ = {lam2:.10f}")
check("λ² = λ + 1", lam1 ** 2, lam1 + 1)
check("λ₁λ₂ = det = −1", lam1 * lam2, -1.0)
check("λ₁+λ₂ = trace = 1", lam1 + lam2, 1.0)

print(f"\n{'k':>5} | {'遞迴計算 F_k':>20} | {'閉式 (λ₁ᵏ−λ₂ᵏ)/√5':>22} | {'F_{k+1}/F_k':>12}")
print("-" * 70)
fib = [0, 1]
while len(fib) < 101:
    fib.append(fib[-1] + fib[-2])
for k in [5, 10, 20, 50, 100]:
    closed = (lam1 ** k - lam2 ** k) / np.sqrt(5)
    ratio = fib[k] / fib[k - 1]
    print(f"{k:>5} | {fib[k]:>20,} | {closed:>22.6g} | {ratio:>12.10f}")
check("閉式公式（k = 50）正確", round((lam1 ** 50 - lam2 ** 50) / np.sqrt(5)), fib[50])
check("F_{k+1}/F_k → φ", fib[100] / fib[99], lam1, tol=1e-12)
print(f"\nF₁₀₀ = {fib[100]:,}，約等於 λ₁¹⁰⁰/√5 = {lam1 ** 100 / np.sqrt(5):.6g}")

# %% [markdown]
# ### 相似矩陣：換座標不改變特徵值
#
# $$C=B^{-1}AB\quad\Longrightarrow\quad
# C\mathbf y=\lambda\mathbf y\ \text{且}\ \mathbf x=B\mathbf y$$
#
# 證明一行：$(B^{-1}AB)(B^{-1}\mathbf x)=B^{-1}A\mathbf x=\lambda(B^{-1}\mathbf x)$。
#
# 推論：$AB$ 與 $BA$ 有相同的非零特徵值（即使形狀不同！）。

# %%
section("6.2 相似矩陣")

rng = np.random.default_rng(11)
A = np.array([[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [0.0, 0.0, 2.0]])
B = rng.standard_normal((3, 3))
C = np.linalg.inv(B) @ A @ B
print(f"A 的 λ = {np.sort(np.real(np.linalg.eigvals(A)))}")
print(f"C = B⁻¹AB 的 λ = {np.sort(np.real(np.linalg.eigvals(C)))}")
check("相似矩陣有相同特徵值",
      np.sort(np.real(np.linalg.eigvals(A))), np.sort(np.real(np.linalg.eigvals(C))),
      tol=1e-7)
check("trace 不變", np.trace(A), np.trace(C), tol=1e-8)
check("det 不變", det_lu(A), det_lu(C), tol=1e-7)

M = rng.standard_normal((3, 5))
N = rng.standard_normal((5, 3))
lam_mn = np.sort(np.real(np.linalg.eigvals(M @ N)))
lam_nm = np.sort(np.real(np.linalg.eigvals(N @ M)))
print(f"\nMN (3x3) 的 λ = {lam_mn}")
print(f"NM (5x5) 的 λ = {lam_nm}")
check("AB 與 BA 的非零特徵值相同",
      lam_mn, np.sort(lam_nm[np.abs(lam_nm) > 1e-8]), tol=1e-6)

# 消去法會改變特徵值！
print("\n注意：消去法不保留特徵值")
A_el = np.array([[1.0, 3.0], [2.0, 6.0]])
U_el = np.array([[1.0, 3.0], [0.0, 0.0]])
print(f"  A 的 λ = {np.sort(np.real(np.linalg.eigvals(A_el)))}（0 和 7）")
print(f"  U 的 λ = {np.sort(np.real(np.linalg.eigvals(U_el)))}（0 和 1）")
check("A 與 U 的特徵值不同",
      not np.allclose(np.sort(np.real(np.linalg.eigvals(A_el))),
                      np.sort(np.real(np.linalg.eigvals(U_el)))))
print("  （LU 的主元是主元，不是特徵值！）")

# %% [markdown]
# ### 不能對角化的矩陣：$GM<AM$
#
# * **代數重數 AM**：$\lambda$ 在特徵多項式中重複幾次
# * **幾何重數 GM**：$\dim N(A-\lambda I)$，即獨立特徵向量個數
#
# 永遠 $GM\le AM$。當某個 $\lambda$ 的 $GM<AM$ 時，特徵向量不夠，**無法對角化**。
# 這時要用 **Jordan 形式**（附錄 5）。
#
# 注意：**可逆性**看 $\lambda\ne0$，**可對角化**看特徵向量夠不夠 ── 兩件完全不同的事。

# %%
section("6.2 代數重數 vs 幾何重數")

cases = {
    "A = [[0,1],[0,0]]（λ=0,0）": np.array([[0.0, 1.0], [0.0, 0.0]]),
    "A = [[5,1],[0,5]]（λ=5,5）": np.array([[5.0, 1.0], [0.0, 5.0]]),
    "A = 5I（λ=5,5）": 5 * np.eye(2),
    "A = [[1,-1],[1,-1]]（λ=0,0）": np.array([[1.0, -1.0], [1.0, -1.0]]),
}
for name, M in cases.items():
    lam = np.linalg.eigvals(M)
    lam_round = np.round(np.real(lam), 10)
    uniq = np.unique(lam_round)
    desc = []
    for l in uniq:
        AM = int(np.sum(np.isclose(lam_round, l)))
        GM = M.shape[0] - np.linalg.matrix_rank(M - l * np.eye(M.shape[0]), tol=1e-10)
        desc.append(f"λ={l:g}: AM={AM}, GM={GM}")
    diagable = diagonalize(M) is not None
    print(f"  {name:<28} {'; '.join(desc):<28} 可對角化：{diagable}")
check("[[0,1],[0,0]] 不可對角化", diagonalize(np.array([[0.0, 1.0], [0.0, 0.0]])) is None)
check("5I 可對角化（GM = AM = 2）", diagonalize(5 * np.eye(2)) is not None)
print("\n可逆但不可對角化：[[5,1],[0,5]]（λ=5≠0 可逆，但只有一條特徵向量線）")
J = np.array([[5.0, 1.0], [0.0, 5.0]])
check("  可逆", abs(det_lu(J)) > 1e-12)
check("  不可對角化", diagonalize(J) is None)
print("不可逆但可對角化：[[1,0],[0,0]]")
D0 = np.array([[1.0, 0.0], [0.0, 0.0]])
check("  不可逆", abs(det_lu(D0)) < 1e-12)
check("  可對角化", diagonalize(D0) is not None)

# %% [markdown]
# ### 幾何圖像：特徵向量是「不轉向」的方向
#
# 下圖畫出單位圓上的點被 $A$ 映射後的位置。
# 只有沿著特徵向量的點，其像仍落在同一條直線上。

# %%
A = np.array([[2.0, 1.0], [1.0, 3.0]])
lam, Q = np.linalg.eigh(A)
fig, ax = new_axes("Eigenvectors keep their direction", figsize=(5.6, 5.2))
th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
circle = np.vstack([np.cos(th), np.sin(th)])
mapped = A @ circle
for k in range(circle.shape[1]):
    ax.plot([circle[0, k], mapped[0, k]], [circle[1, k], mapped[1, k]],
            color="C7", lw=0.7, alpha=0.7)
ax.plot(circle[0], circle[1], "C7o", ms=3)
ax.plot(mapped[0], mapped[1], "C0.", ms=5)
for k, c in zip(range(2), ["C3", "C2"]):
    q = Q[:, k]
    draw_vector(ax, q, f"x{k+1}", c)
    draw_vector(ax, lam[k] * q, f"{lam[k]:.2f}x{k+1}", c)
    ax.plot([-4 * q[0], 4 * q[0]], [-4 * q[1], 4 * q[1]], c, ls="--", lw=0.8)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
finish(fig, "ch06_eigenvector_directions")
print(f"λ = {lam}，特徵向量（正交，因為 A 對稱）=\n{Q}")
check("Ax = λx", A @ Q, Q @ np.diag(lam))
check("對稱矩陣的特徵向量正交", Q.T @ Q, np.eye(2))

# %% [markdown]
# ## 數值方法：實務上怎麼算特徵值
#
# **絕對不要**用 `det(A − λI) = 0` 去算（多項式求根數值極差）。實務做法：
#
# | 方法 | 得到什麼 | 關鍵想法 |
# |---|---|---|
# | 冪法 | 最大 $|\lambda|$ | 反覆 $\mathbf x\leftarrow A\mathbf x/\|A\mathbf x\|$ |
# | 反冪法（帶位移）| 最接近 $\sigma$ 的 $\lambda$ | 對 $(A-\sigma I)^{-1}$ 做冪法 |
# | QR 演算法 | 全部 $\lambda$ | $A_{k+1}=R_kQ_k$（相似變換）|
# | Jacobi 旋轉 | 對稱矩陣全部 $\lambda$ | 用 $2\times2$ 旋轉把非對角元素歸零 |
#
# 冪法就是 **PageRank** 與 Markov 鏈的演算法核心。

# %%
section("數值方法：冪法、反冪法、QR 演算法、Jacobi")

S = np.array([
    [4.0, 1.0, 0.0, 0.0],
    [1.0, 3.0, 1.0, 0.0],
    [0.0, 1.0, 2.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
])
true_lam = np.sort(np.linalg.eigvalsh(S))[::-1]
print(f"真實特徵值（numpy eigh）= {true_lam}\n")

lam_pow, v_pow, hist = power_iteration(S)
print(f"冪法：λ_max ≈ {lam_pow:.10f}（{len(hist)} 次迭代）")
check("冪法收斂到最大特徵值", lam_pow, true_lam[0], tol=1e-8)
check("得到的是真正的特徵向量", S @ v_pow, lam_pow * v_pow, tol=1e-6)
print(f"  收斂歷程（前 6 次）：{np.round(hist[:6], 6)}")

lam_inv, v_inv = inverse_power_iteration(S, shift=1.2)
print(f"\n反冪法（位移 σ = 1.2）：λ ≈ {lam_inv:.10f}")
check("收斂到最接近 1.2 的特徵值", lam_inv,
      true_lam[np.argmin(np.abs(true_lam - 1.2))], tol=1e-8)

lam_qr, _ = qr_algorithm(S)
print(f"\nQR 演算法：λ = {np.sort(lam_qr)[::-1]}")
check("QR 演算法得到全部特徵值", np.sort(lam_qr), np.sort(true_lam), tol=1e-6)

lam_jac, V_jac = jacobi_eigen(S)
print(f"Jacobi 旋轉：λ = {lam_jac}")
check("Jacobi 得到全部特徵值", np.sort(lam_jac), np.sort(true_lam), tol=1e-8)
check("Jacobi 的 V 正交", V_jac.T @ V_jac, np.eye(4), tol=1e-8)
check("S = VΛVᵀ", V_jac @ np.diag(lam_jac) @ V_jac.T, S, tol=1e-8)

print(f"\nRayleigh 商 xᵀSx/xᵀx 的範圍 = [λ_min, λ_max] = "
      f"[{true_lam[-1]:.4f}, {true_lam[0]:.4f}]")
rng = np.random.default_rng(0)
vals = [rayleigh_quotient(S, rng.standard_normal(4)) for _ in range(2000)]
check("隨機向量的 Rayleigh 商都落在這個範圍內",
      min(vals) >= true_lam[-1] - 1e-9 and max(vals) <= true_lam[0] + 1e-9)

# %% [markdown]
# ## 動手練習
#
# 1. Lucas 數列 $L_{k+2}=L_{k+1}+L_k$ 從 $L_1=1,L_2=3$ 開始。
#    證明 $L_{100}=\lambda_1^{100}+\lambda_2^{100}$。
# 2. 求 $A=5I-\mathbf 1\mathbf 1^{\mathsf T}$（$4\times4$）的特徵值、行列式與反矩陣。
#    提示：全 1 矩陣的秩是 1。
# 3. 寫一個函式判斷矩陣是否可對角化（比較每個 $\lambda$ 的 AM 與 GM）。
# 4. 用冪法估計一個隨機 Markov 矩陣的穩態分布，並與解 $(P-I)\mathbf x=\mathbf 0$ 的結果比較。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1：Lucas 數列
Fm = np.array([[1.0, 1.0], [1.0, 0.0]])
l1, l2 = float(np.max(np.linalg.eigvalsh(Fm))), float(np.min(np.linalg.eigvalsh(Fm)))
lucas = [1, 3]
while len(lucas) < 101:
    lucas.append(lucas[-1] + lucas[-2])
# 標準 Lucas 數列 L₀ = 2, L₁ = 1, L₂ = 3, ... 滿足 Lₙ = λ₁ⁿ + λ₂ⁿ
pred = l1 ** 100 + l2 ** 100               # 此處 lucas[0] = L₁，故 lucas[99] = L₁₀₀
print(f"練習 1：L₁₀₀ = {lucas[99]:,}")
print(f"        λ₁¹⁰⁰ + λ₂¹⁰⁰ = {pred:.6g}")
check("  相對誤差 < 1e-10", abs(pred - lucas[99]) / lucas[99] < 1e-10)

# 練習 2
A2 = 5 * np.eye(4) - np.ones((4, 4))
lam2 = np.sort(np.linalg.eigvalsh(A2))
print(f"\n練習 2：λ = {lam2}（即 1, 5, 5, 5）")
check("  λ = 1, 5, 5, 5", lam2, np.array([1.0, 5.0, 5.0, 5.0]))
check("  det = 125", det_lu(A2), 125.0, tol=1e-8)
A2inv_pred = (np.eye(4) + np.ones((4, 4))) / 5
check("  A⁻¹ = (I + 11ᵀ)/5", np.linalg.inv(A2), A2inv_pred)
H = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float) / 2
check("  Hadamard 矩陣 H 正交", H.T @ H, np.eye(4))
HtAH = H.T @ A2 @ H
check("  HᵀAH 是對角矩陣 ⇒ H 的欄就是特徵向量", HtAH, np.diag(np.diag(HtAH)), tol=1e-8)
print(f"  對角線（= 特徵值）= {np.diag(HtAH)}")


# 練習 3
def is_diagonalizable(A, tol=1e-9):
    """比較每個特徵值的 AM 與 GM。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    lam = np.linalg.eigvals(A)
    for l in np.unique(np.round(lam, 8)):
        AM = int(np.sum(np.isclose(np.round(lam, 8), l)))
        GM = n - np.linalg.matrix_rank(A - l * np.eye(n), tol=1e-8)
        if GM < AM:
            return False
    return True


print("\n練習 3：")
check("  [[0,1],[0,0]] 不可對角化", not is_diagonalizable([[0.0, 1.0], [0.0, 0.0]]))
check("  [[2,0],[0,3]] 可對角化", is_diagonalizable([[2.0, 0.0], [0.0, 3.0]]))
check("  與 diagonalize() 結果一致（隨機矩陣）",
      is_diagonalizable(rng.standard_normal((4, 4))))

# 練習 4
Pm = rng.uniform(0.1, 1.0, size=(4, 4))
Pm = Pm / Pm.sum(axis=0)                   # 每欄和為 1
steady_power = power_iteration(Pm)[1]
steady_power = np.abs(steady_power) / np.abs(steady_power).sum()
from linalg_tutorial.subspaces import nullspace
steady_null = nullspace(Pm - np.eye(4))[:, 0]
steady_null = steady_null / steady_null.sum()
print(f"\n練習 4：冪法穩態    = {np.round(steady_power, 6)}")
print(f"        零空間穩態  = {np.round(steady_null, 6)}")
check("  兩者一致", steady_power, steady_null, tol=1e-6)
check("  Pπ = π", Pm @ steady_power, steady_power, tol=1e-6)
check("  λ_max = 1（Perron–Frobenius）", power_iteration(Pm)[0], 1.0, tol=1e-8)

# %% [markdown]
# ## 本章重點回顧
#
# * $A\mathbf x=\lambda\mathbf x$ 找的是「只被拉伸、不被轉向」的方向。
#   $\lambda$ 由 $\det(A-\lambda I)=0$ 決定。
# * 永遠用 $\sum\lambda=\operatorname{trace}$、$\prod\lambda=\det$ 做檢查。
# * $A^k,A^{-1},A+cI$ 的特徵向量與 $A$ 相同，特徵值分別是 $\lambda^k,1/\lambda,\lambda+c$。
#   但 $\lambda(AB)\ne\lambda(A)\lambda(B)$（除非 $AB=BA$）。
# * $A=X\Lambda X^{-1}$ 讓 $A^k=X\Lambda^kX^{-1}$：Fibonacci、Markov 穩態、
#   差分方程全部一次解決。
# * 相似矩陣 $B^{-1}AB$ 特徵值不變（但消去法會改變特徵值！）。
# * $GM<AM$ 時無法對角化；可逆性與可對角化性是兩件獨立的事。
# * 實務上用冪法／反冪法／QR 演算法／Jacobi，不用特徵多項式。
#
# 下一章：對稱矩陣的特徵值全為實數、特徵向量互相正交 ── 線性代數最美的定理。
