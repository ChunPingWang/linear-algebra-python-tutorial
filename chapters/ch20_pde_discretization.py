# %% [markdown]
# # 第 20 章　偏微分方程的離散化與 Kronecker 積
#
# > 對應 Riley 第 20 章（PDE 的一般解與特殊解：波方程、擴散方程、特徵線）、
# > 第 21 章（分離變數、疊加、極座標、積分變換、Green 函數）、
# > 第 27.8 節（PDE 的數值解），並對照本教材第 2 章（$K$ 矩陣）、
# > 第 8 章（熱傳導與波方程）、第 17 章（Sturm–Liouville）、第 19 章（迭代法）。
#
# 把空間離散化後，PDE 就變成**大型稀疏矩陣**問題：
#
# | PDE | 離散後 | 線性代數問題 |
# |---|---|---|
# | $-\nabla^2u=f$（Poisson）| $A\mathbf u=\mathbf f$ | 解稀疏線性系統 |
# | $u_t=\nabla^2u$（熱傳導）| $\mathbf u'=-A\mathbf u$ | 矩陣指數 / 穩定性 |
# | $u_{tt}=\nabla^2u$（波）| $\mathbf u''=-A\mathbf u$ | 特徵值 = 頻率² |
#
# 而二維的 Laplacian 有極漂亮的結構 ── **Kronecker 和**：
#
# $$A_{2D}=K\otimes I+I\otimes K,\qquad
# \lambda_{ij}=\lambda_i+\lambda_j,\qquad
# \mathbf v_{ij}=\mathbf v_i\otimes\mathbf v_j$$
#
# 「分離變數」在線性代數裡就是**張量積的特徵向量**。
#
# ```bash
# python chapters/ch20_pde_discretization.py
# python tools/build_notebooks.py ch20
# ```

# %%
import pathlib
import sys
import time

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.eigen import is_positive_definite, matrix_exp_by_eigen
from linalg_tutorial.elimination import second_difference_matrix
from linalg_tutorial.svd_tools import condition_number
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 20.1　Kronecker 積的代數
#
# $$A\otimes B=\begin{bmatrix}
# a_{11}B&\cdots&a_{1n}B\\\vdots&&\vdots\\a_{m1}B&\cdots&a_{mn}B\end{bmatrix}$$
#
# 必記的四條性質：
#
# 1. $(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$
# 2. $(A\otimes B)^{\mathsf T}=A^{\mathsf T}\otimes B^{\mathsf T}$
# 3. **特徵值相乘**：$\lambda(A\otimes B)=\lambda_i(A)\mu_j(B)$，
#    特徵向量 $\mathbf v_i\otimes\mathbf w_j$
# 4. **vec 技巧**：$(B^{\mathsf T}\otimes A)\operatorname{vec}(X)=\operatorname{vec}(AXB)$
#
# 由 3 可得 $\det(A\otimes B)=(\det A)^{n}(\det B)^{m}$、
# $\operatorname{tr}(A\otimes B)=\operatorname{tr}A\cdot\operatorname{tr}B$。

# %%
section("20.1 Kronecker 積的性質")

rng = np.random.default_rng(0)
A = rng.standard_normal((3, 3))
B = rng.standard_normal((4, 4))
C = rng.standard_normal((3, 3))
D = rng.standard_normal((4, 4))

AB = np.kron(A, B)
print(f"A 是 3x3，B 是 4x4 ⇒ A⊗B 是 {AB.shape}")
check("① (A⊗B)(C⊗D) = (AC)⊗(BD)", AB @ np.kron(C, D), np.kron(A @ C, B @ D))
check("② (A⊗B)ᵀ = Aᵀ⊗Bᵀ", AB.T, np.kron(A.T, B.T))
lam_A = np.linalg.eigvals(A)
lam_B = np.linalg.eigvals(B)
lam_kron = np.sort_complex(np.linalg.eigvals(AB))
lam_prod = np.sort_complex(np.array([a * b for a in lam_A for b in lam_B]))
check("③ λ(A⊗B) = λᵢ(A)·μⱼ(B)", np.sort(np.abs(lam_kron)), np.sort(np.abs(lam_prod)),
      tol=1e-8)
check("   trace(A⊗B) = trace(A)·trace(B)", np.trace(AB),
      np.trace(A) * np.trace(B), tol=1e-10)
check("   det(A⊗B) = (det A)⁴(det B)³", np.linalg.det(AB),
      np.linalg.det(A) ** 4 * np.linalg.det(B) ** 3, tol=1e-6)
check("   rank(A⊗B) = rank(A)·rank(B)",
      np.linalg.matrix_rank(AB), np.linalg.matrix_rank(A) * np.linalg.matrix_rank(B))

# vec 技巧：把矩陣方程變成向量方程
X = rng.standard_normal((4, 3))
vec = lambda M: M.flatten(order="F")          # 按欄堆疊
A4 = rng.standard_normal((4, 4))
B3 = rng.standard_normal((3, 3))
check("④ vec(AXB) = (Bᵀ⊗A)vec(X)", vec(A4 @ X @ B3), np.kron(B3.T, A4) @ vec(X))
print("  ⇒ 任何「矩陣方程」AXB = C 都能寫成普通的線性系統（代價是維度變成 mn）")
# Sylvester 方程 AX + XB = C
A2 = rng.standard_normal((3, 3))
B2 = rng.standard_normal((4, 4))
C2 = rng.standard_normal((3, 4))
S_syl = np.kron(np.eye(4), A2) + np.kron(B2.T, np.eye(3))
x_syl = np.linalg.solve(S_syl, vec(C2))
X_syl = x_syl.reshape((3, 4), order="F")
check("Sylvester 方程 AX + XB = C 可用 Kronecker 和求解",
      A2 @ X_syl + X_syl @ B2, C2, tol=1e-8)
print("  （控制理論的 Lyapunov 方程就是這一型）")

# %% [markdown]
# ## 20.2　二維 Laplacian = Kronecker 和
#
# 在 $n\times n$ 的方格上用五點差分近似 $-\nabla^2$：
#
# $$-\nabla^2u\approx\frac{1}{h^2}
# \left(4u_{ij}-u_{i-1,j}-u_{i+1,j}-u_{i,j-1}-u_{i,j+1}\right)$$
#
# 寫成矩陣就是 **Kronecker 和**
#
# $$A=\frac{1}{h^2}\left(K\otimes I+I\otimes K\right)$$
#
# 於是（這正是「分離變數」）：
#
# $$\lambda_{pq}=\frac{1}{h^2}\left(\lambda_p+\lambda_q\right),\qquad
# \mathbf v_{pq}=\mathbf v_p\otimes\mathbf v_q
# \ \ \left(\sin\frac{p\pi x}{L}\sin\frac{q\pi y}{L}\right)$$

# %%
section("20.2 五點差分 Laplacian 的 Kronecker 結構")

n = 5
h = 1.0 / (n + 1)
K = second_difference_matrix(n)
A2D = (np.kron(K, np.eye(n)) + np.kron(np.eye(n), K)) / h ** 2
print(f"n = {n} ⇒ 未知數 {n * n} 個，矩陣 {A2D.shape}")
print(f"非零元素 = {np.count_nonzero(A2D)}（佔 "
      f"{np.count_nonzero(A2D) / A2D.size:.1%}）⇒ 稀疏！")
check("A 對稱", A2D, A2D.T)
check("A 正定", is_positive_definite(A2D))
check("每列最多 5 個非零（五點差分）",
      np.max(np.count_nonzero(A2D, axis=1)) <= 5)

# 直接用五點差分組一次，確認與 Kronecker 和相同
A_direct = np.zeros((n * n, n * n))
idx = lambda i, j: i * n + j
for i in range(n):
    for j in range(n):
        A_direct[idx(i, j), idx(i, j)] = 4 / h ** 2
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n:
                A_direct[idx(i, j), idx(ii, jj)] = -1 / h ** 2
check("五點差分 = K⊗I + I⊗K", A_direct, A2D)

# 特徵值與特徵向量的張量結構
lam_1d, V_1d = np.linalg.eigh(K)
lam_2d_theory = np.sort(np.array([(lam_1d[p] + lam_1d[q]) / h ** 2
                                  for p in range(n) for q in range(n)]))
check("λ_pq = (λ_p + λ_q)/h²（Kronecker 和的特徵值相加）",
      np.sort(np.linalg.eigvalsh(A2D)), lam_2d_theory, tol=1e-9)

p_test, q_test = 1, 2
v_tensor = np.kron(V_1d[:, p_test], V_1d[:, q_test])
lam_tensor = (lam_1d[p_test] + lam_1d[q_test]) / h ** 2
check("特徵向量 = v_p ⊗ v_q（分離變數！）", A2D @ v_tensor, lam_tensor * v_tensor)

# 與連續解析解比較
# 離散化誤差是 O(h²)，網格太粗時高頻模態誤差很大，所以改用 n = 20 來比較
n_fine = 20
h_fine = 1.0 / (n_fine + 1)
K_fine = second_difference_matrix(n_fine)
A2D_fine = (np.kron(K_fine, np.eye(n_fine))
            + np.kron(np.eye(n_fine), K_fine)) / h_fine ** 2
xf = np.linspace(h_fine, 1 - h_fine, n_fine)
XXf, YYf = np.meshgrid(xf, xf, indexing="ij")
print(f"\n網格 n = {n_fine}（h = {h_fine:.4f}）下的模態：")
print(f"{'(p,q)':>8} | {'數值 λ':>14} | {'理論 (p²+q²)π²':>16} | {'相對誤差':>10}")
print("-" * 56)
for p, q in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3)]:
    mode = np.sin(p * np.pi * XXf) * np.sin(q * np.pi * YYf)
    v = mode.flatten()
    v = v / np.linalg.norm(v)
    lam_num = v @ A2D_fine @ v
    lam_th = (p ** 2 + q ** 2) * np.pi ** 2
    print(f"{str((p, q)):>8} | {lam_num:>14.4f} | {lam_th:>16.4f} | "
          f"{abs(lam_num - lam_th) / lam_th:>9.2%}")
    check(f"  模態 ({p},{q}) 的 λ ≈ (p²+q²)π²（誤差 O(h²)）",
          abs(lam_num - lam_th) / lam_th < 0.05)
print("  ⇒ 二維 Laplacian 的特徵函數就是 sin(pπx)sin(qπy)（分離變數的答案）")

# 稀疏度與頻寬
print(f"\n{'n':>5} | {'未知數 n²':>10} | {'非零元素':>10} | {'稀疏度':>10} | "
      f"{'頻寬':>6} | {'κ(A)':>10}")
print("-" * 62)
for n_ in [5, 10, 20, 40]:
    h_ = 1.0 / (n_ + 1)
    K_ = second_difference_matrix(n_)
    A_ = (np.kron(K_, np.eye(n_)) + np.kron(np.eye(n_), K_)) / h_ ** 2
    nnz = np.count_nonzero(A_)
    band = max(abs(i - j) for i, j in zip(*np.nonzero(A_)))
    print(f"{n_:>5} | {n_ ** 2:>10} | {nnz:>10} | {nnz / A_.size:>9.2%} | "
          f"{band:>6} | {condition_number(A_):>10.1f}")
    check(f"  n = {n_}：非零 ≈ 5n²", nnz <= 5 * n_ ** 2)
    check(f"  n = {n_}：頻寬 = n", band, n_)
print("  ⇒ 稀疏但「頻寬 = n」：LU 分解會產生 fill-in，所以大問題要用迭代法（第 19 章）")

# %% [markdown]
# ### 快速 Poisson 解法：用 DST 對角化
#
# 因為 $K$ 的特徵向量是**離散正弦波**，所以
# 離散正弦變換（DST）把 $A$ 完全對角化：
#
# $$A=\left(S\otimes S\right)\Lambda\left(S\otimes S\right)^{-1}$$
#
# 於是解 Poisson 方程只要三步：**變換 → 逐點除以 $\lambda$ → 反變換**，
# 成本 $O(n^2\log n)$ ── 比 $O(n^6)$ 的稠密消去法快到無法比較。

# %%
section("20.2 DST 快速 Poisson 解法")

n = 32
h = 1.0 / (n + 1)
x_grid = np.linspace(h, 1 - h, n)
XX, YY = np.meshgrid(x_grid, x_grid, indexing="ij")
f_src = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY) + \
    0.5 * np.sin(3 * np.pi * XX) * np.sin(np.pi * YY)

# 正弦變換矩陣（也可以用 FFT 加速成 O(n log n)）
j_idx = np.arange(1, n + 1)
S = np.sqrt(2 / (n + 1)) * np.sin(np.outer(j_idx, j_idx) * np.pi / (n + 1))
check("S 是正交矩陣（S² = I）", S @ S, np.eye(n), tol=1e-10)
lam_K = 2 - 2 * np.cos(j_idx * np.pi / (n + 1))
check("S 對角化 K", S @ second_difference_matrix(n) @ S, np.diag(lam_K), tol=1e-10)

# 快速解法
t0 = time.perf_counter()
F_hat = S @ f_src @ S                                   # 二維正弦變換
denom = (lam_K[:, None] + lam_K[None, :]) / h ** 2      # λ_pq
U_hat = F_hat / denom
u_fast = S @ U_hat @ S
t_fast = time.perf_counter() - t0

# 直接法比較
K_full = second_difference_matrix(n)
A_full = (np.kron(K_full, np.eye(n)) + np.kron(np.eye(n), K_full)) / h ** 2
t0 = time.perf_counter()
u_direct = np.linalg.solve(A_full, f_src.flatten()).reshape(n, n)
t_direct = time.perf_counter() - t0

print(f"n = {n}（{n*n} 個未知數）")
print(f"  DST 快速解法：{t_fast * 1000:.2f} ms")
print(f"  稠密直接解法：{t_direct * 1000:.2f} ms（快 {t_direct / t_fast:.0f} 倍）")
check("兩種解法答案相同", u_fast, u_direct, tol=1e-9)

# 與解析解比較：−∇²u = sin(pπx)sin(qπy) ⇒ u = sin/( (p²+q²)π² )
u_exact = (np.sin(np.pi * XX) * np.sin(2 * np.pi * YY) / ((1 + 4) * np.pi ** 2)
           + 0.5 * np.sin(3 * np.pi * XX) * np.sin(np.pi * YY)
           / ((9 + 1) * np.pi ** 2))
rel_err = np.max(np.abs(u_fast - u_exact)) / np.max(np.abs(u_exact))
print(f"  與解析解的相對誤差 = {rel_err:.3%}（O(h²) 的離散化誤差）")
check("數值解接近解析解", rel_err < 0.01)

print(f"\n{'n':>5} | {'未知數':>10} | {'DST (ms)':>10} | {'稠密 (ms)':>12} | {'加速':>8}")
print("-" * 54)
for n_ in [8, 16, 32]:
    h_ = 1.0 / (n_ + 1)
    jj = np.arange(1, n_ + 1)
    S_ = np.sqrt(2 / (n_ + 1)) * np.sin(np.outer(jj, jj) * np.pi / (n_ + 1))
    lamK_ = 2 - 2 * np.cos(jj * np.pi / (n_ + 1))
    xg = np.linspace(h_, 1 - h_, n_)
    Xg, Yg = np.meshgrid(xg, xg, indexing="ij")
    f_ = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)
    t0 = time.perf_counter()
    u_f = S_ @ ((S_ @ f_ @ S_) / ((lamK_[:, None] + lamK_[None, :]) / h_ ** 2)) @ S_
    t_f = time.perf_counter() - t0
    K2 = second_difference_matrix(n_)
    A_ = (np.kron(K2, np.eye(n_)) + np.kron(np.eye(n_), K2)) / h_ ** 2
    t0 = time.perf_counter()
    u_d = np.linalg.solve(A_, f_.flatten()).reshape(n_, n_)
    t_d = time.perf_counter() - t0
    print(f"{n_:>5} | {n_ ** 2:>10} | {t_f * 1000:>10.2f} | {t_d * 1000:>12.2f} | "
          f"{t_d / t_f:>7.0f}x")
    check(f"  n = {n_}：兩法一致", u_f, u_d, tol=1e-9)

fig = plt.figure(figsize=(11.0, 3.4))
for k, (title, data) in enumerate([("source f", f_src), ("solution u", u_fast),
                                   ("error (u - exact)", u_fast - u_exact)]):
    ax = fig.add_subplot(1, 3, k + 1)
    im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
finish(fig, "ch20_poisson")

# %% [markdown]
# ## 20.3　時間離散化與穩定性（CFL 條件）
#
# 熱傳導 $u_t=u_{xx}$ 離散成 $\mathbf u'=-A\mathbf u$（$A=K/h^2$），再對時間離散：
#
# | 方法 | 公式 | 放大因子 | 穩定條件 |
# |---|---|---|---|
# | 顯式（前向）| $\mathbf u^{k+1}=(I-\Delta tA)\mathbf u^k$ | $1-\Delta t\lambda$ | $\Delta t\le\dfrac{h^2}{2}$ |
# | 隱式（後向）| $(I+\Delta tA)\mathbf u^{k+1}=\mathbf u^k$ | $\dfrac{1}{1+\Delta t\lambda}$ | **無條件穩定** |
# | Crank–Nicolson | 平均 | $\dfrac{1-\Delta t\lambda/2}{1+\Delta t\lambda/2}$ | 無條件穩定、二階 |
#
# 穩定性完全由**特徵值**決定：$|\text{放大因子}|\le1$ 對所有 $\lambda$。
# 這就是第 8 章「前向/後向/中央差分」的 PDE 版本。

# %%
section("20.3 熱傳導方程的三種時間離散")

n = 20
h = 1.0 / (n + 1)
A_heat = second_difference_matrix(n) / h ** 2
lam_heat = np.linalg.eigvalsh(A_heat)
lam_max = lam_heat[-1]
dt_crit = 2 / lam_max
print(f"n = {n}：λ_max = {lam_max:.2f}，顯式法的穩定界限 Δt ≤ 2/λ_max = "
      f"{dt_crit:.3e}")
print(f"  理論 h²/2 = {h ** 2 / 2:.3e}（λ_max ≈ 4/h²）")
check("Δt_crit ≈ h²/2", dt_crit, h ** 2 / 2, tol=h ** 2 * 0.05)

print(f"\n{'方法':<18} | {'Δt':>12} | {'最大放大因子':>14} | {'穩定?':>6}")
print("-" * 58)
for dt in [0.4 * dt_crit, 1.1 * dt_crit]:
    for name, amp in [("顯式（前向）", lambda l: abs(1 - dt * l)),
                      ("隱式（後向）", lambda l: abs(1 / (1 + dt * l))),
                      ("Crank–Nicolson", lambda l: abs((1 - dt * l / 2) / (1 + dt * l / 2)))]:
        max_amp = max(amp(l) for l in lam_heat)
        print(f"{name:<18} | {dt:>12.3e} | {max_amp:>14.6f} | "
              f"{str(max_amp <= 1 + 1e-12):>6}")
        if name.startswith("隱式") or name.startswith("Crank"):
            check(f"  {name}（Δt = {dt:.1e}）：無條件穩定", max_amp <= 1 + 1e-12)
    print()
check("顯式法：Δt < Δt_crit 時穩定",
      max(abs(1 - 0.4 * dt_crit * l) for l in lam_heat) <= 1)
check("顯式法：Δt > Δt_crit 時不穩定",
      max(abs(1 - 1.1 * dt_crit * l) for l in lam_heat) > 1)

# 實際跑一下
# 初始條件要含有高頻成分，才看得到不穩定（純低頻模態不會激發 λ_max）
rng_heat = np.random.default_rng(0)
u0 = (np.where(np.abs(np.linspace(h, 1 - h, n) - 0.5) < 0.1, 1.0, 0.0)
      + 0.05 * rng_heat.standard_normal(n))
print("實際演化（t = 0.05）：")
for dt_factor, tag in [(0.4, "穩定"), (1.1, "不穩定")]:
    dt = dt_factor * dt_crit
    steps = int(0.05 / dt)
    u_exp = u0.copy()
    G = np.eye(n) - dt * A_heat
    for _ in range(steps):
        u_exp = G @ u_exp
    u_imp = u0.copy()
    G_imp = np.linalg.inv(np.eye(n) + dt * A_heat)
    for _ in range(steps):
        u_imp = G_imp @ u_imp
    u_true = np.real(matrix_exp_by_eigen(-A_heat, 0.05) @ u0)
    print(f"  Δt = {dt_factor}·Δt_crit（{steps} 步）：顯式 ‖u‖ = "
          f"{np.linalg.norm(u_exp):.3e}，隱式 ‖u‖ = {np.linalg.norm(u_imp):.3e}"
          f"，精確 ‖u‖ = {np.linalg.norm(u_true):.3e}  [{tag}]")
    if dt_factor < 1:
        check(f"  Δt = {dt_factor}Δt_crit：顯式解接近精確解", u_exp, u_true,
              tol=0.05)
    else:
        growth = np.linalg.norm(u_exp) / np.linalg.norm(u_true)
        check(f"  Δt = {dt_factor}Δt_crit：顯式解爆掉（放大 {growth:.0f} 倍）",
              growth > 10)
    check(f"  Δt = {dt_factor}Δt_crit：隱式解始終有界",
          np.linalg.norm(u_imp) <= np.linalg.norm(u0) + 1e-12)

# 精度比較
print(f"\n時間精度（固定 t = 0.05，看 Δt 減半時誤差如何變）：")
print(f"{'Δt':>12} | {'隱式誤差':>12} | {'CN 誤差':>12} | {'隱式比':>8} | {'CN 比':>8}")
print("-" * 60)
u_ref = np.real(matrix_exp_by_eigen(-A_heat, 0.05) @ u0)
prev = None
for dt in [1e-3, 5e-4, 2.5e-4]:
    steps = int(round(0.05 / dt))
    G_imp = np.linalg.inv(np.eye(n) + dt * A_heat)
    G_cn = np.linalg.solve(np.eye(n) + dt * A_heat / 2, np.eye(n) - dt * A_heat / 2)
    u_i, u_c = u0.copy(), u0.copy()
    for _ in range(steps):
        u_i = G_imp @ u_i
        u_c = G_cn @ u_c
    e_i = np.linalg.norm(u_i - u_ref)
    e_c = np.linalg.norm(u_c - u_ref)
    r_i = f"{prev[0] / e_i:.2f}" if prev else "—"
    r_c = f"{prev[1] / e_c:.2f}" if prev else "—"
    print(f"{dt:>12.1e} | {e_i:>12.3e} | {e_c:>12.3e} | {r_i:>8} | {r_c:>8}")
    prev = (e_i, e_c)
print("  隱式是一階（誤差比 ≈ 2），Crank–Nicolson 是二階（誤差比 ≈ 4）")

# %%
section("20.3 波方程的 CFL 條件")

# u_tt = c²u_xx ⇒ 顯式差分的穩定條件 cΔt/h ≤ 1
c_wave = 1.0
n = 40
h = 1.0 / (n + 1)
A_wave = second_difference_matrix(n) / h ** 2
lam_w = np.linalg.eigvalsh(A_wave)
print(f"n = {n}：h = {h:.5f}")
print(f"{'CFL = cΔt/h':>14} | {'最大 |放大因子|':>16} | {'穩定?':>6}")
print("-" * 44)
for cfl in [0.5, 0.9, 1.0, 1.05]:
    dt = cfl * h / c_wave
    # leapfrog: u^{k+1} = 2u^k − u^{k−1} − (cΔt)²A u^k ⇒ 放大因子由 2x2 矩陣決定
    max_amp = 0.0
    for l in lam_w:
        M = np.array([[2 - dt ** 2 * c_wave ** 2 * l, -1.0], [1.0, 0.0]])
        max_amp = max(max_amp, np.max(np.abs(np.linalg.eigvals(M))))
    print(f"{cfl:>14.2f} | {max_amp:>16.8f} | {str(max_amp <= 1 + 1e-10):>6}")
    check(f"  CFL = {cfl}：穩定性與理論一致", (max_amp <= 1 + 1e-10) == (cfl <= 1.0))
print("  ⇒ CFL 條件 cΔt ≤ h：訊號一步不能跑超過一格（物理直覺！）")
print("  理論：λ_max = 4/h²，穩定要求 (cΔt)²λ_max ≤ 4 ⇒ cΔt ≤ h")
check("λ_max·h²/4 ≈ 1", lam_w[-1] * h ** 2 / 4, 1.0, tol=0.01)

# 真的跑波方程
n = 100
h = 1.0 / (n + 1)
xg = np.linspace(h, 1 - h, n)
A_w = second_difference_matrix(n) / h ** 2
u_init = np.exp(-200 * (xg - 0.3) ** 2)
dt = 0.9 * h
steps = int(0.3 / dt)
u_prev = u_init.copy()
u_cur = u_init - 0.5 * dt ** 2 * (A_w @ u_init)       # 初速為 0
energy0 = None
energies = []
for k in range(steps):
    u_next = 2 * u_cur - u_prev - dt ** 2 * (A_w @ u_cur)
    v = (u_next - u_prev) / (2 * dt)
    energies.append(0.5 * v @ v + 0.5 * u_cur @ A_w @ u_cur)
    u_prev, u_cur = u_cur, u_next
energies = np.array(energies)
print(f"\n波方程（CFL = 0.9，{steps} 步）：能量變化 = "
      f"{(energies.max() - energies.min()) / energies.mean():.3%}")
check("leapfrog 近似守恆能量（變化 < 2%）",
      (energies.max() - energies.min()) / energies.mean() < 0.02)

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
u_p, u_c = u_init.copy(), u_init - 0.5 * dt ** 2 * (A_w @ u_init)
snapshots = {0: u_init.copy()}
for k in range(1, steps + 1):
    u_n = 2 * u_c - u_p - dt ** 2 * (A_w @ u_c)
    u_p, u_c = u_c, u_n
    if k in (steps // 3, 2 * steps // 3, steps):
        snapshots[k] = u_c.copy()
for k, u_snap in snapshots.items():
    ax.plot(xg, u_snap, lw=1.5, label=f"t = {k * dt:.3f}")
ax.set_title("Wave equation (leapfrog, CFL = 0.9)", fontsize=9)
ax.set_xlabel("x")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
for cfl, c in [(0.5, "C0"), (0.9, "C1"), (1.02, "C3")]:
    dt_ = cfl * h
    amps = []
    for l in lam_w:
        M = np.array([[2 - dt_ ** 2 * l, -1.0], [1.0, 0.0]])
        amps.append(np.max(np.abs(np.linalg.eigvals(M))))
    ax.plot(np.sqrt(lam_w) * h, amps, c, lw=1.6, label=f"CFL = {cfl}")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("normalized wavenumber")
ax.set_ylabel("|amplification|")
ax.set_title("Stability: CFL <= 1", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
finish(fig, "ch20_cfl")

# %% [markdown]
# ## 20.4　分離變數 = 張量積的特徵向量
#
# Riley 第 21 章用「分離變數」解 PDE：假設 $u=X(x)Y(y)T(t)$，
# 代入後得到各自的 ODE 與**特徵值條件**。
#
# 線性代數的翻譯：
#
# * 分離變數 $\iff$ 尋找 $A_{2D}=K\otimes I+I\otimes K$ 的**張量積特徵向量**
# * 「本徵值條件」$\iff$ $\lambda_{pq}=\lambda_p+\lambda_q$
# * 「疊加解」$\iff$ 用特徵向量基底展開初始條件
#
# 下面完整解一個二維熱傳導問題：展開 → 各模態獨立衰減 → 重組。

# %%
section("20.4 二維熱傳導：模態獨立衰減")

n = 24
h = 1.0 / (n + 1)
xg = np.linspace(h, 1 - h, n)
XX, YY = np.meshgrid(xg, xg, indexing="ij")
jj = np.arange(1, n + 1)
S = np.sqrt(2 / (n + 1)) * np.sin(np.outer(jj, jj) * np.pi / (n + 1))
lam_1d = (2 - 2 * np.cos(jj * np.pi / (n + 1))) / h ** 2
Lam_2d = lam_1d[:, None] + lam_1d[None, :]

u0_2d = np.where((np.abs(XX - 0.3) < 0.15) & (np.abs(YY - 0.6) < 0.15), 1.0, 0.0)
coef0 = S @ u0_2d @ S                      # 展開成 sin⊗sin 模態
heat_2d = lambda t: S @ (coef0 * np.exp(-Lam_2d * t)) @ S

check("t = 0 時還原初始條件", heat_2d(0.0), u0_2d, tol=1e-10)
print(f"{'t':>8} | {'最高溫':>10} | {'總熱量':>12} | {'主導模態 (1,1) 的佔比':>22}")
print("-" * 60)
for t in [0.0, 0.001, 0.01, 0.05, 0.2]:
    u_t = heat_2d(t)
    coef_t = coef0 * np.exp(-Lam_2d * t)
    frac = coef_t[0, 0] ** 2 / np.sum(coef_t ** 2)
    print(f"{t:>8} | {u_t.max():>10.6f} | {u_t.sum() * h ** 2:>12.6f} | "
          f"{frac:>21.2%}")
check("長時間後只剩最低模態 (1,1)",
      (coef0 * np.exp(-Lam_2d * 1.0))[0, 0] ** 2
      / np.sum((coef0 * np.exp(-Lam_2d * 1.0)) ** 2) > 0.99)
print(f"  最慢衰減率 = λ₁₁ = {Lam_2d[0, 0]:.4f}（理論 2π² = {2 * np.pi ** 2:.4f}）")
check("λ₁₁ ≈ 2π²", Lam_2d[0, 0], 2 * np.pi ** 2, tol=0.5)

# 驗證滿足熱方程
A_2d_op = lambda U: (second_difference_matrix(n) @ U + U @ second_difference_matrix(n)) / h ** 2
dt_num = 1e-6
t0 = 0.01
du_dt = (heat_2d(t0 + dt_num) - heat_2d(t0 - dt_num)) / (2 * dt_num)
check("解滿足 ∂u/∂t = ∇²u（即 −A u）", du_dt, -A_2d_op(heat_2d(t0)), tol=1e-3)

fig = plt.figure(figsize=(12.0, 3.0))
for k, t in enumerate([0.0, 0.002, 0.01, 0.05, 0.2]):
    ax = fig.add_subplot(1, 5, k + 1)
    ax.imshow(heat_2d(t).T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno",
              vmin=0, vmax=1)
    ax.set_title(f"t = {t}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
finish(fig, "ch20_heat2d")

# %% [markdown]
# ### 極座標：分離變數給出 Bessel 函數
#
# 圓形區域上的 $-\nabla^2u=\lambda u$ 用極座標分離變數
# $u=R(r)\Theta(\theta)$ 得到
#
# $$\Theta''=-m^2\Theta,\qquad
# -(rR')'+\frac{m^2}{r}R=\lambda rR\ \ (\text{Bessel 的 SL 形式！})$$
#
# 所以圓形鼓的頻率就是 $J_m$ 的零點 ── 第 17、18 章的結論在此派上用場。

# %%
section("20.4 圓形鼓的頻率 = Bessel 零點")

try:
    from scipy.special import jn_zeros

    # 用極座標網格離散化徑向的 SL 問題（固定邊界 r = 1）
    def radial_eigenvalues(m, n_r=400):
        """−(rR')' + m²R/r = λ rR，R(1) = 0，在 r = 0 處有界。"""
        h_r = 1.0 / (n_r + 1)
        r = h_r * np.arange(1, n_r + 1)
        r_half = h_r * (np.arange(0, n_r + 1) + 0.5)
        K = np.zeros((n_r, n_r))
        for i in range(n_r):
            K[i, i] = (r_half[i] + r_half[i + 1]) / h_r ** 2 + m ** 2 / r[i]
            if i > 0:
                K[i, i - 1] = -r_half[i] / h_r ** 2
            if i < n_r - 1:
                K[i, i + 1] = -r_half[i + 1] / h_r ** 2
        M = np.diag(r)
        from linalg_tutorial.eigen import generalized_symmetric_eig
        return np.sort(generalized_symmetric_eig(K, M)[0])

    print(f"{'m':>4} | {'數值 λ（前 3 個）':<34} | {'J_m 零點的平方':<34}")
    print("-" * 78)
    for m in [1, 2]:
        lam_r = radial_eigenvalues(m)[:3]
        zeros_m = jn_zeros(m, 3)
        print(f"{m:>4} | {str(np.round(lam_r, 3)):<34} | "
              f"{str(np.round(zeros_m ** 2, 3)):<34}")
        for k in range(2):
            check(f"  m = {m}：λ{k+1} ≈ (j_{{{m},{k+1}}})²", lam_r[k],
                  zeros_m[k] ** 2, tol=zeros_m[k] ** 2 * 0.02)
    print("  （m = 0 在 r = 0 是自然邊界條件 R'(0) = 0，需要不同的離散化處理）")
    print("  ⇒ 圓形鼓的振動頻率 ω = j_{m,k}（Bessel 零點）—— 鼓聲不是泛音列！")
    z01, z11 = jn_zeros(0, 1)[0], jn_zeros(1, 1)[0]
    print(f"  最低兩個頻率比 = j₁₁/j₀₁ = {z11 / z01:.6f}（不是整數倍 ⇒ 鼓聲「不和諧」）")
    check("頻率比不是整數", abs(z11 / z01 - round(z11 / z01)) > 0.1)
except ImportError:
    print("  （未安裝 scipy，略過 Bessel 零點的比較）")

# %% [markdown]
# ## 動手練習
#
# 1. 驗證 $(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}$，並說明為什麼。
# 2. 用 Kronecker 和寫出三維 Laplacian，確認特徵值是 $\lambda_p+\lambda_q+\lambda_r$。
# 3. 對 $n=64$ 的二維 Poisson 問題，比較「稠密 LU」「CG」「DST」三種解法的時間。
# 4. 把熱傳導的顯式法 $\Delta t$ 設在臨界值附近，觀察解如何開始震盪。
# 5. 用 vec 技巧把 Lyapunov 方程 $AX+XA^{\mathsf T}=-Q$ 轉成線性系統並求解。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A5 = rng.standard_normal((3, 3))
B5 = rng.standard_normal((4, 4))
check("練習 1：(A⊗B)⁻¹ = A⁻¹⊗B⁻¹",
      np.linalg.inv(np.kron(A5, B5)), np.kron(np.linalg.inv(A5), np.linalg.inv(B5)),
      tol=1e-8)
print("  原因：(A⊗B)(A⁻¹⊗B⁻¹) = (AA⁻¹)⊗(BB⁻¹) = I⊗I = I（性質 ①）")

# 練習 2：三維 Laplacian
n2 = 4
K2_ = second_difference_matrix(n2)
I2 = np.eye(n2)
A3D = (np.kron(np.kron(K2_, I2), I2) + np.kron(np.kron(I2, K2_), I2)
       + np.kron(np.kron(I2, I2), K2_))
print(f"\n練習 2：三維 Laplacian 的大小 = {A3D.shape}（n³ = {n2 ** 3}）")
lam1d = np.linalg.eigvalsh(K2_)
lam3d_theory = np.sort(np.array([lam1d[p] + lam1d[q] + lam1d[r]
                                 for p in range(n2) for q in range(n2)
                                 for r in range(n2)]))
check("  λ = λ_p + λ_q + λ_r", np.sort(np.linalg.eigvalsh(A3D)), lam3d_theory,
      tol=1e-9)
check("  矩陣對稱正定", is_positive_definite(A3D))
print(f"  非零比例 = {np.count_nonzero(A3D) / A3D.size:.2%}"
      f"（每列最多 7 個非零：七點差分）")
check("  每列最多 7 個非零", np.max(np.count_nonzero(A3D, axis=1)) <= 7)

# 練習 3
print("\n練習 3：三種解法的時間（n = 48）")
n3 = 48
h3 = 1.0 / (n3 + 1)
jj3 = np.arange(1, n3 + 1)
S3 = np.sqrt(2 / (n3 + 1)) * np.sin(np.outer(jj3, jj3) * np.pi / (n3 + 1))
lam3 = (2 - 2 * np.cos(jj3 * np.pi / (n3 + 1))) / h3 ** 2
xg3 = np.linspace(h3, 1 - h3, n3)
X3, Y3 = np.meshgrid(xg3, xg3, indexing="ij")
f3 = np.sin(np.pi * X3) * np.sin(np.pi * Y3)
t0 = time.perf_counter()
u_dst = S3 @ ((S3 @ f3 @ S3) / (lam3[:, None] + lam3[None, :])) @ S3
t_dst = time.perf_counter() - t0
K3f = second_difference_matrix(n3)
A3f = (np.kron(K3f, np.eye(n3)) + np.kron(np.eye(n3), K3f)) / h3 ** 2
t0 = time.perf_counter()
u_lu = np.linalg.solve(A3f, f3.flatten())
t_lu = time.perf_counter() - t0


def cg(A, b, tol=1e-10):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    k = 0
    while np.sqrt(rr) > tol * np.linalg.norm(b) and k < len(b):
        Ap = A @ p
        al = rr / (p @ Ap)
        x += al * p
        r -= al * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
        k += 1
    return x, k


t0 = time.perf_counter()
u_cg, n_it = cg(A3f, f3.flatten())
t_cg = time.perf_counter() - t0
print(f"  DST：{t_dst * 1000:>8.2f} ms")
print(f"  CG （{n_it} 步）：{t_cg * 1000:>8.2f} ms")
print(f"  稠密 LU：{t_lu * 1000:>8.2f} ms")
check("  三種解法答案一致（DST vs LU）", u_dst.flatten(), u_lu, tol=1e-9)
check("  CG 也一致", u_cg, u_lu, tol=1e-7)
check("  DST 最快", t_dst < t_lu)

# 練習 4：臨界 Δt 附近的行為（直接追蹤「最高頻模態」的放大因子）
print("\n練習 4：顯式法在臨界 Δt 附近的行為")
n4 = 20
h4 = 1.0 / (n4 + 1)
A4 = second_difference_matrix(n4) / h4 ** 2
lam4, V4 = np.linalg.eigh(A4)
lam_top = lam4[-1]
v_top = V4[:, -1]                       # 最高頻（鋸齒狀）模態
dtc = 2 / lam_top
u04 = np.random.default_rng(5).standard_normal(n4)
print(f"  λ_max = {lam_top:.2f}，Δt_crit = 2/λ_max = {dtc:.3e}")
print(f"{'Δt/Δt_crit':>12} | {'放大因子 1−Δtλ_max':>20} | {'400 步後最高模態':>18} | "
      f"{'400 步後 ‖u‖':>14}")
print("-" * 74)
for factor in [0.5, 0.99, 1.01]:
    dt = factor * dtc
    amp_top = 1 - dt * lam_top
    G = np.eye(n4) - dt * A4
    u = u04.copy()
    c0 = v_top @ u
    for _ in range(400):
        u = G @ u
    c400 = v_top @ u
    print(f"{factor:>12.2f} | {amp_top:>20.6f} | {c400:>18.3e} | "
          f"{np.linalg.norm(u):>14.3e}")
    theory_c = amp_top ** 400 * c0
    check(f"  factor = {factor}：最高模態的係數 = (1−Δtλ_max)⁴⁰⁰·c₀",
          c400, theory_c,
          tol=max(abs(theory_c) * 1e-6, 1e-12 * np.linalg.norm(u04)))
    if factor < 1:
        check(f"  factor = {factor}：|放大因子| < 1 ⇒ 解衰減",
              abs(amp_top) < 1 and np.linalg.norm(u) < np.linalg.norm(u04))
    else:
        check(f"  factor = {factor}：|放大因子| > 1 ⇒ 最高模態爆炸",
              abs(amp_top) > 1 and abs(c400) > abs(c0))
print("  ⇒ 不穩定時最先爆掉的正是 λ_max 對應的「鋸齒狀」模態；")
print("     放大因子 (1−Δtλ) 在 Δt 略大於 Δt_crit 時對 λ_max 變成 < −1（震盪成長）")

# 練習 5：Lyapunov
print("\n練習 5：Lyapunov 方程 AX + XAᵀ = −Q")
A_ly = np.array([[-2.0, 1.0], [0.0, -3.0]])
Q_ly = np.array([[2.0, 0.0], [0.0, 4.0]])
S_ly = np.kron(np.eye(2), A_ly) + np.kron(A_ly, np.eye(2))
x_ly = np.linalg.solve(S_ly, vec(-Q_ly))
X_ly = x_ly.reshape((2, 2), order="F")
show_matrix("  解 X", X_ly)
check("  AX + XAᵀ = −Q", A_ly @ X_ly + X_ly @ A_ly.T, -Q_ly, tol=1e-10)
check("  X 對稱正定（A 穩定且 Q 正定 ⇒ Lyapunov 函數存在）",
      is_positive_definite(X_ly))
print("  ⇒ X 正定證明了系統 ẋ = Ax 的穩定性（Lyapunov 定理）")

# %% [markdown]
# ## 本章重點回顧
#
# * **Kronecker 積**：$(A\otimes B)(C\otimes D)=AC\otimes BD$、
#   特徵值相乘、$\det$ 與 $\operatorname{tr}$ 的乘法律；
#   **vec 技巧**把矩陣方程（Sylvester、Lyapunov）變成普通線性系統。
# * 二維五點差分 Laplacian $=\dfrac{1}{h^2}(K\otimes I+I\otimes K)$（Kronecker **和**）；
#   特徵值 $\lambda_p+\lambda_q$、特徵向量 $\mathbf v_p\otimes\mathbf v_q$
#   ── **這就是「分離變數」**。
# * 因為 $K$ 的特徵向量是離散正弦波，**DST 完全對角化 Laplacian**：
#   Poisson 方程三步解完（變換 → 除以 $\lambda$ → 反變換），遠快於稠密消去。
# * 時間離散的穩定性由**放大因子**決定：
#   顯式法要求 $\Delta t\le h^2/2$（熱傳導）或 $c\Delta t\le h$（波，CFL 條件）；
#   隱式法與 Crank–Nicolson 無條件穩定，後者還是二階精度。
# * 不穩定時最先爆掉的是**最高頻（鋸齒）模態** ── 因為它對應 $\lambda_{\max}$。
# * 極座標的分離變數給出 Bessel 方程（第 17、18 章的 SL 問題）：
#   圓形鼓的頻率是 $J_m$ 的零點，比值不是整數 ⇒ 鼓聲沒有泛音列。
#
# 下一章：量子力學的線性代數 ── Hermitian 算子、交換子與不確定性原理。
