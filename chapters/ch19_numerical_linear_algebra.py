# %% [markdown]
# # 第 19 章　數值線性代數：迭代法、Krylov 子空間與求積
#
# > 對應 Riley 第 27 章（27.1 代數方程、27.2 迭代法的收斂、27.3 同時線性方程、
# > 27.4 數值積分、27.5 有限差分），並對照 Strang 附錄 4（數值演算法）、
# > 本教材第 2 章（消去法成本）、第 13 章（條件數）。
#
# 當矩陣大到無法做 $O(n^3)$ 的消去法時（$n=10^6$ 的稀疏矩陣很常見），
# 就要改用**迭代法**：每一步只做矩陣–向量乘法 $O(\text{nnz})$。
#
# $$\mathbf x_{k+1}=M^{-1}N\mathbf x_k+M^{-1}\mathbf b
# \qquad\text{收斂}\iff\rho(M^{-1}N)<1$$
#
# | 方法 | 分裂 $A=M-N$ | 收斂條件 |
# |---|---|---|
# | Jacobi | $M=D$ | 嚴格對角優勢 |
# | Gauss–Seidel | $M=D+L$ | 對角優勢或對稱正定 |
# | SOR | $M=\frac{1}{\omega}D+L$ | $0<\omega<2$（正定時）|
# | **共軛梯度** | Krylov 子空間 | 對稱正定，$\le n$ 步精確 |
#
# ```bash
# python chapters/ch19_numerical_linear_algebra.py
# python tools/build_notebooks.py ch19
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

from linalg_tutorial.elimination import second_difference_matrix, solve
from linalg_tutorial.eigen import is_positive_definite
from linalg_tutorial.svd_tools import condition_number
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 19.1　迭代法的收斂：譜半徑說了算
#
# 把 $A$ 分裂成 $A=M-N$（$M$ 容易反轉），得到迭代
#
# $$M\mathbf x_{k+1}=N\mathbf x_k+\mathbf b
# \quad\Longrightarrow\quad
# \mathbf e_{k+1}=G\mathbf e_k,\ G=M^{-1}N$$
#
# 誤差以 $\|G^k\|$ 衰減，所以
#
# $$\boxed{\text{收斂}\iff\rho(G)=\max|\lambda_i(G)|<1}$$
#
# 而且**漸近收斂率就是 $\rho(G)$**（每步誤差乘 $\rho$）。

# %%
section("19.1 三種古典迭代法與它們的譜半徑")


def splitting_matrices(A, method, omega=1.0):
    """回傳迭代矩陣 G = M⁻¹N 與 M⁻¹（A = M − N）。"""
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    if method == "jacobi":
        M = D
    elif method == "gauss-seidel":
        M = D + L
    elif method == "sor":
        M = D / omega + L
    else:
        raise ValueError(method)
    N = M - A
    Minv = np.linalg.inv(M)
    return Minv @ N, Minv


def iterate(A, b, method, omega=1.0, iters=2000, tol=1e-12):
    """跑迭代法，回傳 (解, 每步的誤差)。"""
    G, Minv = splitting_matrices(A, method, omega)
    x_exact = np.linalg.solve(A, b)
    x = np.zeros_like(b)
    errs = [np.linalg.norm(x - x_exact)]
    for _ in range(iters):
        x = G @ x + Minv @ b
        errs.append(np.linalg.norm(x - x_exact))
        if errs[-1] < tol * max(1.0, np.linalg.norm(x_exact)):
            break
    return x, np.array(errs)


def geometric_rate(errs, lo=1e-11, hi=1e-2):
    """用「中段」的誤差估計幾何收斂率，避開已達機器精度的尾段。

    取 errs[k]/errs[0] 落在 [lo, hi] 的那一段，用 (e_b/e_a)^{1/(b−a)}。
    """
    errs = np.asarray(errs, dtype=float)
    rel = errs / errs[0]
    idx = np.where((rel <= hi) & (rel >= lo))[0]
    if len(idx) < 3:
        idx = np.arange(1, len(errs) - 1)
    a, b = idx[0], idx[-1]
    return (errs[b] / errs[a]) ** (1.0 / (b - a))


A = np.array([[4.0, -1.0, 0.0, 0.0],
              [-1.0, 4.0, -1.0, 0.0],
              [0.0, -1.0, 4.0, -1.0],
              [0.0, 0.0, -1.0, 4.0]])
b = np.array([1.0, 2.0, 0.0, 3.0])
show_matrix("A（對稱正定、嚴格對角優勢）", A)
check("A 對稱正定", is_positive_definite(A))
check("嚴格對角優勢：|aᵢᵢ| > Σ|aᵢⱼ|",
      np.all(np.abs(np.diag(A)) > np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))))

print(f"\n{'方法':<18} | {'ρ(G)':>10} | {'收斂?':>6} | {'步數（到 1e−12）':>18} | "
      f"{'實測收斂率':>12}")
print("-" * 76)
results = {}
for name, method, omega in [("Jacobi", "jacobi", 1.0),
                            ("Gauss–Seidel", "gauss-seidel", 1.0),
                            ("SOR (ω=1.3)", "sor", 1.3)]:
    G, _ = splitting_matrices(A, method, omega)
    rho = np.max(np.abs(np.linalg.eigvals(G)))
    x_it, errs = iterate(A, b, method, omega)
    measured = geometric_rate(errs)
    results[name] = (rho, errs)
    print(f"{name:<18} | {rho:>10.6f} | {str(rho < 1):>6} | {len(errs) - 1:>18} | "
          f"{measured:>12.6f}")
    check(f"  {name}：收斂到正解", x_it, np.linalg.solve(A, b), tol=1e-9)
    check(f"  {name}：實測收斂率 ≈ ρ(G)（相對差 "
          f"{abs(measured - rho) / rho:.1%}）", abs(measured - rho) / rho < 0.05)
print("  ⇒ 漸近收斂率精確地等於譜半徑")

# 不收斂的例子
A_bad = np.array([[1.0, 3.0], [2.0, 1.0]])
G_bad, _ = splitting_matrices(A_bad, "jacobi")
rho_bad = np.max(np.abs(np.linalg.eigvals(G_bad)))
print(f"\n不對角優勢的例子 A = [[1,3],[2,1]]：ρ(G_Jacobi) = {rho_bad:.4f} > 1")
check("ρ > 1 ⇒ Jacobi 發散", rho_bad > 1)
x_div, errs_div = iterate(A_bad, np.array([1.0, 1.0]), "jacobi", iters=30)
print(f"  30 步後的誤差 = {errs_div[-1]:.3e}（確實爆掉）")
check("誤差隨步數成長", errs_div[-1] > errs_div[0])

# 對角優勢 ⇒ Jacobi 必收斂（Gershgorin！）
print("\n為什麼對角優勢保證收斂？Gershgorin 圓盤（第 6 章）：")
G_j, _ = splitting_matrices(A, "jacobi")
radii = np.sum(np.abs(G_j), axis=1)
print(f"  G 的每列絕對值和 = {np.round(radii, 4)}（都 < 1）")
print(f"  Gershgorin ⇒ 所有 |λ(G)| ≤ max(列和) = {radii.max():.4f} < 1")
check("Gershgorin 給出 ρ(G) ≤ max 列和",
      np.max(np.abs(np.linalg.eigvals(G_j))) <= radii.max() + 1e-12)

# %% [markdown]
# ### SOR 的最佳鬆弛參數
#
# 對模型問題（$K=\text{tridiag}(-1,2,-1)$），理論最佳值是
#
# $$\omega_{\text{opt}}=\frac{2}{1+\sqrt{1-\rho_J^2}},\qquad
# \rho_{\text{SOR}}(\omega_{\text{opt}})=\omega_{\text{opt}}-1$$
#
# 這把收斂率從 $1-O(h^2)$ 改善到 $1-O(h)$ ── 和第 11 章「動量法」的
# $b\to\sqrt b$ 是同一種加速。

# %%
section("19.1 SOR 的最佳 ω")

n = 20
K = second_difference_matrix(n)
b_K = np.ones(n)
rho_J = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K, "jacobi")[0])))
omega_opt = 2 / (1 + np.sqrt(1 - rho_J ** 2))
print(f"n = {n}：ρ_Jacobi = {rho_J:.6f}，理論最佳 ω = {omega_opt:.6f}")
print(f"{'ω':>8} | {'ρ(G_SOR)':>12} | {'步數':>8}")
print("-" * 34)
best_omega, best_rho = None, 2.0
for omega in [0.8, 1.0, 1.2, 1.4, omega_opt, 1.9]:
    G, _ = splitting_matrices(K, "sor", omega)
    rho = np.max(np.abs(np.linalg.eigvals(G)))
    _, errs = iterate(K, b_K, "sor", omega, iters=20000)
    print(f"{omega:>8.4f} | {rho:>12.6f} | {len(errs) - 1:>8}")
    if rho < best_rho:
        best_omega, best_rho = omega, rho
check("最佳 ω 給出最小的 ρ", best_omega, omega_opt, tol=1e-9)
G_opt, _ = splitting_matrices(K, "sor", omega_opt)
check("ρ_SOR(ω_opt) = ω_opt − 1",
      np.max(np.abs(np.linalg.eigvals(G_opt))), omega_opt - 1, tol=1e-8)

print(f"\n收斂率隨 n 的變化（h = 1/(n+1)）：")
print(f"{'n':>5} | {'ρ_Jacobi':>12} | {'ρ_GS':>12} | {'ρ_SOR(opt)':>12} | "
      f"{'κ(K)':>10}")
print("-" * 62)
for n_ in [10, 20, 40, 80]:
    K_ = second_difference_matrix(n_)
    rJ = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K_, "jacobi")[0])))
    rG = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K_, "gauss-seidel")[0])))
    w_ = 2 / (1 + np.sqrt(1 - rJ ** 2))
    rS = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K_, "sor", w_)[0])))
    print(f"{n_:>5} | {rJ:>12.8f} | {rG:>12.8f} | {rS:>12.8f} | "
          f"{condition_number(K_):>10.1f}")
    check(f"  n = {n_}：ρ_GS = ρ_J²（模型問題的性質）", rG, rJ ** 2, tol=1e-8)
print("  ρ_Jacobi = cos(πh) ≈ 1 − π²h²/2（很慢）")
print("  ρ_SOR    ≈ 1 − 2πh（快一個數量級）⇒ 與 κ(K) ~ h⁻² 的平方根有關")
check("ρ_Jacobi = cos(πh)",
      np.max(np.abs(np.linalg.eigvals(splitting_matrices(
          second_difference_matrix(40), "jacobi")[0]))),
      np.cos(np.pi / 41), tol=1e-10)

fig, ax = new_axes("Convergence of classical iterations (K, n = 20)",
                   figsize=(6.2, 4.2), equal=False)
for name, (rho, errs) in results.items():
    pass
for name, method, omega, c in [("Jacobi", "jacobi", 1.0, "C0"),
                               ("Gauss-Seidel", "gauss-seidel", 1.0, "C1"),
                               (f"SOR (opt)", "sor", omega_opt, "C2")]:
    _, errs = iterate(K, b_K, method, omega, iters=5000)
    ax.semilogy(errs[:400], c, lw=1.6, label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("error")
ax.legend(fontsize=8)
finish(fig, "ch19_iterations")

# %% [markdown]
# ## 19.2　共軛梯度：Krylov 子空間的最佳解
#
# 對稱正定時，解 $A\mathbf x=\mathbf b$ 等價於最小化
# $\phi(\mathbf x)=\frac12\mathbf x^{\mathsf T}A\mathbf x-\mathbf b^{\mathsf T}\mathbf x$（第 11 章）。
#
# **共軛梯度**在第 $k$ 步給出 **Krylov 子空間**
#
# $$\mathcal K_k=\operatorname{span}\{\mathbf b,A\mathbf b,\dots,A^{k-1}\mathbf b\}$$
#
# 中使 $\phi$ 最小的向量。因此：
#
# * 最多 $n$ 步**精確**解出（Krylov 子空間填滿 $\mathbb R^n$）
# * 誤差界：$\dfrac{\|\mathbf e_k\|_A}{\|\mathbf e_0\|_A}\le2\left(\dfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^{k}$
#   ── 又是 $\kappa\to\sqrt\kappa$ 的加速！

# %%
section("19.2 共軛梯度法")


def conjugate_gradient(A, b, iters=None, tol=1e-14, M_inv=None):
    """共軛梯度（可選預條件子 M_inv）。回傳 (解, 殘差歷程)。"""
    n = len(b)
    iters = n if iters is None else iters
    x = np.zeros(n)
    r = b - A @ x
    z = r if M_inv is None else M_inv @ r
    p = z.copy()
    rz = r @ z
    hist = [np.linalg.norm(r)]
    for _ in range(iters):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        hist.append(np.linalg.norm(r))
        if hist[-1] < tol * max(1.0, np.linalg.norm(b)):
            break
        z = r if M_inv is None else M_inv @ r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, np.array(hist)


n = 30
K = second_difference_matrix(n)
rng = np.random.default_rng(0)
b_cg = rng.standard_normal(n)
x_exact = np.linalg.solve(K, b_cg)
x_cg, hist_cg = conjugate_gradient(K, b_cg)
print(f"n = {n}：CG 在 {len(hist_cg) - 1} 步後殘差 = {hist_cg[-1]:.3e}")
check("CG 收斂到正解", x_cg, x_exact, tol=1e-8)
check("CG 步數 ≤ n", len(hist_cg) - 1 <= n)

# 精確性：n 步之內一定解出（有限精度下近似）
print(f"\n{'步數 k':>8} | {'殘差':>14} | {'Krylov 子空間維度':>20}")
print("-" * 48)
Krylov = [b_cg]
for k in range(1, 8):
    Krylov.append(K @ Krylov[-1])
    dim = np.linalg.matrix_rank(np.array(Krylov).T, tol=1e-10)
    x_k, h_k = conjugate_gradient(K, b_cg, iters=k)
    print(f"{k:>8} | {h_k[-1]:>14.6e} | {dim:>20}")
    check(f"  k = {k}：Krylov 維度 = k+1", dim, k + 1)

# CG 的解是 Krylov 子空間裡的最佳近似（A-範數）
print("\nCG 的解在 Krylov 子空間中使 ‖e‖_A 最小：")
k_test = 5
Kry = np.column_stack([np.linalg.matrix_power(K, j) @ b_cg for j in range(k_test)])
x_k, _ = conjugate_gradient(K, b_cg, iters=k_test)
A_norm = lambda e: np.sqrt(e @ K @ e)
err_cg = A_norm(x_exact - x_k)
worse = True
for _ in range(2000):
    coef = rng.standard_normal(k_test)
    cand = Kry @ coef
    worse = worse and (A_norm(x_exact - cand) >= err_cg - 1e-10)
check(f"k = {k_test}：CG 解的 ‖e‖_A 最小（隨機比較 2000 次）", worse)
# Galerkin 條件：殘差 ⟂ Krylov 子空間
r_k = b_cg - K @ x_k
check("殘差 ⟂ Krylov 子空間（Galerkin 條件）", Kry.T @ r_k, np.zeros(k_test),
      tol=1e-8)

# 收斂率 vs 理論界
print(f"\n{'n':>5} | {'κ(K)':>10} | {'CG 步數':>10} | {'理論 √κ 的倍數':>16}")
print("-" * 50)
for n_ in [20, 40, 80, 160]:
    K_ = second_difference_matrix(n_)
    b_ = np.ones(n_)
    _, h_ = conjugate_gradient(K_, b_, iters=n_, tol=1e-10)
    kappa = condition_number(K_)
    print(f"{n_:>5} | {kappa:>10.1f} | {len(h_) - 1:>10} | "
          f"{(len(h_) - 1) / np.sqrt(kappa):>16.3f}")
    check(f"  n = {n_}：步數 = O(√κ)", (len(h_) - 1) / np.sqrt(kappa) < 2.0)
print("  ⇒ CG 的步數 ~ √κ（古典迭代法是 ~κ）── 這就是 Krylov 方法的威力")

# 與直接法比較
n_big = 400
K_big = second_difference_matrix(n_big)
b_big = np.ones(n_big)
t0 = time.perf_counter()
x_cg_big, h_big = conjugate_gradient(K_big, b_big, tol=1e-12)
t_cg = time.perf_counter() - t0
t0 = time.perf_counter()
x_dir = np.linalg.solve(K_big, b_big)
t_dir = time.perf_counter() - t0
print(f"\nn = {n_big}：CG {len(h_big)-1} 步耗時 {t_cg:.4f}s，"
      f"直接法 {t_dir:.4f}s")
check("兩者答案相同", x_cg_big, x_dir, tol=1e-8)
print(f"  CG 只需要矩陣–向量乘法：稀疏時每步 O(nnz) = O({3 * n_big})，"
      f"而消去法是 O(n³/3) = O({n_big ** 3 // 3:,})")

# %% [markdown]
# ### 預條件（preconditioning）
#
# 解 $M^{-1}A\mathbf x=M^{-1}\mathbf b$，只要 $M\approx A$ 且 $M$ 容易反轉，
# 就能把 $\kappa$ 大幅降低。
#
# | 預條件子 | $M$ |
# |---|---|
# | Jacobi | $\operatorname{diag}(A)$ |
# | 不完全 Cholesky | $\tilde R^{\mathsf T}\tilde R\approx A$ |
# | 多重網格 | 遞迴粗網格求解 |

# %%
section("19.2 預條件讓條件數下降")

# 人造一個對角尺度差很大的矩陣
n = 60
rng = np.random.default_rng(1)
scale = np.logspace(0, 4, n)
K_raw = second_difference_matrix(n)
A_scaled = np.diag(scale) @ K_raw @ np.diag(scale)
A_scaled = (A_scaled + A_scaled.T) / 2
b_sc = rng.standard_normal(n)
print(f"κ(A) = {condition_number(A_scaled):.4g}")

M_jacobi = np.diag(np.diag(A_scaled))
M_inv_j = np.linalg.inv(M_jacobi)
D_half = np.diag(1 / np.sqrt(np.diag(A_scaled)))
A_prec = D_half @ A_scaled @ D_half
print(f"κ(對稱 Jacobi 預條件後) = {condition_number(A_prec):.4g}")
check("預條件讓條件數大幅下降",
      condition_number(A_prec) < condition_number(A_scaled) / 100)

_, h_plain = conjugate_gradient(A_scaled, b_sc, iters=2000, tol=1e-10)
_, h_prec = conjugate_gradient(A_scaled, b_sc, iters=2000, tol=1e-10, M_inv=M_inv_j)
print(f"\nCG 步數：無預條件 {len(h_plain)-1}，Jacobi 預條件 {len(h_prec)-1}")
check("預條件減少迭代次數", len(h_prec) < len(h_plain))
x_pcg, _ = conjugate_gradient(A_scaled, b_sc, iters=2000, tol=1e-12, M_inv=M_inv_j)
check("預條件 CG 仍收斂到正解", x_pcg, np.linalg.solve(A_scaled, b_sc), tol=1e-6)

fig, ax = new_axes("Preconditioning accelerates CG", figsize=(6.2, 4.0), equal=False)
ax.semilogy(h_plain, "C0", lw=1.6, label=f"CG (kappa = {condition_number(A_scaled):.1e})")
ax.semilogy(h_prec, "C2", lw=1.6, label=f"PCG with Jacobi (kappa = {condition_number(A_prec):.1e})")
ax.set_xlabel("iteration")
ax.set_ylabel("residual")
ax.legend(fontsize=8)
finish(fig, "ch19_preconditioning")

# %% [markdown]
# ## 19.3　非線性方程的迭代：收斂階數
#
# Riley 27.1–27.2：解 $f(x)=0$ 的迭代 $x_{k+1}=g(x_k)$ 收斂條件是
# $|g'(x^{*})|<1$，而收斂**階數**由 $g$ 在 $x^{*}$ 的導數決定：
#
# | 方法 | 迭代 | 階數 |
# |---|---|---|
# | 簡單迭代 | $x_{k+1}=g(x_k)$ | 1（線性，率 $|g'|$）|
# | 牛頓法 | $x-f/f'$ | **2**（平方）|
# | 割線法 | — | 1.618（黃金比例！）|
#
# 多變數時 $g'$ 變成 **Jacobi 矩陣**，條件變成 $\rho(J)<1$ ──
# 又回到譜半徑。

# %%
section("19.3 迭代的收斂階數")

f = lambda x: x ** 3 - 2 * x - 5
fp = lambda x: 3 * x ** 2 - 2
root = 2.0945514815423265          # 已知的根

methods = {}
# 牛頓法
x, hist = 2.0, []
for _ in range(8):
    hist.append(abs(x - root))
    if abs(fp(x)) < 1e-300:
        break
    x = x - f(x) / fp(x)
methods["牛頓法"] = (np.array(hist), 2.0)
# 割線法
x0, x1 = 2.0, 2.5
hist = []
for _ in range(12):
    hist.append(abs(x1 - root))
    denom = f(x1) - f(x0)
    if abs(denom) < 1e-300:                  # 已收斂到機器精度，停止
        break
    x2 = x1 - f(x1) * (x1 - x0) / denom
    x0, x1 = x1, x2
methods["割線法"] = (np.array(hist), (1 + np.sqrt(5)) / 2)
# 簡單迭代 x = (2x+5)^{1/3}
x, hist = 2.0, []
g = lambda t: (2 * t + 5) ** (1 / 3)
for _ in range(25):
    hist.append(abs(x - root))
    x = g(x)
methods["簡單迭代"] = (np.array(hist), 1.0)

print(f"{'方法':<12} | {'理論階數':>10} | {'實測階數':>10} | {'10 步後的誤差':>16}")
print("-" * 56)
for name, (hist, order_theory) in methods.items():
    hist = hist[hist > 1e-15]
    if len(hist) >= 4:
        p_meas = np.log(hist[-1] / hist[-2]) / np.log(hist[-2] / hist[-3])
    else:
        p_meas = np.nan
    print(f"{name:<12} | {order_theory:>10.4f} | {p_meas:>10.4f} | "
          f"{hist[-1]:>16.3e}")
    check(f"  {name}：實測階數 ≈ {order_theory:.3f}", p_meas, order_theory, tol=0.25)
g_prime = (g(root + 1e-7) - g(root - 1e-7)) / 2e-7
print(f"\n簡單迭代的 |g'(x*)| = {abs(g_prime):.6f} < 1 ⇒ 收斂（率就是這個數）")
check("簡單迭代的收斂率 = |g'(x*)|",
      geometric_rate(methods["簡單迭代"][0]), abs(g_prime), tol=1e-3)

# 多變數：Jacobi 矩陣的譜半徑
print("\n多變數：收斂條件變成 ρ(Jacobi 矩陣) < 1")
G_multi = lambda v: np.array([0.3 * v[0] + 0.2 * v[1] + 1.0,
                              0.1 * v[0] - 0.4 * v[1] + 2.0])
J_multi = np.array([[0.3, 0.2], [0.1, -0.4]])
rho_multi = np.max(np.abs(np.linalg.eigvals(J_multi)))
print(f"  Jacobi 矩陣的 ρ = {rho_multi:.6f} < 1 ⇒ 收斂")
v = np.zeros(2)
v_star = np.linalg.solve(np.eye(2) - J_multi, np.array([1.0, 2.0]))
errs_multi = []
for _ in range(40):
    errs_multi.append(np.linalg.norm(v - v_star))
    v = G_multi(v)
check("收斂到不動點", v, v_star, tol=1e-10)
rate_multi = geometric_rate(np.array(errs_multi))
check(f"收斂率 ≈ ρ(J)（相對差 {abs(rate_multi - rho_multi) / rho_multi:.1%}）",
      abs(rate_multi - rho_multi) / rho_multi < 0.05)
print("  （有限步數下次主特徵值仍有殘餘貢獻，所以實測率與 ρ 差幾個百分點）")

# %% [markdown]
# ## 19.4　數值積分 = 線性代數
#
# 求積公式就是**線性泛函**：
#
# $$\int_a^b f(x)\,dx\approx\sum_{i=1}^{n}w_if(x_i)$$
#
# 要求對所有 $\le m$ 次多項式精確，就得到一個 **Vandermonde 線性系統**
#
# $$\sum_i w_ix_i^{k}=\int_a^bx^{k}dx,\qquad k=0,\dots,m$$
#
# | 公式 | 節點 | 精確到幾次 |
# |---|---|---|
# | 梯形法 | 2 個等距 | 1 |
# | Simpson | 3 個等距 | 3 |
# | $n$ 點 Newton–Cotes | $n$ 個等距 | $n-1$ |
# | **$n$ 點 Gauss** | $n$ 個最佳位置 | $2n-1$ |

# %%
section("19.4 求積權重 = 解 Vandermonde 系統")


def quadrature_weights(nodes, a=-1.0, b=1.0):
    """給定節點，解 Vandermonde 系統得到求積權重。"""
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    V = np.vander(nodes, n, increasing=True).T          # V[k,i] = xᵢᵏ
    moments = np.array([(b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(n)])
    return np.linalg.solve(V, moments)


print("梯形法：節點 {−1, 1}")
w_trap = quadrature_weights([-1.0, 1.0])
print(f"  權重 = {w_trap}（理論 (1,1)）")
check("梯形法權重 = (1,1)", w_trap, np.array([1.0, 1.0]))

print("\nSimpson 法：節點 {−1, 0, 1}")
w_simp = quadrature_weights([-1.0, 0.0, 1.0])
print(f"  權重 = {np.round(w_simp, 8)}（理論 (1/3, 4/3, 1/3)）")
check("Simpson 權重 = (1/3, 4/3, 1/3)", w_simp,
      np.array([1 / 3, 4 / 3, 1 / 3]))

print(f"\n{'公式':<22} | {'精確到幾次':>12} | {'∫x³ 誤差':>12} | {'∫x⁴ 誤差':>12}")
print("-" * 68)
for name, nodes in [("梯形（2 點等距）", [-1.0, 1.0]),
                    ("Simpson（3 點等距）", [-1.0, 0.0, 1.0]),
                    ("4 點等距", np.linspace(-1, 1, 4)),
                    ("2 點 Gauss", [-1 / np.sqrt(3), 1 / np.sqrt(3)]),
                    ("3 點 Gauss", [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])]:
    w = quadrature_weights(nodes)
    nodes = np.asarray(nodes, dtype=float)
    exact_deg = -1
    for deg in range(12):
        approx = np.sum(w * nodes ** deg)
        exact = (1 - (-1) ** (deg + 1)) / (deg + 1)
        if abs(approx - exact) < 1e-12:
            exact_deg = deg
        else:
            break
    e3 = abs(np.sum(w * nodes ** 3) - 0.0)
    e4 = abs(np.sum(w * nodes ** 4) - 2 / 5)
    print(f"{name:<22} | {exact_deg:>12} | {e3:>12.2e} | {e4:>12.2e}")
check("2 點 Gauss 精確到 3 次（= 2n−1）",
      abs(np.sum(quadrature_weights([-1 / np.sqrt(3), 1 / np.sqrt(3)])
                 * np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]) ** 3)) < 1e-14)
check("3 點 Gauss 精確到 5 次",
      abs(np.sum(quadrature_weights([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
                 * np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]) ** 5)) < 1e-14)
print("  ⇒ Gauss 用 n 個節點達到 2n−1 次精確度：節點位置也當成未知數來解")

# %% [markdown]
# ### Golub–Welsch：Gauss 節點 = 對稱三對角矩陣的特徵值
#
# 最漂亮的連結（第 17 章的正交多項式 + 第 6 章的特徵值）：
#
# 正交多項式滿足三項遞迴
# $x p_n=b_np_{n+1}+a_np_n+b_{n-1}p_{n-1}$，
# 把 $(a_n,b_n)$ 排成**對稱三對角 Jacobi 矩陣** $J$，則
#
# $$\text{Gauss 節點}=\lambda_i(J),\qquad
# \text{Gauss 權重}=\mu_0\,(\text{特徵向量第一分量})^2$$
#
# 求積問題被完全轉化成一個特徵值問題！

# %%
section("19.4 Golub–Welsch：求積 = 特徵值問題")


def golub_welsch(n):
    """用 Legendre 遞迴係數建 Jacobi 矩陣，特徵分解得到 Gauss–Legendre 節點與權重。"""
    k = np.arange(1, n)
    beta = k / np.sqrt(4 * k ** 2 - 1)        # Legendre 的次對角元素
    J = np.diag(beta, -1) + np.diag(beta, 1)  # 主對角為 0（Legendre 的 aₙ = 0）
    lam, V = np.linalg.eigh(J)
    weights = 2.0 * V[0, :] ** 2              # μ₀ = ∫₋₁¹1dx = 2
    return lam, weights


for n in [2, 3, 5, 8]:
    nodes_gw, w_gw = golub_welsch(n)
    nodes_np, w_np = np.polynomial.legendre.leggauss(n)
    print(f"  n = {n}：節點最大誤差 = "
          f"{np.max(np.abs(np.sort(nodes_gw) - np.sort(nodes_np))):.2e}，"
          f"權重最大誤差 = {np.max(np.abs(np.sort(w_gw) - np.sort(w_np))):.2e}")
    check(f"  n = {n}：Golub–Welsch 的節點 = numpy 的 Gauss 節點",
          np.sort(nodes_gw), np.sort(nodes_np), tol=1e-12)
    check(f"  n = {n}：權重相符", np.sort(w_gw), np.sort(w_np), tol=1e-12)
    # 精確度檢驗
    for deg in range(2 * n):
        approx = np.sum(w_gw * nodes_gw ** deg)
        exact = (1 - (-1) ** (deg + 1)) / (deg + 1)
        assert abs(approx - exact) < 1e-11, (n, deg)
    check(f"  n = {n}：對 ≤ {2*n-1} 次多項式精確", True)
print("  ⇒ 「最佳求積節點」就是「正交多項式的根」= 「Jacobi 矩陣的特徵值」")

# 收斂速度比較
f_test = lambda x: 1 / (1 + 25 * x ** 2)
exact_val = 2 * np.arctan(5) / 5
print(f"\n積分 ∫₋₁¹ dx/(1+25x²) = 2arctan(5)/5 = {exact_val:.12f}")
print(f"{'節點數':>8} | {'梯形法誤差':>14} | {'Gauss 誤差':>14}")
print("-" * 42)
for n_ in [5, 10, 20, 40]:
    xs_eq = np.linspace(-1, 1, n_)
    w_eq = np.full(n_, 2.0 / (n_ - 1))
    w_eq[0] = w_eq[-1] = 1.0 / (n_ - 1)
    err_trap = abs(np.sum(w_eq * f_test(xs_eq)) - exact_val)
    xg, wg = np.polynomial.legendre.leggauss(n_)
    err_gauss = abs(np.sum(wg * f_test(xg)) - exact_val)
    print(f"{n_:>8} | {err_trap:>14.3e} | {err_gauss:>14.3e}")
xg40, wg40 = np.polynomial.legendre.leggauss(40)
err_g40 = abs(np.sum(wg40 * f_test(xg40)) - exact_val)
xs40 = np.linspace(-1, 1, 40)
w40 = np.full(40, 2.0 / 39)
w40[0] = w40[-1] = 1.0 / 39
err_t40 = abs(np.sum(w40 * f_test(xs40)) - exact_val)
check("Gauss 求積比梯形法準得多（40 點時好 100 倍以上）",
      err_g40 < err_t40 / 100)
print(f"  40 點：Gauss 誤差 {err_g40:.2e} vs 梯形 {err_t40:.2e}"
      f"（好 {err_t40 / err_g40:.0f} 倍）")
print("  （f 的極點 ±i/5 很靠近 [−1,1]，所以 Gauss 的幾何收斂速率被壓低）")

fig, ax = new_axes("Quadrature nodes are eigenvalues (Golub-Welsch)",
                   figsize=(6.2, 4.0), equal=False)
for n_, c in [(4, "C0"), (8, "C1"), (16, "C2")]:
    nodes_gw, w_gw = golub_welsch(n_)
    ax.plot(nodes_gw, w_gw, c + "o-", ms=5, label=f"n = {n_}")
ax.set_xlabel("Gauss node (eigenvalue of J)")
ax.set_ylabel("weight (first component squared)")
ax.legend(fontsize=8)
finish(fig, "ch19_golub_welsch")

# %% [markdown]
# ## 動手練習
#
# 1. 對 $A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$ 手算 Jacobi 與 Gauss–Seidel 的
#    迭代矩陣與譜半徑。
# 2. 驗證模型問題中 $\rho_{GS}=\rho_J^2$（Gauss–Seidel 剛好快一倍）。
# 3. 實作「迭代精化」（iterative refinement）：用單精度解再用雙精度修正。
# 4. 用 Golub–Welsch 算出 Chebyshev–Gauss 的節點與權重，與 $\cos\frac{(2k-1)\pi}{2n}$ 比較。
# 5. 比較 CG 與 SOR 在 $n=200$ 的模型問題上的步數。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
G_j1, _ = splitting_matrices(A1, "jacobi")
G_gs1, _ = splitting_matrices(A1, "gauss-seidel")
show_matrix("練習 1：Jacobi 的 G = D⁻¹(L+U)", G_j1)
show_matrix("  Gauss–Seidel 的 G", G_gs1)
rho_j1 = np.max(np.abs(np.linalg.eigvals(G_j1)))
rho_gs1 = np.max(np.abs(np.linalg.eigvals(G_gs1)))
print(f"  ρ_Jacobi = {rho_j1:.6f}（理論 1/2），ρ_GS = {rho_gs1:.6f}（理論 1/4）")
check("  ρ_Jacobi = 1/2", rho_j1, 0.5)
check("  ρ_GS = 1/4 = ρ_J²", rho_gs1, 0.25)

# 練習 2
print("\n練習 2：模型問題的 ρ_GS = ρ_J²")
for n_ in [5, 15, 30]:
    K_ = second_difference_matrix(n_)
    rj = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K_, "jacobi")[0])))
    rg = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K_, "gauss-seidel")[0])))
    print(f"  n = {n_:>3}：ρ_J = {rj:.8f}，ρ_J² = {rj**2:.8f}，ρ_GS = {rg:.8f}")
    check(f"  n = {n_}：ρ_GS = ρ_J²", rg, rj ** 2, tol=1e-9)

# 練習 3：迭代精化
print("\n練習 3：迭代精化")
rng = np.random.default_rng(3)
n3 = 50
A3 = rng.standard_normal((n3, n3))
A3 = A3 + n3 * np.eye(n3)
x_true3 = rng.standard_normal(n3)
b3 = A3 @ x_true3
# 模擬「低精度」解：用 float32
A32, b32 = A3.astype(np.float32), b3.astype(np.float32)
x_low = np.linalg.solve(A32, b32).astype(np.float64)
err0 = np.linalg.norm(x_low - x_true3) / np.linalg.norm(x_true3)
print(f"  單精度解的相對誤差 = {err0:.3e}")
x_ref = x_low.copy()
for step in range(3):
    r = b3 - A3 @ x_ref                       # 殘差用雙精度算
    d = np.linalg.solve(A32, r.astype(np.float32)).astype(np.float64)
    x_ref = x_ref + d
    err = np.linalg.norm(x_ref - x_true3) / np.linalg.norm(x_true3)
    print(f"  精化第 {step + 1} 次後 = {err:.3e}")
check("  迭代精化把誤差降到雙精度水準",
      np.linalg.norm(x_ref - x_true3) / np.linalg.norm(x_true3) < err0 / 100)

# 練習 4：Chebyshev–Gauss
print("\n練習 4：Chebyshev–Gauss（遞迴係數 b₁ = 1/√2，其餘 1/2）")
n4 = 6
beta_c = np.full(n4 - 1, 0.5)
beta_c[0] = 1 / np.sqrt(2)
J_c = np.diag(beta_c, -1) + np.diag(beta_c, 1)
lam_c, V_c = np.linalg.eigh(J_c)
w_c = np.pi * V_c[0, :] ** 2
nodes_theory = np.sort(np.cos((2 * np.arange(1, n4 + 1) - 1) * np.pi / (2 * n4)))
print(f"  Golub–Welsch 節點 = {np.round(np.sort(lam_c), 8)}")
print(f"  理論 cos((2k−1)π/2n) = {np.round(nodes_theory, 8)}")
check("  節點相符", np.sort(lam_c), nodes_theory, tol=1e-12)
check("  權重全部 = π/n", w_c, np.full(n4, np.pi / n4), tol=1e-12)

# 練習 5
print("\n練習 5：n = 200 的模型問題")
n5 = 200
K5 = second_difference_matrix(n5)
b5 = np.ones(n5)
rj5 = np.max(np.abs(np.linalg.eigvals(splitting_matrices(K5, "jacobi")[0])))
w5 = 2 / (1 + np.sqrt(1 - rj5 ** 2))
_, h_sor = iterate(K5, b5, "sor", w5, iters=20000, tol=1e-10)
_, h_cg5 = conjugate_gradient(K5, b5, iters=n5, tol=1e-10)
print(f"  SOR（最佳 ω = {w5:.4f}）：{len(h_sor)-1} 步")
print(f"  共軛梯度：{len(h_cg5)-1} 步")
print(f"  κ(K) = {condition_number(K5):.1f}，√κ = {np.sqrt(condition_number(K5)):.1f}")
check("  CG 的步數 ≲ √κ", len(h_cg5) - 1 <= 2 * np.sqrt(condition_number(K5)))
check("  CG 比 SOR 少步數", len(h_cg5) < len(h_sor))

# %% [markdown]
# ## 本章重點回顧
#
# * 迭代法 $A=M-N$ 的收斂**完全由譜半徑決定**：$\rho(M^{-1}N)<1$，
#   而漸近收斂率就等於 $\rho$。嚴格對角優勢 $\Rightarrow$ Jacobi 收斂
#   （用 Gershgorin 圓盤一行證明）。
# * 模型問題上 $\rho_{GS}=\rho_J^2$；SOR 用最佳 $\omega$ 把
#   $1-O(h^2)$ 改善到 $1-O(h)$ ── 與第 11 章動量法的 $\kappa\to\sqrt\kappa$ 同源。
# * **共軛梯度** = 在 Krylov 子空間 $\{\mathbf b,A\mathbf b,\dots\}$ 中最小化
#   $\|\mathbf e\|_A$；殘差正交於子空間（Galerkin 條件）；
#   最多 $n$ 步精確，實際步數 $O(\sqrt\kappa)$。
# * **預條件**：解 $M^{-1}A\mathbf x=M^{-1}\mathbf b$，降低 $\kappa$ 就降低步數。
# * 非線性迭代的收斂階數：簡單迭代 1（率 $|g'|$）、牛頓法 2、割線法 1.618；
#   多變數時條件變成 $\rho(J)<1$。
# * **求積公式 = 解 Vandermonde 系統**；$n$ 點 Gauss 用最佳節點達到 $2n-1$ 次精確。
# * **Golub–Welsch**：Gauss 節點 = 正交多項式三項遞迴所構成的對稱三對角矩陣的
#   **特徵值**，權重 = 特徵向量第一分量的平方 ──
#   第 6 章（特徵值）、第 17 章（正交多項式）與數值積分在此會合。
#
# 下一章：偏微分方程的離散化與 Kronecker 積。
