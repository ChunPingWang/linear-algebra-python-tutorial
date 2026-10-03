# %% [markdown]
# # 第 9 章　奇異值分解（SVD）、影像壓縮與 PCA
#
# > 對應 Strang 第 7 章（7.1 奇異值與奇異向量、7.2 影像處理、7.3 主成分分析），
# > 附錄 7（條件數），以及 Riley 8.18（條件數）、31.2（樣本統計）。
#
# 第 7 章告訴我們：**對稱**矩陣有正交特徵向量。
# SVD 把這個好性質推廣到**任意** $m\times n$ 矩陣 ── 代價是需要**兩組**正交向量：
#
# $$\boxed{A\mathbf v_i=\sigma_i\mathbf u_i}\qquad
# A=U\Sigma V^{\mathsf T}
# =\sigma_1\mathbf u_1\mathbf v_1^{\mathsf T}+\dots+\sigma_r\mathbf u_r\mathbf v_r^{\mathsf T}$$
#
# | | 來源 | 是什麼的基底 |
# |---|---|---|
# | $\mathbf v_1,\dots,\mathbf v_r$（右奇異向量）| $A^{\mathsf T}A$ 的特徵向量 | 列空間 $C(A^{\mathsf T})$ |
# | $\mathbf u_1,\dots,\mathbf u_r$（左奇異向量）| $AA^{\mathsf T}$ 的特徵向量 | 欄空間 $C(A)$ |
# | $\mathbf v_{r+1},\dots,\mathbf v_n$ | | 零空間 $N(A)$ |
# | $\mathbf u_{r+1},\dots,\mathbf u_m$ | | 左零空間 $N(A^{\mathsf T})$ |
#
# **一張圖記住 SVD**：$V^{\mathsf T}$ 旋轉 → $\Sigma$ 拉伸 → $U$ 旋轉，
# 單位圓變成橢圓，半軸長就是 $\sigma_i$。
#
# ```bash
# python chapters/ch09_svd_and_pca.py
# python tools/build_notebooks.py ch09
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

from linalg_tutorial.orthogonal import pseudoinverse
from linalg_tutorial.subspaces import nullspace, rank
from linalg_tutorial.svd_tools import (
    condition_number,
    frobenius_error,
    low_rank_approx,
    pca,
    polar_decomposition,
    svd_from_eigen,
    svd_pieces,
)
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import draw_vector, finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 9.1　奇異值與奇異向量
#
# 建構步驟（也是證明）：
#
# 1. $A^{\mathsf T}A$ 對稱半正定 → 正交特徵向量 $\mathbf v_k$、特徵值 $\lambda_k\ge0$
# 2. $\sigma_k=\sqrt{\lambda_k}$（由大到小排）
# 3. $\mathbf u_k=A\mathbf v_k/\sigma_k$（$\sigma_k>0$ 的部分）
# 4. 檢查：$\mathbf u$ 自動正交，且是 $AA^{\mathsf T}$ 的特徵向量
#
# 關鍵計算 $AA^{\mathsf T}\mathbf u_k=A(A^{\mathsf T}A)\mathbf v_k/\sigma_k=\sigma_k^2\mathbf u_k$
# ── 用的就是結合律 $(AB)C=A(BC)$。

# %%
section("9.1 親手建構 SVD")

A = np.array([[5.0, 4.0], [0.0, 3.0]])
show_matrix("A（非對稱）", A)
show_matrix("AᵀA", A.T @ A)
show_matrix("AAᵀ", A @ A.T)
print(f"兩者的 trace 相同：{np.trace(A.T @ A):.0f} = {np.trace(A @ A.T):.0f}")
print(f"兩者的特徵值相同：{np.sort(np.linalg.eigvalsh(A.T @ A))} "
      f"vs {np.sort(np.linalg.eigvalsh(A @ A.T))}")

U, s, Vt = svd_from_eigen(A)
show_matrix("V（AᵀA 的特徵向量）", Vt.T)
print(f"σ = √λ = {s}（= 3√5 ≈ {3 * np.sqrt(5):.4f} 與 √5 ≈ {np.sqrt(5):.4f}）")
show_matrix("U（= Av/σ）", U)

check("σ = √(AᵀA 的特徵值)", s ** 2, np.sort(np.linalg.eigvalsh(A.T @ A))[::-1])
check("A = UΣVᵀ", U @ np.diag(s) @ Vt, A)
check("VᵀV = I", Vt @ Vt.T, np.eye(2))
check("UᵀU = I", U.T @ U, np.eye(2))
for k in range(2):
    check(f"Av{k + 1} = σ{k + 1}u{k + 1}", A @ Vt[k, :], s[k] * U[:, k])
check("與 numpy.linalg.svd 的 σ 一致", s, np.linalg.svd(A, compute_uv=False))
check("σ₁σ₂ = |det A|", np.prod(s), abs(np.linalg.det(A)))
print(f"\nσ₁ = {s[0]:.4f} ≥ 最大 |λ| = {np.max(np.abs(np.linalg.eigvals(A))):.4f} "
      f"≥ 最小 |λ| = {np.min(np.abs(np.linalg.eigvals(A))):.4f} ≥ σ₂ = {s[1]:.4f}")
check("|λ| ≤ σ₁ 恆成立", np.max(np.abs(np.linalg.eigvals(A))) <= s[0] + 1e-12)

# 秩一矩陣的 SVD
x = np.array([3.0, 4.0])
y = np.array([1.0, 2.0, 2.0])
R1 = np.outer(x, y)
U1, s1, Vt1 = np.linalg.svd(R1)
print(f"\n秩一矩陣 A = xyᵀ：σ₁ = ‖x‖‖y‖ = {np.linalg.norm(x) * np.linalg.norm(y):.4f}")
check("σ₁ = ‖x‖‖y‖", s1[0], np.linalg.norm(x) * np.linalg.norm(y))
check("其餘 σ = 0", s1[1:], np.zeros(len(s1) - 1))
check("u₁ = x/‖x‖（可能差一個符號）",
      np.abs(U1[:, 0]), np.abs(x / np.linalg.norm(x)))
print("  推論：|λ₁| ≤ σ₁ 在此退化成 |yᵀx| ≤ ‖x‖‖y‖ —— 正是 Schwarz 不等式！")

# %% [markdown]
# ### 幾何：圓 → 橢圓
#
# $$A=U\Sigma V^{\mathsf T}:\quad
# \text{旋轉}\ \xrightarrow{V^{\mathsf T}}\ \text{拉伸}\ \xrightarrow{\Sigma}\
# \text{旋轉}\ \xrightarrow{U}$$
#
# 任何 $2\times2$ 矩陣的 4 個數字 $\iff$ 2 個旋轉角 + 2 個奇異值。

# %%
circle = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 200)]).T
fig = plt.figure(figsize=(12.0, 3.4))
stages = [
    ("x (unit circle)", circle),
    (r"$V^Tx$ (rotate)", Vt @ circle),
    (r"$\Sigma V^Tx$ (stretch)", np.diag(s) @ Vt @ circle),
    (r"$U\Sigma V^Tx = Ax$ (rotate)", A @ circle),
]
for k, (title, pts) in enumerate(stages):
    ax = fig.add_subplot(1, 4, k + 1)
    ax.plot(pts[0], pts[1], "C0", lw=1.8)
    for j, c in zip(range(2), ["C3", "C2"]):
        base = [Vt.T[:, j], Vt.T[:, j], np.diag(s) @ Vt.T[:, j], A @ Vt.T[:, j]][k] \
            if k != 1 else Vt @ Vt.T[:, j]
        ax.annotate("", xy=base, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.grid(alpha=0.3)
finish(fig, "ch09_svd_geometry")
print("橢圓的半軸長 = 奇異值 =", np.round(s, 4))
check("橢圓最長半軸 = σ₁ = max‖Ax‖（‖x‖=1）",
      np.max(np.linalg.norm(A @ circle, axis=0)), s[0], tol=1e-3)
check("最短半軸 = σ₂ = min‖Ax‖",
      np.min(np.linalg.norm(A @ circle, axis=0)), s[1], tol=1e-3)

# %% [markdown]
# ### 完整型 vs 精簡型；SVD 給出四個子空間
#
# $$\underbrace{A}_{m\times n}=
# \underbrace{U}_{m\times m}\underbrace{\Sigma}_{m\times n}\underbrace{V^{\mathsf T}}_{n\times n}
# \qquad\text{或}\qquad
# A=\underbrace{U_r}_{m\times r}\underbrace{\Sigma_r}_{r\times r}\underbrace{V_r^{\mathsf T}}_{r\times n}$$
#
# 精簡型只留下 $r$ 個非零 $\sigma$，是真正有資訊的部分。

# %%
section("9.1 SVD 一次給出四個基本子空間")

A = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 4.0, 6.0, 8.0],
    [1.0, 1.0, 1.0, 1.0],
])
U, s, Vt = np.linalg.svd(A)
r = int(np.sum(s > 1e-10))
m, n = A.shape
print(f"A 是 {m}x{n}，σ = {np.round(s, 6)} ⇒ rank r = {r}")
check("rank 由 σ > 0 的個數決定", r, rank(A))

print(f"\nC(A)  基底 = U 的前 {r} 欄（在 R^{m}）")
print(f"N(Aᵀ) 基底 = U 的後 {m - r} 欄")
print(f"C(Aᵀ) 基底 = V 的前 {r} 欄（在 R^{n}）")
print(f"N(A)  基底 = V 的後 {n - r} 欄")
check("A·(V 的後幾欄) = 0 ⇒ 真的是 N(A)", A @ Vt[r:, :].T, np.zeros((m, n - r)))
check("(U 的後幾欄)ᵀ·A = 0 ⇒ 真的是 N(Aᵀ)", U[:, r:].T @ A, np.zeros((m - r, n)))
check("U 的前 r 欄張出 C(A)",
      rank(np.column_stack([U[:, :r], A])), r)
Nn = nullspace(A)
check("與消去法求得的 N(A) 張出同一空間",
      rank(np.column_stack([Vt[r:, :].T, Nn])), n - r)

A_reduced = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
check("精簡型 UᵣΣᵣVᵣᵀ = A", A_reduced, A)
print(f"\n儲存量：完整 {m * m + m * n + n * n} vs 精簡 {m * r + r + r * n} vs 原矩陣 {m * n}")

# 對稱矩陣的 SVD = 特徵分解（但負特徵值要翻符號）
S = np.array([[3.0, 1.0], [1.0, 3.0]])
lam, Q = np.linalg.eigh(S)
Us, ss, Vts = np.linalg.svd(S)
print(f"\n對稱正定 S 的 λ = {lam}，σ = {ss} → 完全相同")
check("S 正定時 σ = λ", np.sort(ss), np.sort(lam))
Sneg = np.array([[-3.0, 0.0], [0.0, 2.0]])
print(f"有負特徵值時：λ = {np.linalg.eigvalsh(Sneg)}，"
      f"σ = {np.linalg.svd(Sneg, compute_uv=False)} → σ = |λ|")
check("σ = |λ|", np.sort(np.linalg.svd(Sneg, compute_uv=False)),
      np.sort(np.abs(np.linalg.eigvalsh(Sneg))))

# 極分解：A = QS（旋轉 × 拉伸）
Q_pol, S_pol = polar_decomposition(np.array([[5.0, 4.0], [0.0, 3.0]]))
check("極分解 A = QS", Q_pol @ S_pol, np.array([[5.0, 4.0], [0.0, 3.0]]))
check("  Q 正交（純旋轉/反射）", Q_pol.T @ Q_pol, np.eye(2))
check("  S 對稱正定（純拉伸）", np.all(np.linalg.eigvalsh(S_pol) > 0))
print("  極分解是連續力學中「變形 = 旋轉 + 拉伸」的數學基礎")

# %% [markdown]
# ## 9.2　矩陣範數與 Eckart–Young 定理
#
# 三種只看奇異值的範數：
#
# $$\|A\|_2=\sigma_1,\qquad
# \|A\|_F=\sqrt{\sigma_1^2+\dots+\sigma_r^2},\qquad
# \|A\|_N=\sigma_1+\dots+\sigma_r$$
#
# **Eckart–Young 定理**：在上述任一範數下，
#
# $$\text{若 }\operatorname{rank}(B)=k,\ \text{則}\quad
# \|A-A_k\|\le\|A-B\|,\qquad
# A_k=\sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^{\mathsf T}$$
#
# 也就是說：**砍掉小的奇異值就是最佳的低秩近似**。
# 誤差剛好是被丟掉的那些奇異值：$\|A-A_k\|_F^2=\sigma_{k+1}^2+\dots+\sigma_r^2$。
#
# ⚠️ 注意：特徵值的最大絕對值**不是**範數（非零矩陣可能所有 $\lambda=0$）。

# %%
section("9.2 Eckart–Young：最佳低秩近似")

rng = np.random.default_rng(0)
A = rng.standard_normal((8, 6))
U, s, Vt = np.linalg.svd(A)
print(f"σ = {np.round(s, 4)}")
print(f"‖A‖₂ = σ₁ = {s[0]:.6f}      （numpy: {np.linalg.norm(A, 2):.6f}）")
print(f"‖A‖_F = √Σσ² = {np.sqrt(np.sum(s ** 2)):.6f}  （numpy: {np.linalg.norm(A, 'fro'):.6f}）")
print(f"‖A‖_N = Σσ = {np.sum(s):.6f}")
check("‖A‖₂ = σ₁", np.linalg.norm(A, 2), s[0])
check("‖A‖_F = √Σσ²", np.linalg.norm(A, "fro"), np.sqrt(np.sum(s ** 2)))
check("正交矩陣乘上去不改變範數",
      np.linalg.norm(np.linalg.qr(rng.standard_normal((8, 8)))[0] @ A, "fro"),
      np.linalg.norm(A, "fro"))

print(f"\n{'k':>3} | {'‖A−Aₖ‖_F':>12} | {'√(σ²_{k+1}+…)':>16} | {'隨機秩 k 矩陣的最小誤差':>22}")
print("-" * 64)
for k in range(1, 6):
    Ak = low_rank_approx(A, k)
    err = frobenius_error(A, Ak)
    theory = np.sqrt(np.sum(s[k:] ** 2))
    # 隨機試 300 個秩 k 矩陣，看能不能贏過 Aₖ
    best_rand = min(frobenius_error(A, rng.standard_normal((8, k)) @ rng.standard_normal((k, 6)))
                    for _ in range(300))
    print(f"{k:>3} | {err:>12.6f} | {theory:>16.6f} | {best_rand:>22.6f}")
    check(f"  k = {k}：誤差 = √(剩下的 σ² 之和)", err, theory)
    check(f"  k = {k}：沒有隨機秩 k 矩陣比 Aₖ 更好", best_rand >= err - 1e-9)

print("\n對角矩陣的直觀例子：A = diag(4,3,2,1)")
D = np.diag([4.0, 3.0, 2.0, 1.0])
D2 = low_rank_approx(D, 2)
show_matrix("最佳秩 2 近似 = diag(4,3,0,0)", D2)
check("誤差 = √(2²+1²) = √5", frobenius_error(D, D2), np.sqrt(5.0))

print("\n為什麼 max|λ| 不能當範數：")
N_nil = np.array([[0.0, 5.0], [0.0, 0.0]])
print(f"  A = [[0,5],[0,0]]：所有 λ = {np.linalg.eigvals(N_nil)}，但 A ≠ 0")
print(f"  σ₁ = {np.linalg.svd(N_nil, compute_uv=False)[0]:.1f} 才正確反映大小")
check("σ₁ = 5 > 0", np.linalg.svd(N_nil, compute_uv=False)[0], 5.0)

# %% [markdown]
# ## 9.3　影像壓縮
#
# 一張灰階影像就是一個矩陣。鄰近像素高度相關 ⇒ 奇異值快速衰減 ⇒ 可以壓縮。
#
# 儲存 $A_k$ 需要 $k(m+n+1)$ 個數字，原圖需要 $mn$ 個，壓縮比 $\dfrac{mn}{k(m+n+1)}$。
#
# 幾個經典例子：
#
# | 影像 | 秩 | 可壓縮性 |
# |---|---|---|
# | 三色旗（法國／德國）| **1** | 極好（$\mathbf 1\times$（顏色列））|
# | 希臘國旗（十字）| 3 | 很好 |
# | 下三角 1 矩陣 | 滿秩 | **差**（所有 $\sigma>1/2$）|
# | Hilbert 矩陣 | 滿秩 | 極好（$\sigma$ 陡降）|

# %%
section("9.3 旗幟的秩")

flags = {}
flag = np.zeros((6, 6))
flag[:, 0:2], flag[:, 2:4], flag[:, 4:6] = 0.2, 1.0, 0.6     # 三條垂直色帶
flags["法國式三色旗（垂直）"] = flag
flags["德國式三色旗（水平）"] = flag.T
greek = np.zeros((9, 9))          # 「十字」= 一條橫帶 + 一條直帶 → rank 2
greek[3:5, :] = 1.0
greek[:, 3:5] = 1.0
flags["十字（橫帶 + 直帶）"] = greek
greek2 = np.full((9, 9), 0.2)      # 三種互相獨立的「列樣式」→ rank 3
greek2[3:5, :] = 1.0
greek2[3:5, 3:5] = 0.5
greek2[5:, :] = 0.6
greek2[5:, 0:3] = 0.9
flags["三種列樣式的圖案"] = greek2
tonga = np.zeros((8, 8))
tonga[0:2, 0:4] = 0.0
tonga[2:3, 0:4] = 0.5
tonga[3:4, 0:4] = 0.8
tonga[4:, :] = 1.0
tonga[0:4, 4:] = 1.0
flags["Tonga 式（左上角方塊）"] = tonga
flags["下三角 1 矩陣"] = np.tril(np.ones((8, 8)))
for name, M in flags.items():
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"  {name:<24} rank = {rank(M)}，σ = {np.round(sv[:4], 3)} ...")
check("三色旗的 rank = 1", rank(flags["法國式三色旗（垂直）"]) == 1)
check("秩一旗幟 = (全1欄)(顏色列)",
      flags["法國式三色旗（垂直）"],
      np.outer(np.ones(6), flags["法國式三色旗（垂直）"][0, :]))
check("十字（橫帶+直帶）的 rank = 2", rank(flags["十字（橫帶 + 直帶）"]) == 2)
check("三種獨立的列樣式 ⇒ rank = 3", rank(flags["三種列樣式的圖案"]) == 3)
print("  ⇒ 圖案越「可分解成少數列/欄樣式」，秩越低、越容易壓縮")

# 下三角 1 矩陣：σ 永遠 > 1/2（Strang 的觀察）
print("\n下三角 1 矩陣（n = 40）的奇異值：")
T40 = np.tril(np.ones((40, 40)))
sT = np.linalg.svd(T40, compute_uv=False)
print(f"  最大 σ = {sT[0]:.4f}，最小 σ = {sT[-1]:.4f}")
check("所有 σ > 1/2 ⇒ 不可壓縮", np.all(sT > 0.5))
theory = 1 / (2 * np.sin(np.pi * (2 * np.arange(1, 41) - 1) / (2 * (2 * 40 + 1))))
check("σₖ = 1/(2 sin((2k−1)π/(2(2n+1))))", np.sort(sT), np.sort(theory), tol=1e-8)

# Hilbert 矩陣：σ 陡降
H = np.array([[1.0 / (i + j + 1) for j in range(40)] for i in range(40)])
sH = np.linalg.svd(H, compute_uv=False)
print(f"\nHilbert 矩陣（n = 40）：σ₁ = {sH[0]:.4g}，σ₁₀ = {sH[9]:.4g}，"
      f"σ₂₀ = {sH[19]:.4g}")
print(f"  條件數 = {condition_number(H):.4g} ← 惡名昭彰的病態矩陣")
check("Hilbert 的 σ 陡降（σ₁₀/σ₁ < 1e-7）", sH[9] / sH[0] < 1e-7)

fig, ax = new_axes("Singular value decay decides compressibility",
                   figsize=(6.0, 4.2), equal=False)
ax.semilogy(np.arange(1, 41), sT / sT[0], "o-", ms=4,
            label="lower triangular ones (not compressible)")
ax.semilogy(np.arange(1, 41), np.maximum(sH / sH[0], 1e-20), "s-", ms=4,
            label="Hilbert matrix (very compressible)")
ax.set_xlabel("k")
ax.set_ylabel(r"$\sigma_k/\sigma_1$")
ax.legend(fontsize=8)
finish(fig, "ch09_singular_value_decay")

# %%
section("9.3 合成影像的壓縮")


def synthetic_image(n=160):
    """產生一張有結構的合成灰階圖（漸層 + 圓 + 條紋 + 雜訊）。"""
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.35 + 0.3 * x + 0.15 * np.sin(8 * np.pi * y)              # 漸層 + 條紋
    img += 0.35 * (((x - 0.65) ** 2 + (y - 0.35) ** 2) < 0.03)       # 圓
    img += 0.2 * ((np.abs(x - y) < 0.06))                            # 對角帶
    rng = np.random.default_rng(0)
    img += 0.02 * rng.standard_normal((n, n))                        # 雜訊
    return np.clip(img, 0.0, 1.0)


img = synthetic_image()
m, n = img.shape
s_img = np.linalg.svd(img, compute_uv=False)
print(f"影像 {m}x{n} = {m * n:,} 個像素，rank = {rank(img)}")
print(f"\n{'k':>4} | {'儲存量 k(m+n+1)':>18} | {'壓縮比':>8} | {'相對誤差':>10} | {'保留能量':>10}")
print("-" * 62)
energy_total = np.sum(s_img ** 2)
for k in [1, 3, 10, 25, 50, 100]:
    Ak = low_rank_approx(img, k)
    store = k * (m + n + 1)
    rel = frobenius_error(img, Ak) / np.linalg.norm(img, "fro")
    energy = np.sum(s_img[:k] ** 2) / energy_total
    print(f"{k:>4} | {store:>18,} | {m * n / store:>7.2f}x | {rel:>9.2%} | {energy:>9.2%}")
check("k 越大誤差越小（單調）",
      all(frobenius_error(img, low_rank_approx(img, k)) >=
          frobenius_error(img, low_rank_approx(img, k + 5)) - 1e-12
          for k in [1, 5, 10, 20]))
check("保留能量 = Σσ²(前k)/Σσ²（全部）",
      np.sum(s_img ** 2) / energy_total, 1.0)

fig = plt.figure(figsize=(12.0, 2.9))
for i, k in enumerate([1, 3, 10, 25, None]):
    ax = fig.add_subplot(1, 5, i + 1)
    show = img if k is None else low_rank_approx(img, k)
    ax.imshow(show, cmap="gray", vmin=0, vmax=1)
    title = "original" if k is None else f"rank {k}"
    if k is not None:
        title += f"\n{m * n / (k * (m + n + 1)):.0f}x smaller"
    ax.set_title(title, fontsize=8)
    ax.axis("off")
finish(fig, "ch09_image_compression")

# %% [markdown]
# ## 9.4　主成分分析（PCA）
#
# 資料矩陣 $X$（$n$ 筆樣本 × $m$ 個變數），**先置中**（每欄減去平均），則
#
# $$S=\frac{X_c^{\mathsf T}X_c}{n-1}\quad\text{（樣本共變異數矩陣）}$$
#
# * $S$ 的對角線 = 各變數的**變異數**
# * 非對角線 = **共變異數**
# * $S$ 的特徵向量 = **主成分** = $X_c$ 的右奇異向量
# * $S$ 的特徵值 $=\sigma_i^2/(n-1)$ = 各主成分方向上的變異數
#
# **總變異數 = trace $S$ = $\sum\sigma_i^2/(n-1)$**，
# 第 $i$ 個主成分「解釋」了 $\sigma_i^2/\sum\sigma_j^2$ 的比例。
#
# ⚠️ 實務上**不要**先算 $S$ 再做特徵分解：直接對 $X_c$ 做 SVD 更快更準
# （算 $S$ 會把條件數平方）。

# %%
section("9.4 PCA = 置中資料的 SVD")

# Strang 的年齡-身高例子（已置中）
X = np.array([[3.0, 7.0], [-4.0, -6.0], [7.0, 8.0],
              [1.0, -1.0], [-4.0, -1.0], [-3.0, -7.0]])
print(f"資料（6 筆，2 個變數，已置中）：\n{X}")
n_samples = X.shape[0]
S = X.T @ X / (n_samples - 1)
show_matrix("樣本共變異數矩陣 S = XᵀX/(n−1)", S)
print(f"  變異數（對角線）= {np.diag(S)}")
print(f"  共變異數（非對角）= {S[0, 1]:.1f} > 0 ⇒ 年齡大身高也高")

res = pca(X, n_components=2, center=False)
lamS = np.linalg.eigvalsh(S)[::-1]
print(f"\nS 的特徵值 = {lamS}（和 = trace = {np.trace(S):.1f}）")
print(f"σ²/(n−1)  = {res['singular_values'] ** 2 / (n_samples - 1)}")
check("σ²/(n−1) = S 的特徵值", res["singular_values"] ** 2 / (n_samples - 1), lamS)
check("S 的 trace = 總變異數 = Σλ", np.trace(S), float(np.sum(lamS)))
print(f"\n第一主成分方向 = {np.round(res['components'][0], 4)}"
      f"（斜率 ≈ {res['components'][0][1] / res['components'][0][0]:.3f}）")
print(f"解釋變異比例 = {np.round(res['explained_ratio'], 4)}"
      f" → 第一主成分解釋 {res['explained_ratio'][0]:.1%}")
check("解釋比例加總 = 1", float(np.sum(res["explained_ratio"])), 1.0)
check("主成分正交", res["components"] @ res["components"].T, np.eye(2))

# 垂直最小平方（PCA）vs 垂直距離最小平方（一般回歸）
print("\n兩種「最佳直線」是不同的問題：")
slope_pca = res["components"][0][1] / res["components"][0][0]
A_reg = np.column_stack([np.ones(n_samples), X[:, 0]])
coef_reg = np.linalg.lstsq(A_reg, X[:, 1], rcond=None)[0]
print(f"  PCA（垂直距離平方和最小）斜率 = {slope_pca:.4f}")
print(f"  最小平方回歸（鉛直距離平方和最小）斜率 = {coef_reg[1]:.4f}")


def perp_sse(slope):
    """點到直線 y = slope·x 的垂直距離平方和。"""
    d = (X[:, 1] - slope * X[:, 0]) / np.sqrt(1 + slope ** 2)
    return float(d @ d)


print(f"  垂直距離平方和：PCA = {perp_sse(slope_pca):.4f}，"
      f"回歸 = {perp_sse(coef_reg[1]):.4f}")
check("PCA 在垂直距離意義下更好", perp_sse(slope_pca) <= perp_sse(coef_reg[1]) + 1e-9)
grid = np.linspace(slope_pca - 0.5, slope_pca + 0.5, 2001)
check("PCA 斜率真的是垂直距離的最小值",
      grid[int(np.argmin([perp_sse(g) for g in grid]))], slope_pca, tol=1e-3)

fig, ax = new_axes("PCA line vs least-squares line", figsize=(5.6, 5.0))
ax.plot(X[:, 0], X[:, 1], "ko", ms=6, label="data (centered)")
xs = np.linspace(-6, 8, 50)
ax.plot(xs, slope_pca * xs, "C0", lw=2, label=f"PCA (slope {slope_pca:.2f})")
ax.plot(xs, coef_reg[0] + coef_reg[1] * xs, "C3--", lw=2,
        label=f"least squares (slope {coef_reg[1]:.2f})")
for p in X:
    t = (p @ res["components"][0]) * res["components"][0]
    ax.plot([p[0], t[0]], [p[1], t[1]], "C0", lw=0.8, alpha=0.6)
    ax.plot([p[0], p[0]], [p[1], coef_reg[0] + coef_reg[1] * p[0]], "C3",
            lw=0.8, alpha=0.6)
ax.legend(fontsize=8)
finish(fig, "ch09_pca_vs_regression")

# %% [markdown]
# ### 高維 PCA 與 scree plot
#
# 真實資料的「有效秩」就是 scree plot 上訊號降到雜訊水準前的奇異值個數。
# 下面造一組「3 個真實主成分 + 雜訊」的 50 維資料，看 PCA 能不能找出 3。

# %%
section("9.4 找出資料的有效秩")

rng = np.random.default_rng(7)
n_samples, n_features, true_rank = 400, 50, 3
latent = rng.standard_normal((n_samples, true_rank)) * np.array([5.0, 3.0, 1.5])
loading = np.linalg.qr(rng.standard_normal((n_features, true_rank)))[0]
X_hi = latent @ loading.T + 0.10 * rng.standard_normal((n_samples, n_features))

res = pca(X_hi, n_components=6)
print(f"資料：{n_samples} 筆 x {n_features} 維，真實主成分數 = {true_rank}")
print(f"前 6 個解釋變異比例 = {np.round(res['explained_ratio'], 4)}")
cum = np.cumsum(res["explained_ratio"])
print(f"累積解釋比例 = {np.round(cum, 4)}")
check("前 3 個主成分解釋 > 95% 的變異", cum[2] > 0.95)
check("第 4 個主成分的貢獻已掉到雜訊水準（< 1%）", res["explained_ratio"][3] < 0.01)

sv = res["singular_values"]
print(f"\nσ 的前 8 個 = {np.round(sv[:8], 3)}")
drop = sv[:-1] / sv[1:]
print(f"相鄰比值最大處（elbow）在 k = {int(np.argmax(drop[:10])) + 1}")
check("elbow 出現在 k = 3", int(np.argmax(drop[:10])) + 1, true_rank)

# 直接 SVD vs 先算共變異數再特徵分解：條件數差異
Xc = X_hi - X_hi.mean(axis=0)
cond_X = condition_number(Xc)
cond_S = condition_number(Xc.T @ Xc)
print(f"\n條件數：κ(X) = {cond_X:.4g}，κ(XᵀX) = {cond_S:.4g} ≈ κ(X)² = {cond_X ** 2:.4g}")
check("κ(XᵀX) ≈ κ(X)²", cond_S / cond_X ** 2, 1.0, tol=1e-2)
print("  ⇒ 直接對 X 做 SVD，不要先算 XᵀX（這是 PCA 實作的黃金守則）")

fig = plt.figure(figsize=(10.5, 3.6))
ax = fig.add_subplot(1, 2, 1)
ax.semilogy(np.arange(1, 21), np.linalg.svd(Xc, compute_uv=False)[:20], "o-", ms=5)
ax.axvline(true_rank + 0.5, color="C3", ls="--", lw=1, label="true rank = 3")
ax.set_xlabel("component k")
ax.set_ylabel(r"$\sigma_k$")
ax.set_title("Scree plot: the elbow reveals the rank", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
scores = res["scores"]
ax.scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Data projected on the first two PCs", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
finish(fig, "ch09_pca_scree")

# %% [markdown]
# ### SVD 與偽逆、條件數
#
# $$A^{+}=V\Sigma^{+}U^{\mathsf T}\quad(\Sigma^{+}\text{：把 }\sigma>0\text{ 倒數，}0\text{ 留 }0)$$
#
# $$\kappa(A)=\frac{\sigma_{\max}}{\sigma_{\min}}
# \quad\text{（解 }A\mathbf x=\mathbf b\text{ 時相對誤差可能被放大的倍數）}$$

# %%
section("9.4 偽逆與條件數")

A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])      # 秩 1
U, s, Vt = np.linalg.svd(A, full_matrices=False)
s_plus = np.array([1 / si if si > 1e-12 else 0.0 for si in s])
A_plus = Vt.T @ np.diag(s_plus) @ U.T
check("A⁺ = VΣ⁺Uᵀ", A_plus, pseudoinverse(A))
check("AA⁺A = A", A @ A_plus @ A, A)
print(f"σ = {np.round(s, 6)} → Σ⁺ 的對角線 = {np.round(s_plus, 6)}（1/0 當成 0）")

print(f"\n條件數與誤差放大：")
for n_ in [5, 8, 12]:
    Hn = np.array([[1.0 / (i + j + 1) for j in range(n_)] for i in range(n_)])
    kappa = condition_number(Hn)
    x_true = np.ones(n_)
    b = Hn @ x_true
    x_num = np.linalg.solve(Hn, b)
    rel_err = np.linalg.norm(x_num - x_true) / np.linalg.norm(x_true)
    print(f"  Hilbert {n_}x{n_}：κ = {kappa:.3e}，解的相對誤差 = {rel_err:.3e}"
          f"，κ·ε_machine = {kappa * 2.2e-16:.3e}")
    check(f"  誤差 ≲ κ·ε_machine", rel_err <= max(kappa * 2.2e-16 * 100, 1e-14))

# %% [markdown]
# ## 動手練習
#
# 1. 對 $A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$ 手算 SVD（先算 $A^{\mathsf T}A$），
#    再用 `np.linalg.svd` 驗證。
# 2. 證明 $\|A\|_F^2=\sum\sigma_i^2=\operatorname{trace}(A^{\mathsf T}A)$。
# 3. 拿一張自己的照片（灰階），畫出奇異值衰減圖，找出「看不出差別」的最小 $k$。
# 4. 用 PCA 分析 $(x,y,z)$ 三維資料，其中 $z=2x-y+$ 小雜訊。
#    第三個主成分的奇異值應該很小 ── 為什麼？
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A1 = np.array([[3.0, 0.0], [4.0, 5.0]])
print(f"練習 1：AᵀA =\n{A1.T @ A1}")
lam1 = np.linalg.eigvalsh(A1.T @ A1)[::-1]
print(f"  AᵀA 的特徵值 = {lam1} ⇒ σ = {np.sqrt(lam1)}")
check("  σ 與 numpy 一致", np.sqrt(lam1), np.linalg.svd(A1, compute_uv=False))
check("  σ₁σ₂ = |det A| = 15", np.prod(np.sqrt(lam1)), abs(np.linalg.det(A1)))

# 練習 2
rng = np.random.default_rng(1)
for mn in [(4, 6), (7, 3)]:
    M = rng.standard_normal(mn)
    sv = np.linalg.svd(M, compute_uv=False)
    check(f"練習 2：{mn} 矩陣 ‖A‖²_F = Σσ² = trace(AᵀA)",
          np.linalg.norm(M, "fro") ** 2, float(np.sum(sv ** 2)))
    check(f"           = trace(AᵀA)", float(np.sum(sv ** 2)), np.trace(M.T @ M))

# 練習 3（用合成影像代替照片）
print("\n練習 3：合成影像需要多少個 σ 才「看不出差別」？")
img = synthetic_image(120)
total = np.linalg.norm(img, "fro")
for thresh in [0.10, 0.05, 0.02, 0.01]:
    k_needed = next(k for k in range(1, 121)
                    if frobenius_error(img, low_rank_approx(img, k)) / total < thresh)
    print(f"  相對誤差 < {thresh:.0%} 需要 k = {k_needed:>3}"
          f"（壓縮比 {120 * 120 / (k_needed * 241):.1f}x）")
check("  誤差 < 2% 時 k < 60（確實有壓縮效果）",
      next(k for k in range(1, 121)
           if frobenius_error(img, low_rank_approx(img, k)) / total < 0.02) < 60)

# 練習 4
print("\n練習 4：z = 2x − y + 雜訊")
rng = np.random.default_rng(3)
xy = rng.standard_normal((300, 2))
z = 2 * xy[:, 0] - xy[:, 1] + 0.05 * rng.standard_normal(300)
data = np.column_stack([xy, z])
res4 = pca(data, n_components=3)
print(f"  解釋變異比例 = {np.round(res4['explained_ratio'], 5)}")
print(f"  第三主成分方向 = {np.round(res4['components'][2], 4)}"
      f"（≈ (2,−1,−1)/√6 = {np.round(np.array([2, -1, -1]) / np.sqrt(6), 4)}）")
check("  第三個主成分幾乎沒有變異（< 0.1%）", res4["explained_ratio"][2] < 0.001)
print("  原因：資料幾乎落在平面 2x − y − z = 0 上，第三個方向就是平面的法向量")
normal = res4["components"][2] / np.linalg.norm(res4["components"][2])
target = np.array([2.0, -1.0, -1.0]) / np.sqrt(6)
check("  第三主成分 ∥ 平面法向量", abs(abs(normal @ target) - 1.0) < 0.02)

# %% [markdown]
# ## 本章重點回顧
#
# * SVD $A=U\Sigma V^{\mathsf T}$ 對**任何**矩陣都成立：
#   $\mathbf v$ 是 $A^{\mathsf T}A$ 的特徵向量、$\mathbf u=A\mathbf v/\sigma$、
#   $\sigma^2$ 是 $A^{\mathsf T}A$ 與 $AA^{\mathsf T}$ 的共同特徵值。
# * 幾何上是「旋轉 → 拉伸 → 旋轉」，單位圓變橢圓，半軸長就是 $\sigma$。
#   $\sigma_1=\max\|A\mathbf x\|/\|\mathbf x\|=\|A\|_2$，且 $|\lambda|\le\sigma_1$。
# * SVD 一次給出四個基本子空間的**正交**基底 ── 這是第 3、4 章的完美結局。
# * **Eckart–Young**：砍掉小的 $\sigma$ 就是最佳低秩近似，誤差 $=\sqrt{\sum_{i>k}\sigma_i^2}$。
#   影像壓縮、去雜訊、推薦系統都靠這一條。
# * **PCA** = 置中資料矩陣的 SVD。主成分 = 共變異數矩陣的特徵向量，
#   解釋變異比例 $=\sigma_i^2/\sum\sigma_j^2$。PCA 最小化**垂直**距離（不同於回歸的鉛直距離）。
# * 實作守則：直接對資料矩陣做 SVD，不要先算 $X^{\mathsf T}X$（條件數會平方）。
#
# 下一章：線性轉換 ── 不靠座標也能談矩陣，以及「什麼才是好的基底」。
