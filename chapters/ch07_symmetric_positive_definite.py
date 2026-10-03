# %% [markdown]
# # 第 7 章　對稱矩陣、正定性與複數／Hermitian 矩陣
#
# > 對應 Strang 6.3（對稱正定矩陣）、6.4（複數向量與矩陣、Fourier 矩陣、循環矩陣），
# > 以及 Riley 8.7（Hermitian 共軛）、8.12（特殊方陣）、8.17（二次式與 Hermitian 式）、
# > 8.13–8.14（Hermitian 算子的特徵值）。
#
# **譜定理（Spectral Theorem）** ── 線性代數最美的定理：
#
# $$S=S^{\mathsf T}\quad\Longrightarrow\quad
# S=Q\Lambda Q^{\mathsf T},\qquad Q^{\mathsf T}Q=I,\ \lambda_i\in\mathbb R$$
#
# 對稱矩陣的特徵值**全為實數**、特徵向量**可選成互相正交**。
# 再加上「全部 $\lambda>0$」就是**正定矩陣** ── 應用數學的核心。
#
# ```bash
# python chapters/ch07_symmetric_positive_definite.py
# python tools/build_notebooks.py ch07
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
    is_positive_definite,
    is_symmetric,
    rayleigh_quotient,
    spectral_decomposition,
)
from linalg_tutorial.elimination import cholesky, ldu, lu_no_pivot, second_difference_matrix
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 7.1　譜定理：$S=Q\Lambda Q^{\mathsf T}$
#
# 兩個證明的要點：
#
# * **特徵值為實數**：$S\mathbf x=\lambda\mathbf x$ 兩邊左乘 $\bar{\mathbf x}^{\mathsf T}$，
#   $\lambda=\dfrac{\bar{\mathbf x}^{\mathsf T}S\mathbf x}{\bar{\mathbf x}^{\mathsf T}\mathbf x}$，
#   分子分母都是實數。
# * **特徵向量正交**：若 $S\mathbf x=\lambda\mathbf x$、$S\mathbf y=\alpha\mathbf y$ 且 $\lambda\ne\alpha$，
#   則 $\mathbf x$ 在 $S-\alpha I$ 的欄空間（= 列空間，因為對稱！），
#   $\mathbf y$ 在它的零空間 ── 而列空間永遠垂直零空間。
#
# 譜分解把 $S$ 寫成 $r$ 個**正交投影**的加權和：
#
# $$S=\lambda_1\mathbf q_1\mathbf q_1^{\mathsf T}+\dots+\lambda_n\mathbf q_n\mathbf q_n^{\mathsf T}$$

# %%
section("7.1 譜定理與譜分解")

S = np.array([
    [4.0, 1.0, 1.0],
    [1.0, 3.0, 0.0],
    [1.0, 0.0, 2.0],
])
show_matrix("S（對稱）", S)
check("S = Sᵀ", is_symmetric(S))

lam, Q, pieces = spectral_decomposition(S)
print(f"特徵值 λ = {lam}（全為實數）")
show_matrix("Q（特徵向量，正交）", Q)
check("λ 全為實數", np.all(np.isreal(lam)))
check("QᵀQ = I（特徵向量正交單位化）", Q.T @ Q, np.eye(3))
check("S = QΛQᵀ", Q @ np.diag(lam) @ Q.T, S)
check("Q⁻¹ = Qᵀ", np.linalg.inv(Q), Q.T)

print("\n譜分解 S = Σ λₖ qₖqₖᵀ（每一項都是秩一的正交投影）：")
acc = np.zeros_like(S)
for k, (l, P) in enumerate(pieces):
    acc += l * P
    print(f"  λ{k + 1} = {l:+.4f}，‖S − 累積‖ = {np.linalg.norm(S - acc):.3e}，"
          f"P² = P：{np.allclose(P @ P, P)}")
check("Σ λₖqₖqₖᵀ = S", acc, S)
check("Σ qₖqₖᵀ = I（投影加起來是恆等）", sum(P for _, P in pieces), np.eye(3))

# 反對稱與正交矩陣的特徵值
print("\n三類矩陣的特徵值位置（複平面）：")
A_skew = np.array([[0.0, 2.0, -1.0], [-2.0, 0.0, 3.0], [1.0, -3.0, 0.0]])
Q_orth, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))
print(f"  對稱 S  ：λ = {np.round(np.linalg.eigvals(S), 4)}        → 實軸上")
print(f"  反對稱 A：λ = {np.round(np.linalg.eigvals(A_skew), 4)}  → 純虛數")
print(f"  正交 Q  ：|λ| = {np.round(np.abs(np.linalg.eigvals(Q_orth)), 4)}      → 單位圓上")
check("反對稱矩陣的 λ 是純虛數", np.allclose(np.real(np.linalg.eigvals(A_skew)), 0,
                                            atol=1e-10))
check("正交矩陣的 |λ| = 1", np.abs(np.linalg.eigvals(Q_orth)), np.ones(3))
print("  類比：對稱 ↔ 實數、反對稱 ↔ 虛數、正交 ↔ 單位圓上的複數")

# %% [markdown]
# ## 7.2　正定矩陣的五個等價測試
#
# | 測試 | 條件 | 數學領域 |
# |---|---|---|
# | 1 | 所有 $\lambda>0$ | 特徵值 |
# | 2 | **能量** $\mathbf x^{\mathsf T}S\mathbf x>0$（$\forall\mathbf x\ne0$）| 二次式 |
# | 3 | $S=A^{\mathsf T}A$，$A$ 各欄獨立 | 分解 |
# | 4 | 所有**前導主子式** $D_1,\dots,D_n>0$ | 行列式 |
# | 5 | 所有**主元** $>0$ | 消去法 |
#
# 五個測試完全等價。**能量測試是最好的定義**，因為它直接連到最小化問題：
#
# $$S\text{ 正定}\iff f(\mathbf x)=\tfrac12\mathbf x^{\mathsf T}S\mathbf x
# \text{ 是嚴格凸函數（碗狀）}$$
#
# 允許等號（$\lambda\ge0$、能量 $\ge0$）就是**半正定**。

# %%
section("7.2 五個測試")


def five_tests(S, name=""):
    """對同一個對稱矩陣跑五個正定測試，印出結果。"""
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    lam = np.linalg.eigvalsh(S)
    t1 = bool(np.all(lam > 1e-12))
    rng = np.random.default_rng(0)
    # 隨機向量幾乎不可能恰好落在「零能量方向」上，所以把特徵向量一併納入測試
    samples = np.vstack([rng.standard_normal((500, n)), np.linalg.eigh(S)[1].T])
    t2 = all(float(x @ S @ x) > 1e-12 for x in samples)
    try:
        R = cholesky(S)
        t3 = np.allclose(R.T @ R, S)
    except ValueError:
        t3, R = False, None
    dets = [det_lu(S[:k, :k]) for k in range(1, n + 1)]
    t4 = bool(np.all(np.array(dets) > 1e-12))
    try:
        pivots = np.diag(lu_no_pivot(S)[1])
        t5 = bool(np.all(pivots > 1e-12))
    except ValueError:
        pivots, t5 = np.array([np.nan]), False
    print(f"\n{name}")
    print(f"  ① 特徵值 λ = {np.round(lam, 4)}                  → {t1}")
    print(f"  ② 能量 xᵀSx > 0（500 個隨機 x）                  → {t2}")
    print(f"  ③ S = RᵀR（Cholesky 成功）                       → {t3}")
    print(f"  ④ 前導主子式 = {np.round(dets, 4)}            → {t4}")
    print(f"  ⑤ 主元 = {np.round(pivots, 4)}                  → {t5}")
    verdict = "正定" if t1 else ("半正定" if np.all(lam > -1e-10) else "不定")
    print(f"  ⇒ 結論：{verdict}（五個測試一致：{t1 == t2 == t3 == t4 == t5}）")
    return t1 == t2 == t3 == t4 == t5


agree = []
agree.append(five_tests([[2.0, 4.0], [4.0, 9.0]], "S = [[2,4],[4,9]]（正定）"))
agree.append(five_tests([[9.0, 3.0], [3.0, 1.0]], "T = [[9,3],[3,1]]（半正定，det = 0）"))
agree.append(five_tests([[0.0, 2.0], [2.0, 1.0]], "U = [[0,2],[2,1]]（不定）"))
agree.append(five_tests(second_difference_matrix(4), "K₄（二階差分矩陣，正定）"))
check("\n五個測試對所有例子都給一致的結論", all(agree))

# 能量的代數：平方和
print("\n能量的配方法（為什麼能量 > 0）：")
print("  xᵀSx = 2x₁² + 8x₁x₂ + 9x₂² = 2(x₁ + 2x₂)² + x₂²  ← 兩個平方之和，必 ≥ 0")
Sd = np.array([[2.0, 4.0], [4.0, 9.0]])
rng = np.random.default_rng(1)
for _ in range(3):
    x = rng.standard_normal(2)
    lhs = x @ Sd @ x
    rhs = 2 * (x[0] + 2 * x[1]) ** 2 + x[1] ** 2
    check(f"  x = {np.round(x, 3)}：兩種算法相同", lhs, rhs)

# 正定矩陣的運算封閉性
S1 = np.array([[2.0, 1.0], [1.0, 3.0]])
S2 = np.array([[5.0, -2.0], [-2.0, 4.0]])
check("S₁, S₂ 正定 ⇒ S₁ + S₂ 正定（能量相加）", is_positive_definite(S1 + S2))
check("S 正定 ⇒ S⁻¹ 正定（λ → 1/λ）", is_positive_definite(np.linalg.inv(S1)))
check("S 正定 ⇒ cS 正定（c > 0）", is_positive_definite(3 * S1))
A_any = np.random.default_rng(2).standard_normal((5, 3))
check("AᵀA 半正定（任意 A）", np.all(np.linalg.eigvalsh(A_any.T @ A_any) > -1e-10))
check("A 各欄獨立 ⇒ AᵀA 正定", is_positive_definite(A_any.T @ A_any))

# %% [markdown]
# ### 主元、行列式與 $LDL^{\mathsf T}$、Cholesky
#
# 第 $k$ 個主元 $=\dfrac{D_k}{D_{k-1}}$（前導主子式之比）── 所以測試 4 與 5 等價。
#
# 消去法對對稱矩陣給出
#
# $$S=LDL^{\mathsf T}=(L\sqrt D)(\sqrt D L^{\mathsf T})=A^{\mathsf T}A$$
#
# 這就是 **Cholesky 分解**，也是測試 3 中那個 $A$ 的具體建構法。

# %%
section("7.2 主元 = 行列式之比；Cholesky = √主元")

K = second_difference_matrix(4)
show_matrix("K₄", K)
dets = [1.0] + [det_lu(K[:k, :k]) for k in range(1, 5)]
pivots = np.diag(lu_no_pivot(K)[1])
print(f"前導主子式 D₀..D₄ = {np.round(dets, 4)}")
print(f"主元             = {np.round(pivots, 4)}")
ratios = [dets[k] / dets[k - 1] for k in range(1, 5)]
print(f"Dₖ/Dₖ₋₁          = {np.round(ratios, 4)}")
check("第 k 個主元 = Dₖ/Dₖ₋₁", pivots, np.array(ratios))

L, D, Lt = ldu(K)
check("K = L·D·Lᵀ", L @ D @ L.T, K)
check("D 的對角線就是主元", np.diag(D), pivots)
R = cholesky(K)
show_matrix("R（Cholesky 上三角，對角線 = √主元）", R)
check("K = RᵀR", R.T @ R, K)
check("R 的對角線 = √主元", np.diag(R), np.sqrt(pivots))
check("R = √D·Lᵀ", R, np.sqrt(D) @ L.T)

# 另一種 A：由特徵分解取平方根
lamK, QK = np.linalg.eigh(K)
S_half = QK @ np.diag(np.sqrt(lamK)) @ QK.T
check("對稱平方根 S^{1/2}：(S^{1/2})² = K", S_half @ S_half, K)
check("S^{1/2} 本身對稱正定", is_positive_definite(S_half))
print("  所以 A 不唯一：三角的（Cholesky）與對稱的（平方根）都滿足 S = AᵀA")

# %% [markdown]
# ## 7.3　正定性 = 最小值問題
#
# 單變數：$f'(x_0)=0$ 且 $f''(x_0)>0$ $\Rightarrow$ 極小。
#
# 多變數：$\nabla f(\mathbf x_0)=\mathbf 0$ 且 **Hessian 矩陣正定** $\Rightarrow$ 極小。
#
# $$H=\begin{bmatrix}
# \partial^2f/\partial x^2 & \partial^2f/\partial x\partial y\\
# \partial^2f/\partial x\partial y & \partial^2f/\partial y^2\end{bmatrix}$$
#
# | Hessian | 幾何 | 臨界點類型 |
# |---|---|---|
# | 正定 | 碗（bowl）| 極小 |
# | 負定 | 倒碗 | 極大 |
# | 不定 | 馬鞍 | 鞍點 |
# | 半正定 | 山谷（valley）| 一條極小線 |
#
# 能量 $E=\mathbf x^{\mathsf T}S\mathbf x=1$ 的等高線是**橢圓**，
# 主軸沿著 $S$ 的特徵向量，半軸長 $=1/\sqrt{\lambda_i}$。

# %%
section("7.3 能量的幾何：橢圓、碗、馬鞍")

for name, M in [("正定 [[5,4],[4,5]]", np.array([[5.0, 4.0], [4.0, 5.0]])),
                ("半正定 [[9,3],[3,1]]", np.array([[9.0, 3.0], [3.0, 1.0]])),
                ("不定 [[1,2],[2,1]]", np.array([[1.0, 2.0], [2.0, 1.0]]))]:
    lam = np.linalg.eigvalsh(M)
    shape = ("碗（嚴格凸，唯一極小）" if np.all(lam > 1e-12)
             else "山谷（凸，一整條極小線）" if np.all(lam > -1e-12)
             else "馬鞍（鞍點）")
    axes = ["∞" if l <= 1e-12 else f"{1 / np.sqrt(abs(l)):.3f}" for l in lam]
    print(f"  {name:<22} λ = {np.round(lam, 4)}  → {shape}")
    print(f"      能量 = 1 的等高線半軸長 1/√λ = {axes}")

fig = plt.figure(figsize=(11.0, 3.6))
grid = np.linspace(-2.2, 2.2, 240)
Xg, Yg = np.meshgrid(grid, grid)
for k, (title, M) in enumerate([
        ("positive definite: bowl", np.array([[5.0, 4.0], [4.0, 5.0]])),
        ("semidefinite: valley", np.array([[9.0, 3.0], [3.0, 1.0]])),
        ("indefinite: saddle", np.array([[1.0, 2.0], [2.0, 1.0]]))]):
    E = M[0, 0] * Xg ** 2 + 2 * M[0, 1] * Xg * Yg + M[1, 1] * Yg ** 2
    ax = fig.add_subplot(1, 3, k + 1)
    cs = ax.contour(Xg, Yg, E, levels=[-8, -4, -1, 0, 1, 4, 8, 16], cmap="coolwarm")
    ax.clabel(cs, inline=True, fontsize=6)
    lam, V = np.linalg.eigh(M)
    for j in range(2):
        ax.plot([-2 * V[0, j], 2 * V[0, j]], [-2 * V[1, j], 2 * V[1, j]],
                "k--", lw=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
finish(fig, "ch07_energy_contours")

# Hessian 測試：f = e^{x²+y²}
section("7.3 Hessian 判斷臨界點")


def hessian_numeric(f, x, h=1e-5):
    """用中央差分近似 Hessian（驗證解析式）。"""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                       - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


f = lambda v: np.exp(v[0] ** 2 + v[1] ** 2)


def hessian_exact(v):
    x, y = v
    e = np.exp(x ** 2 + y ** 2)
    return e * np.array([[2 + 4 * x ** 2, 4 * x * y], [4 * x * y, 2 + 4 * y ** 2]])


for pt in [np.array([0.0, 0.0]), np.array([0.5, -0.3])]:
    H = hessian_exact(pt)
    check(f"f = e^(x²+y²) 在 {pt} 的 Hessian（解析 vs 數值）",
          H, hessian_numeric(f, pt), tol=1e-3)
    check(f"  Hessian 正定 ⇒ 嚴格凸", is_positive_definite(H))

print("\n各類臨界點：")
tests = {
    "f = x² + y²（極小）": np.array([[2.0, 0.0], [0.0, 2.0]]),
    "f = −x² − y²（極大）": np.array([[-2.0, 0.0], [0.0, -2.0]]),
    "f = x² − y²（鞍點）": np.array([[2.0, 0.0], [0.0, -2.0]]),
    "f = (x+y)²（山谷）": np.array([[2.0, 2.0], [2.0, 2.0]]),
}
for name, H in tests.items():
    lam = np.linalg.eigvalsh(H)
    kind = ("極小" if np.all(lam > 1e-12) else "極大" if np.all(lam < -1e-12)
            else "山谷（半正定）" if np.all(lam > -1e-12) else "鞍點")
    print(f"  {name:<22} λ = {np.round(lam, 2)} → {kind}")

# Rayleigh 商：λ_max 與 λ_min 的變分刻畫
print("\nRayleigh 商的變分原理（第 22 章變分法的起點）：")
Sv = np.array([[5.0, 4.0], [4.0, 5.0]])
lamv = np.linalg.eigvalsh(Sv)
rng = np.random.default_rng(3)
vals = [rayleigh_quotient(Sv, rng.standard_normal(2)) for _ in range(5000)]
print(f"  λ_min = {lamv[0]:.4f} ≤ min(R) = {min(vals):.4f}"
      f" ≤ max(R) = {max(vals):.4f} ≤ λ_max = {lamv[-1]:.4f}")
check("λ_max = max Rayleigh 商", max(vals), lamv[-1], tol=1e-2)
check("λ_min = min Rayleigh 商", min(vals), lamv[0], tol=1e-2)

# %% [markdown]
# ## 7.4　複數向量與 Hermitian 矩陣
#
# 複數情況下，所有的「轉置」都要換成**共軛轉置** $A^{\mathsf H}=\bar A^{\mathsf T}$：
#
# | 實數 | 複數 |
# |---|---|
# | 長度 $\mathbf x^{\mathsf T}\mathbf x$ | $\mathbf x^{\mathsf H}\mathbf x=\sum|x_i|^2$ |
# | 內積 $\mathbf x^{\mathsf T}\mathbf y$ | $\mathbf x^{\mathsf H}\mathbf y$ |
# | 對稱 $S^{\mathsf T}=S$ | **Hermitian** $S^{\mathsf H}=S$ |
# | 正交 $Q^{\mathsf T}Q=I$ | **unitary** $Q^{\mathsf H}Q=I$ |
# | 列空間 $C(A^{\mathsf T})$ | $C(A^{\mathsf H})$ |
#
# Hermitian 矩陣的特徵值仍然**全為實數**、特徵向量仍然**正交** ──
# 這正是量子力學用 Hermitian 算子表示可觀測量的原因（第 21 章）。

# %%
section("7.4 Hermitian 與 unitary 矩陣")

H = np.array([[2.0 + 0j, 3.0 - 3j], [3.0 + 3j, 5.0 + 0j]])
show_matrix("H（Hermitian）", H)
check("H^H = H", H.conj().T, H)
lamH, QH = np.linalg.eigh(H)
print(f"特徵值 λ = {lamH}（實數！）")
check("Hermitian 的 λ 全為實數", np.all(np.isreal(lamH)))
check("特徵向量在複內積下正交：Q^H Q = I", QH.conj().T @ QH, np.eye(2, dtype=complex))
check("H = QΛQ^H", QH @ np.diag(lamH) @ QH.conj().T, H)
print(f"對角線元素必為實數：{np.real(np.diag(H))}；trace = Σλ = {np.trace(H).real:.4g}")
check("trace H = Σλ", np.trace(H).real, float(np.sum(lamH)))

x = np.array([1.0 + 0j, 1j])
print(f"\nx = {x}")
print(f"  錯誤的「長度平方」xᵀx = {x @ x}（不是實數，也可能是 0！）")
print(f"  正確的長度平方 x^H x = {(x.conj() @ x).real:.4g}")
check("‖x‖² = Σ|xᵢ|² = 2", (x.conj() @ x).real, 2.0)

# unitary 矩陣
U = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
theta = 0.7
Uphase = np.diag([np.exp(1j * theta), np.exp(-2j * theta)])
Umat = U @ Uphase
check("U^H U = I（unitary）", Umat.conj().T @ Umat, np.eye(2, dtype=complex))
check("unitary 保長", np.linalg.norm(Umat @ x), np.linalg.norm(x))
check("unitary 的 |λ| = 1", np.abs(np.linalg.eigvals(Umat)), np.ones(2))

# %% [markdown]
# ### Fourier 矩陣：最重要的複數矩陣
#
# 循環位移矩陣 $P$（$P^N=I$）的特徵值是 1 的 $N$ 次方根
# $\lambda_k=w^k$，$w=e^{2\pi i/N}$，而它的特徵向量矩陣就是 **Fourier 矩陣**
#
# $$F_N[j,k]=w^{jk},\qquad F_N^{\mathsf H}F_N=N\,I$$
#
# **所有循環矩陣**（Toeplitz 且首尾相接）$C=c_0I+c_1P+\dots+c_{N-1}P^{N-1}$
# 都被同一個 $F$ 對角化，特徵值是 $F\mathbf c$。由此得到訊號處理的基本定理：
#
# $$\boxed{F(\mathbf c\circledast\mathbf d)=(F\mathbf c)\odot(F\mathbf d)}
# \qquad\text{（卷積定理）}$$
#
# 而 **FFT** 把 $F_N\mathbf x$ 的成本從 $N^2$ 降到 $N\log_2N$。

# %%
section("7.4 Fourier 矩陣、循環矩陣與卷積定理")


def fourier_matrix(N):
    """F[j,k] = w^{jk}，w = e^{2πi/N}。"""
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(2j * np.pi * j * k / N)


def circulant(c):
    """由第一欄 c 造出循環矩陣（每一欄往下循環移位）。"""
    c = np.asarray(c, dtype=complex)
    N = len(c)
    return np.column_stack([np.roll(c, k) for k in range(N)])


def sort_eigs(z, decimals=9):
    """穩定地排序複數特徵值（先比實部再比虛部）。

    np.sort_complex 會因為共軛對實部的浮點誤差而把 a+bi 與 a−bi 排反，
    所以先四捨五入再 lexsort。"""
    z = np.asarray(z, dtype=complex)
    return z[np.lexsort((np.round(z.imag, decimals), np.round(z.real, decimals)))]


N = 4
F = fourier_matrix(N)
P = circulant([0.0, 1.0, 0.0, 0.0])          # 循環位移矩陣
show_matrix("P（循環位移）", np.real(P))
print(f"P⁴ = I：{np.allclose(np.linalg.matrix_power(P, 4), np.eye(4))}")
lamP = np.linalg.eigvals(P)
print(f"P 的特徵值 = {np.round(sort_eigs(lamP), 4)}（1 的四次方根 1, i, −1, −i）")
check("F^H F = N·I", F.conj().T @ F, N * np.eye(N, dtype=complex))
check("P·F = F·Λ（F 的欄是 P 的特徵向量）",
      P @ F, F @ np.diag(np.exp(-2j * np.pi * np.arange(N) / N)), tol=1e-8)

c = np.array([2.0, 1.0, 0.0, 3.0])
d = np.array([3.0, 1.0, 1.0, 4.0])
C, D = circulant(c), circulant(d)
check("循環矩陣可交換：CD = DC", C @ D, D @ C)
cd = np.real((C @ D)[:, 0])
print(f"\nc = {c}, d = {d}")
print(f"循環卷積 c ⊛ d（= CD 的第一欄）= {cd}")
check("循環卷積 = numpy 的 FFT 乘法",
      cd, np.real(np.fft.ifft(np.fft.fft(c) * np.fft.fft(d))))
check("卷積定理 F(c⊛d) = (Fc)⊙(Fd)",
      np.fft.fft(cd), np.fft.fft(c) * np.fft.fft(d), tol=1e-8)
check("循環矩陣的特徵值 = Fc（用 numpy fft）",
      sort_eigs(np.linalg.eigvals(C)), sort_eigs(np.fft.fft(c)), tol=1e-8)

# FFT 的成本
print(f"\n{'N':>8} | {'N² (直接)':>14} | {'N log₂N (FFT)':>16} | {'加速':>8}")
print("-" * 54)
for N_ in [16, 256, 1024, 1 << 20]:
    direct = N_ ** 2
    fft_cost = N_ * np.log2(N_)
    print(f"{N_:>8} | {direct:>14,} | {fft_cost:>16,.0f} | {direct / fft_cost:>7.0f}x")

# 真的驗證 FFT 的遞迴分解 F_N = [[I, B],[I, −B]] · diag(F_{N/2}, F_{N/2}) · 偶奇置換
N = 8
FN = fourier_matrix(N)
half = N // 2
Fh = fourier_matrix(half)
w = np.exp(2j * np.pi * np.arange(half) / N)
B = np.diag(w)
top = np.hstack([np.eye(half), B])
bot = np.hstack([np.eye(half), -B])
middle = np.block([[Fh, np.zeros((half, half))], [np.zeros((half, half)), Fh]])
perm = np.zeros((N, N))
for i, idx in enumerate(list(range(0, N, 2)) + list(range(1, N, 2))):
    perm[i, idx] = 1.0
check("FFT 的一步分解 F₈ = [[I,B],[I,−B]]·diag(F₄,F₄)·P",
      np.vstack([top, bot]) @ middle @ perm, FN, tol=1e-8)
x_sig = np.random.default_rng(0).standard_normal(N)
check("F₈x 與 numpy.fft 一致（共軛約定差異已處理）",
      FN.conj() @ x_sig, np.fft.fft(x_sig), tol=1e-8)

# %% [markdown]
# ### 離散正弦基底：$K$ 的特徵向量
#
# 二階差分矩陣 $K=\operatorname{tridiag}(-1,2,-1)$ 的特徵向量是**離散正弦波**
#
# $$\mathbf q_k[j]=\sin\frac{jk\pi}{n+1},\qquad
# \lambda_k=2-2\cos\frac{k\pi}{n+1}=4\sin^2\frac{k\pi}{2(n+1)}$$
#
# 這是連續問題 $-u''=\lambda u$、$u(0)=u(1)=0$ 的離散版，
# 也是第 17 章 Fourier／Sturm–Liouville 的預演。

# %%
section("7.4 K 的特徵向量就是正弦波")

n = 6
K = second_difference_matrix(n)
lamK, QK = np.linalg.eigh(K)
j = np.arange(1, n + 1)
print(f"{'k':>3} | {'數值 λ':>12} | {'4sin²(kπ/2(n+1))':>18}")
print("-" * 40)
for k in range(1, n + 1):
    theory = 4 * np.sin(k * np.pi / (2 * (n + 1))) ** 2
    idx = k - 1
    print(f"{k:>3} | {lamK[idx]:>12.8f} | {theory:>18.8f}")
theory_all = np.sort([4 * np.sin(k * np.pi / (2 * (n + 1))) ** 2
                      for k in range(1, n + 1)])
check("λₖ = 4sin²(kπ/2(n+1))", np.sort(lamK), theory_all)
for k in [1, 2, 3]:
    q_theory = np.sin(j * k * np.pi / (n + 1))
    q_theory = q_theory / np.linalg.norm(q_theory)
    q_num = QK[:, k - 1]
    if q_num @ q_theory < 0:
        q_num = -q_num
    check(f"第 {k} 個特徵向量 = 離散正弦波 sin(jkπ/(n+1))", q_num, q_theory, tol=1e-8)

fig, ax = new_axes("Eigenvectors of K are discrete sine waves",
                   figsize=(6.0, 4.0), equal=False)
jj = np.linspace(0, n + 1, 300)
for k, c in zip([1, 2, 3], ["C0", "C1", "C2"]):
    q = np.sin(j * k * np.pi / (n + 1))
    q = q / np.linalg.norm(q)
    sign = 1 if (QK[:, k - 1] @ q) > 0 else -1
    ax.plot(j, sign * QK[:, k - 1], "o", color=c, ms=6,
            label=f"k = {k} (lambda = {lamK[k-1]:.3f})")
    smooth = np.sin(jj * k * np.pi / (n + 1))
    ax.plot(jj, sign * smooth / np.linalg.norm(q) * np.linalg.norm(q) /
            np.linalg.norm(np.sin(j * k * np.pi / (n + 1))) *
            np.linalg.norm(q), color=c, lw=1, alpha=0.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("mesh point j")
ax.legend(fontsize=8)
finish(fig, "ch07_sine_eigenvectors")

# %% [markdown]
# ## 動手練習
#
# 1. 證明：若 $S$ 對稱且 $\lambda_1\ne\lambda_2$，則對應的特徵向量正交。
# 2. 找出所有使 $\begin{bmatrix}1&b\\b&9\end{bmatrix}$ 正定的 $b$（用四個不同測試各做一次）。
# 3. 寫出 $f(x,y)=x^4+y^4-4xy$ 的 Hessian，找出所有臨界點並分類。
# 4. 驗證任意循環矩陣都被 Fourier 矩陣對角化，並用 `np.fft` 計算其特徵值。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1（數值驗證）
rng = np.random.default_rng(5)
for trial in range(3):
    M = rng.standard_normal((4, 4))
    M = M + M.T
    lam, Qm = np.linalg.eigh(M)
    check(f"練習 1：隨機對稱矩陣 #{trial + 1} 的特徵向量正交",
          Qm.T @ Qm, np.eye(4))

# 練習 2
print("\n練習 2：[[1,b],[b,9]] 正定的條件")
print("  ① 特徵值：λ = 5 ± √(16 + b²) > 0 ⇔ b² < 9")
print("  ② 能量：x² + 2bxy + 9y² = (x + by)² + (9 − b²)y² > 0 ⇔ b² < 9")
print("  ④ 行列式：D₁ = 1 > 0，D₂ = 9 − b² > 0 ⇔ |b| < 3")
print("  ⑤ 主元：1 與 9 − b² 都 > 0 ⇔ |b| < 3")
for b in [0.0, 2.0, 2.999, 3.0, 3.5]:
    M = np.array([[1.0, b], [b, 9.0]])
    pd = is_positive_definite(M)
    print(f"  b = {b:<6} → 正定：{str(pd):<5}（|b| < 3：{abs(b) < 3}）")
    check(f"  b = {b} 的判定與 |b| < 3 一致", pd == (abs(b) < 3))

# 練習 3：f = x⁴ + y⁴ − 4xy
print("\n練習 3：f = x⁴ + y⁴ − 4xy")
print("  ∇f = (4x³ − 4y, 4y³ − 4x) = 0 ⇒ x³ = y 且 y³ = x ⇒ x⁹ = x")
crit = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
for pt in crit:
    x, y = pt
    Hf = np.array([[12 * x ** 2, -4.0], [-4.0, 12 * y ** 2]])
    lam = np.linalg.eigvalsh(Hf)
    kind = ("極小" if np.all(lam > 1e-12) else "極大" if np.all(lam < -1e-12)
            else "鞍點")
    print(f"  臨界點 {pt}：Hessian 的 λ = {np.round(lam, 3)} → {kind}")
    grad = np.array([4 * x ** 3 - 4 * y, 4 * y ** 3 - 4 * x])
    check(f"    ∇f = 0", grad, np.zeros(2))

# 練習 4
print("\n練習 4：")
for N_ in [4, 6, 8]:
    cc = rng.standard_normal(N_)
    Cc = circulant(cc)
    lam_fft = np.fft.fft(cc)
    check(f"  N = {N_}：循環矩陣的特徵值 = fft(c)",
          sort_eigs(np.linalg.eigvals(Cc)), sort_eigs(lam_fft), tol=1e-8)

# %% [markdown]
# ## 本章重點回顧
#
# * **譜定理**：$S=S^{\mathsf T}\Rightarrow S=Q\Lambda Q^{\mathsf T}$，
#   $\lambda$ 全為實數、特徵向量可取成正交。類比：對稱↔實數、反對稱↔虛數、正交↔單位圓。
# * **正定的五個等價測試**：$\lambda>0$、能量 $>0$、$S=A^{\mathsf T}A$、
#   主子式 $>0$、主元 $>0$。能量測試是最好的定義。
# * 第 $k$ 個主元 $=D_k/D_{k-1}$；$S=LDL^{\mathsf T}=R^{\mathsf T}R$（Cholesky）。
# * 正定性 = **嚴格凸** = Hessian 判定的極小值；能量等高線是橢圓，
#   主軸沿特徵向量、半軸 $1/\sqrt{\lambda}$。Rayleigh 商的極值就是 $\lambda_{\min},\lambda_{\max}$。
# * 複數世界把 $A^{\mathsf T}$ 全部換成 $A^{\mathsf H}$：Hermitian 取代對稱、unitary 取代正交，
#   結論完全平行。
# * Fourier 矩陣對角化**所有**循環矩陣 ⇒ 卷積定理 ⇒ FFT（$N\log N$）。
# * $K=\operatorname{tridiag}(-1,2,-1)$ 的特徵向量是離散正弦波，
#   預告了第 17 章的 Fourier 與 Sturm–Liouville 理論。
#
# 下一章：用特徵值解微分方程 ── 矩陣指數 $e^{At}$ 與穩定性。
