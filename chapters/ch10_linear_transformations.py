# %% [markdown]
# # 第 10 章　線性轉換、基底變換與 Jordan 形式
#
# > 對應 Strang 第 8 章（8.1 線性轉換的想法、8.2 轉換的矩陣、8.3 尋找好的基底）、
# > 附錄 5（Jordan 形式）、附錄 10（計算機圖學），以及 Riley 8.2（線性算子）、
# > 8.15（基底變換與相似變換）、26.2（座標變換）。
#
# 前九章從矩陣出發。本章反過來問：**不先給座標**，線性轉換是什麼？
#
# $$T(c\mathbf v+d\mathbf w)=cT(\mathbf v)+dT(\mathbf w)$$
#
# 兩個關鍵事實：
#
# 1. **基底決定一切**：知道 $T$ 對基底的作用，就知道 $T$ 對所有向量的作用
# 2. **選了基底就得到矩陣**：$T$ 的矩陣 $=$ 把 $T(\mathbf v_j)$ 用輸出基底展開的係數
#
# 整本書的高潮：**什麼是好的基底？**
#
# | 基底 | 矩陣變成 | 條件 |
# |---|---|---|
# | 特徵向量 $X$ | 對角 $\Lambda$ | 可對角化 |
# | 奇異向量 $V,U$ | 對角 $\Sigma$ | 永遠可以 |
# | 廣義特徵向量 $B$ | Jordan 形 $J$ | 永遠可以 |
# | Fourier 矩陣 $F$ | 對角（所有循環矩陣）| 常係數／週期 |
# | Legendre／Chebyshev | 近似對角 | 函數空間 |
#
# ```bash
# python chapters/ch10_linear_transformations.py
# python tools/build_notebooks.py ch10
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
from linalg_tutorial.svd_tools import condition_number
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 10.1　什麼是線性轉換
#
# 必要條件：$T(\mathbf 0)=\mathbf 0$。所以「平移」$T(\mathbf v)=A\mathbf v+\mathbf u_0$
# **不是**線性的（它叫 **affine**，計算機圖學靠它平移圖形）。
#
# 只要輸出含有平方、乘積或長度（$v_1^2$、$v_1v_2$、$\|\mathbf v\|$），就不是線性。
#
# | 轉換 | 線性？ |
# |---|---|
# | $T(\mathbf v)=A\mathbf v$ | ✓ |
# | 旋轉、投影、反射 | ✓ |
# | 微分 $df/dx$、積分 $\int_0^x f$ | ✓ |
# | $T(\mathbf v)=\|\mathbf v\|$ | ✗ |
# | $T(\mathbf v)=\mathbf v+\mathbf u_0$ | ✗（affine）|
# | $T(x,y,z)=(x,y,1)$ | ✗ |

# %%
section("10.1 線性測試")


def is_linear(T, dim, trials=200, seed=0, tol=1e-9):
    """隨機抽 v, w, c, d 檢查 T(cv+dw) = cT(v)+dT(w)。"""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        v, w = rng.standard_normal(dim), rng.standard_normal(dim)
        c, d = rng.standard_normal(2)
        lhs = np.atleast_1d(T(c * v + d * w))
        rhs = c * np.atleast_1d(T(v)) + d * np.atleast_1d(T(w))
        if lhs.shape != rhs.shape or not np.allclose(lhs, rhs, atol=tol):
            return False
    return True


A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
u0 = np.array([1.0, 1.0])
transformations = {
    "T(v) = Av（矩陣乘法）": (lambda v: A @ v, 3, True),
    "T(v) = a·v（點積）": (lambda v: np.array([1.0, 3.0, 4.0]) @ v, 3, True),
    "旋轉 30°": (lambda v: np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                                     [np.sin(np.pi / 6), np.cos(np.pi / 6)]]) @ v, 2, True),
    "T(v) = ‖v‖（長度）": (lambda v: np.linalg.norm(v), 3, False),
    "T(v) = Av + u₀（平移 = affine）": (lambda v: A @ v + u0, 3, False),
    "T(x,y,z) = (x,y,1)": (lambda v: np.array([v[0], v[1], 1.0]), 3, False),
    "T(v) = v₁v₂（乘積）": (lambda v: v[0] * v[1], 2, False),
}
for name, (T, dim, expected) in transformations.items():
    got = is_linear(T, dim)
    print(f"  {name:<32} 線性：{str(got):<5}（預期 {expected}）")
    check(f"  判定正確", got == expected)
print("\n關鍵：T(0) = 0 是必要條件")
check("affine 轉換不把 0 送到 0", not np.allclose(A @ np.zeros(3) + u0, 0))

# %% [markdown]
# ### 線性轉換把直線送到直線、等距點送到等距點
#
# Strang 的「房子」是最好的視覺教材：11 個角點放進 $2\times12$ 矩陣 $H$，
# 用 $AH$ 看各種矩陣對房子做了什麼。

# %%
section("10.1 線性轉換的視覺化：Strang 的房子")

H = np.array([
    [-6.0, -6, -7, 0, 7, 6, 6, -3, -3, 0, 0, -6],
    [-7.0, 2, 1, 8, 1, 2, -7, -7, -2, -2, -7, -7],
])
theta = np.radians(35)
mats = {
    "original (I)": np.eye(2),
    "shear [[1,0],[1,1]]": np.array([[1.0, 0.0], [1.0, 1.0]]),
    "rotate 35°": np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]]),
    "[[0.7,0.3],[0.3,0.7]]": np.array([[0.7, 0.3], [0.3, 0.7]]),
    "project onto y=x": np.array([[0.5, 0.5], [0.5, 0.5]]),
}
fig = plt.figure(figsize=(13.0, 2.9))
for k, (name, M) in enumerate(mats.items()):
    ax = fig.add_subplot(1, 5, k + 1)
    pts = M @ H
    ax.plot(np.append(pts[0], pts[0, 0]), np.append(pts[1], pts[1, 0]), "C0-", lw=1.6)
    ax.set_title(f"{name}\ndet = {np.linalg.det(M):.2f}", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.grid(alpha=0.3)
finish(fig, "ch10_houses")

# 直線送到直線、中點送到中點
M = mats["shear [[1,0],[1,1]]"]
v, w = H[:, 0], H[:, 3]
mid = 0.5 * (v + w)
check("中點 → 中點：T((v+w)/2) = (T(v)+T(w))/2", M @ mid, 0.5 * (M @ v + M @ w))
ts = np.linspace(0, 1, 11)
line_in = np.array([v + t * (w - v) for t in ts]).T
line_out = M @ line_in
spacing = np.linalg.norm(np.diff(line_out, axis=1), axis=0)
check("等距點 → 等距點（間距全部相同）", spacing, np.full(10, spacing[0]))
print(f"投影矩陣的 det = 0 ⇒ 房子被壓成一條線（面積變 0）")
check("投影把房子壓到一維", np.linalg.matrix_rank(mats["project onto y=x"] @ H) == 1)
print(f"各轉換的面積放大率 |det|：")
for name, Mt in mats.items():
    print(f"  {name:<24} |det| = {abs(np.linalg.det(Mt)):.4f}")

# %% [markdown]
# ## 10.2　微積分裡的線性轉換
#
# $T=\dfrac{d}{dx}$ 與 $T^{+}=\int_0^x$ 都是線性的。選定基底後它們就是矩陣：
#
# 輸入基底 $1,x,x^2,x^3$，輸出基底 $1,x,x^2$：
#
# $$A_{d/dx}=\begin{bmatrix}0&1&0&0\\0&0&2&0\\0&0&0&3\end{bmatrix},\qquad
# A_{\int}=\begin{bmatrix}0&0&0\\1&0&0\\0&\frac12&0\\0&0&\frac13\end{bmatrix}$$
#
# **微積分基本定理的線性代數版本**：
# $A_{d/dx}A_{\int}=I$（先積分再微分回到原點），
# 但 $A_{\int}A_{d/dx}\ne I$（先微分會丟掉常數項）── 常數函數就是 $d/dx$ 的 **kernel**。
#
# | 轉換的語言 | 矩陣的語言 |
# |---|---|
# | range（值域）| 欄空間 $C(A)$ |
# | kernel（核）| 零空間 $N(A)$ |

# %%
section("10.2 微分與積分的矩陣")

D = np.array([                      # 輸入 1,x,x²,x³ → 輸出 1,x,x²
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 3.0],
])
Int = np.array([                    # 輸入 1,x,x² → 輸出 1,x,x²,x³
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1 / 3],
])
show_matrix("微分矩陣 A", D)
show_matrix("積分矩陣 A⁺", Int)

coef = np.array([6.0, -4.0, 3.0, 0.0])     # u = 6 − 4x + 3x²
print(f"\nu = 6 − 4x + 3x²（係數 {coef}）")
print(f"  du/dx 的係數 = {D @ coef} → −4 + 6x")
check("微分矩陣算出正確導數", D @ coef, np.array([-4.0, 6.0, 0.0]))

check("A·A⁺ = I（先積分再微分 → 回到原點）", D @ Int, np.eye(3))
show_matrix("A⁺·A（先微分再積分 → 第一欄全 0，常數項丟了）", Int @ D)
check("A⁺A ≠ I", not np.allclose(Int @ D, np.eye(4)))
check("A⁺A 的第一欄全為 0（常數函數是 kernel）", (Int @ D)[:, 0], np.zeros(4))

print(f"\nkernel(d/dx) = 常數函數，維度 1")
print(f"range(d/dx) = 所有二次以下多項式，維度 3")
print(f"計數定理仍然成立：dim range + dim kernel = 3 + 1 = 4 = dim 輸入空間")
check("rank + nullity = n", np.linalg.matrix_rank(D) + (4 - np.linalg.matrix_rank(D)), 4)
check("A⁺ 是 A 的偽逆嗎？（這裡 A⁺ 恰好就是 Moore–Penrose 偽逆）",
      Int, pseudoinverse(D), tol=1e-10)

# 用實際函數驗證
xs = np.linspace(0, 1, 7)
poly = lambda c, x: sum(c[k] * x ** k for k in range(len(c)))
h = 1e-6
for x0 in [0.2, 0.7]:
    numeric = (poly(coef, x0 + h) - poly(coef, x0 - h)) / (2 * h)
    check(f"x = {x0}：矩陣導數 = 數值導數", poly(D @ coef, x0), numeric, tol=1e-6)

# %% [markdown]
# ## 10.3　轉換的矩陣與基底變換
#
# **建構法則**：第 $j$ 欄 $=$ 把 $T(\mathbf v_j)$ 用輸出基底 $\mathbf w_1,\dots,\mathbf w_m$ 展開的係數
#
# $$T(\mathbf v_j)=a_{1j}\mathbf w_1+\dots+a_{mj}\mathbf w_m$$
#
# **基底變換**（$T=I$ 的情形）：輸入基底在 $V$ 的欄、輸出基底在 $W$ 的欄，則
#
# $$\boxed{B=W^{-1}V}$$
#
# 世界級的混淆點：標準基底換成 $W$ 時，座標要乘 $W^{-1}$（**基底變大，座標變小**）。
#
# **矩陣乘法的真正理由**：$TS$ 對應 $AB$。

# %%
section("10.3 轉換的矩陣：第 j 欄 = T(vⱼ) 的展開係數")

# T: R² → R³，T(1,0) = (2,3,4)，T(0,1) = (5,5,5)
T_std = np.column_stack([np.array([2.0, 3.0, 4.0]), np.array([5.0, 5.0, 5.0])])
show_matrix("標準基底下的矩陣 A", T_std)
check("A·(1,1) = T(v₁) + T(v₂)", T_std @ np.array([1.0, 1.0]),
      np.array([7.0, 8.0, 9.0]))

# 基底變換 B = W⁻¹V
V = np.array([[3.0, 6.0], [3.0, 8.0]])      # 輸入基底（欄）
W = np.array([[3.0, 0.0], [1.0, 2.0]])      # 輸出基底（欄）
B = np.linalg.inv(W) @ V
show_matrix("輸入基底 V", V)
show_matrix("輸出基底 W", W)
show_matrix("基底變換矩陣 B = W⁻¹V", B)
check("W·B = V", W @ B, V)
print("意思是：v₁ = 1·w₁ + 1·w₂，v₂ = 2·w₁ + 3·w₂")
check("v₁ = 1·w₁ + 1·w₂", W @ B[:, 0], V[:, 0])
check("v₂ = 2·w₁ + 3·w₂", W @ B[:, 1], V[:, 1])

u = np.array([9.0, 11.0])
c = np.linalg.solve(V, u)
d = np.linalg.solve(W, u)
print(f"\n同一個向量 u = {u}：")
print(f"  在 v 基底下的座標 c = {np.round(c, 4)}")
print(f"  在 w 基底下的座標 d = {np.round(d, 4)}")
check("d = B·c（座標的變換）", d, B @ c)
check("Vc = Wd = u", V @ c, W @ d)
print("  注意：基底矩陣是 W，但座標要乘 W⁻¹ —— 基底變大，座標變小")

# TS ↔ AB：兩次旋轉 = 旋轉兩倍角
section("10.3 矩陣乘法 = 轉換的合成")

th = np.radians(25)
R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
check("R(θ)·R(θ) = R(2θ)", R(th) @ R(th), R(2 * th))
check("R(θ)·R(−θ) = I", R(th) @ R(-th), np.eye(2))
print(f"由 R(θ)² = R(2θ) 讀出三角恆等式：")
print(f"  cos 2θ = cos²θ − sin²θ：{np.cos(2 * th):.8f} = "
      f"{np.cos(th) ** 2 - np.sin(th) ** 2:.8f}")
print(f"  sin 2θ = 2 sin θ cos θ：{np.sin(2 * th):.8f} = "
      f"{2 * np.sin(th) * np.cos(th):.8f}")
check("cos 2θ = cos²θ − sin²θ", np.cos(2 * th), np.cos(th) ** 2 - np.sin(th) ** 2)
check("sin 2θ = 2 sinθ cosθ", np.sin(2 * th), 2 * np.sin(th) * np.cos(th))
print("  ⇒ 矩陣乘法的定義「就是」轉換合成的定義，這才是它長那樣的原因")

# 先積分再微分 vs 先微分再積分（合成的順序很重要）
check("d/dx ∘ ∫ = I（AB）", D @ Int, np.eye(3))
check("∫ ∘ d/dx ≠ I（BA）", not np.allclose(Int @ D, np.eye(4)))

# %% [markdown]
# ## 10.4　尋找好的基底
#
# 同一個轉換 $T$，換基底就換矩陣：
#
# $$A_{\text{new}}=B_{\text{out}}^{-1}\,A\,B_{\text{in}}$$
#
# 四個經典好選擇：
#
# 1. $B_{\text{in}}=B_{\text{out}}=X$（特徵向量）→ $\Lambda$（對角）
# 2. $B_{\text{in}}=V,\ B_{\text{out}}=U$（奇異向量）→ $\Sigma$（對角，**永遠可以**）
# 3. $B_{\text{in}}=B_{\text{out}}=B$（廣義特徵向量）→ Jordan 形 $J$
# 4. $B_{\text{in}}=B_{\text{out}}=F$（Fourier）→ 對角化**所有**循環矩陣

# %%
section("10.4 同一個投影，三種基底三種矩陣")

# 投影到 y = −x 這條線
A_proj = np.array([[0.5, -0.5], [-0.5, 0.5]])
show_matrix("標準基底下：A（不是對角）", A_proj)
check("P² = P", A_proj @ A_proj, A_proj)

lam, X = np.linalg.eigh(A_proj)
show_matrix("特徵向量基底下：X⁻¹AX = Λ（對角！）", np.linalg.inv(X) @ A_proj @ X)
check("換成特徵向量基底後是對角矩陣",
      np.linalg.inv(X) @ A_proj @ X, np.diag(lam), tol=1e-10)
print(f"  λ = {lam}：投影矩陣的特徵值只有 0 和 1")
print(f"  λ=1 的特徵向量 = 直線方向（投影後不動）")
print(f"  λ=0 的特徵向量 = 垂直方向（投影成 0）")

rng = np.random.default_rng(0)
Bany = rng.standard_normal((2, 2))
A_sim = np.linalg.inv(Bany) @ A_proj @ Bany
print(f"\n任意基底 B：B⁻¹AB 的特徵值 = {np.sort(np.real(np.linalg.eigvals(A_sim)))}"
      f"（與原本相同，但矩陣不再對角）")
check("相似變換保留特徵值", np.sort(np.real(np.linalg.eigvals(A_sim))), np.sort(lam),
      tol=1e-8)

# 非方陣：只能用奇異向量
A_rect = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0]])
U, s, Vt = np.linalg.svd(A_rect)
Sigma = U.T @ A_rect @ Vt.T
show_matrix("奇異向量基底下：U⁻¹AV = Σ（對角，長方形也行！）", Sigma)
check("Uᵀ A V = Σ", np.abs(Sigma[:, :2]), np.diag(s), tol=1e-10)
print("  長方形矩陣沒有特徵值，但永遠有奇異值 ⇒ SVD 是最通用的「好基底」")

# %% [markdown]
# ### Jordan 形式：特徵向量不夠時的最佳替代
#
# 若 $A$ 只有 $s$ 個獨立特徵向量，則它相似於有 $s$ 個 **Jordan 塊**的 $J$：
#
# $$J_i=\begin{bmatrix}\lambda_i&1&&\\&\lambda_i&\ddots&\\&&\ddots&1\\&&&\lambda_i\end{bmatrix}$$
#
# **每個缺少的特徵向量就對應一個對角線上方的 1**。
# 廣義特徵向量滿足 $(A-\lambda I)\mathbf b_{k+1}=\mathbf b_k$。
#
# Jordan 形式讓 $e^{Jt}$ 可以寫下來（會出現 $t,t^2,\dots$ 乘上 $e^{\lambda t}$），
# 但**數值上不穩定**（擾動一下重根就分開了），實務上用 SVD 或 Schur 分解。

# %%
section("10.4 Jordan 形式")

J = np.zeros((4, 4))
J[0, 0] = J[1, 1] = 2.0
J[2, 2] = J[3, 3] = 3.0
J[2, 3] = 1.0                 # λ=3 的 2x2 Jordan 塊
show_matrix("Jordan 矩陣 J（λ = 2,2,3,3）", J)
for l in [2.0, 3.0]:
    AM = int(np.sum(np.isclose(np.linalg.eigvals(J), l)))
    GM = 4 - np.linalg.matrix_rank(J - l * np.eye(4), tol=1e-10)
    print(f"  λ = {l}：代數重數 AM = {AM}，幾何重數 GM = {GM}"
          f"（缺 {AM - GM} 個特徵向量 ⇒ 對角線上方有 {AM - GM} 個 1）")
check("λ=2 有兩個獨立特徵向量（兩個 1x1 塊）",
      4 - np.linalg.matrix_rank(J - 2 * np.eye(4), tol=1e-10), 2)
check("λ=3 只有一個特徵向量（一個 2x2 塊）",
      4 - np.linalg.matrix_rank(J - 3 * np.eye(4), tol=1e-10), 1)

e3, e4 = np.eye(4)[2], np.eye(4)[3]
check("x₃ 是真正的特徵向量：(J − 3I)x₃ = 0", (J - 3 * np.eye(4)) @ e3, np.zeros(4))
check("x₄ 是廣義特徵向量：(J − 3I)x₄ = x₃", (J - 3 * np.eye(4)) @ e4, e3)

# 相似於 J 的矩陣仍有同樣的 Jordan 結構
rng = np.random.default_rng(1)
Bj = rng.standard_normal((4, 4))
C = Bj @ J @ np.linalg.inv(Bj)
print(f"\nC = BJB⁻¹ 的特徵值 = {np.sort(np.real(np.linalg.eigvals(C)))}")
check("C 與 J 有相同特徵值", np.sort(np.real(np.linalg.eigvals(C))),
      np.array([2.0, 2.0, 3.0, 3.0]), tol=1e-6)
check("C 也只有 3 個獨立特徵向量",
      np.linalg.matrix_rank(np.linalg.eig(C)[1], tol=1e-6), 3)

# e^{Jt} 出現 t·e^{λt}
from linalg_tutorial.eigen import matrix_exp_series
t = 0.5
EJ = matrix_exp_series(J, t, 80)
print(f"\ne^{{Jt}}（t = {t}）的 2x2 塊 = [[e^{{3t}}, t·e^{{3t}}], [0, e^{{3t}}]]：")
print(f"  {np.round(EJ[2:4, 2:4], 6)}  vs 理論 "
      f"{np.round(np.array([[np.exp(3 * t), t * np.exp(3 * t)], [0, np.exp(3 * t)]]), 6)}")
check("e^{Jt} 的 Jordan 塊含 t·e^{λt}", EJ[2, 3], t * np.exp(3 * t))

# 數值不穩定性示範
print("\nJordan 形式的數值脆弱性：")
for eps in [0.0, 1e-10, 1e-6]:
    Jp = J.copy()
    Jp[3, 3] += eps
    n_eig = np.linalg.matrix_rank(np.linalg.eig(Jp)[1], tol=1e-12)
    print(f"  把 J[3,3] 加上 {eps:g} → 獨立特徵向量數 = {n_eig}"
          f"（{'仍缺' if n_eig < 4 else '變成可對角化！'}）")
check("任意小的擾動就讓重根分開、矩陣變成可對角化",
      np.linalg.matrix_rank(np.linalg.eig(J + np.diag([0, 0, 0, 1e-10]))[1],
                            tol=1e-14) == 4)

# %% [markdown]
# ## 10.5　函數空間的好基底
#
# 把向量換成函數，內積換成積分：
#
# $$\langle f,g\rangle=\int_{-1}^{1}f(x)g(x)\,dx,\qquad
# \|f\|^2=\int_{-1}^{1}f(x)^2dx$$
#
# 這就是 **Hilbert 空間**（第 17 章的主角）。
#
# 最直覺的基底 $1,x,x^2,x^3,\dots$ 是**最糟的基底**：
# 它的 Gram 矩陣就是惡名昭彰的 **Hilbert 矩陣**，條件數指數爆炸。
#
# 三個經典的好基底：
#
# | 基底 | 正交條件 | 用途 |
# |---|---|---|
# | Fourier $e^{ikx}$ | $\int_0^{2\pi}$ | 週期問題、常係數 |
# | Legendre $P_n(x)$ | $\int_{-1}^1$ | 球座標、多項式逼近 |
# | Chebyshev $T_n(x)$ | $\int_{-1}^1\frac{dx}{\sqrt{1-x^2}}$ | 最佳一致逼近、數值積分 |

# %%
section("10.5 單項式基底為什麼很糟：Gram 矩陣 = Hilbert 矩陣")

print(f"{'n':>4} | {'κ(Gram) 單項式':>18} | {'κ(Gram) Legendre':>18} | {'κ(Gram) Chebyshev':>19}")
print("-" * 68)
# 用高斯–勒讓德求積點精確計算內積
nodes, weights = np.polynomial.legendre.leggauss(400)
for n in [4, 6, 8, 10, 12]:
    mono = np.array([nodes ** k for k in range(n)])
    leg = np.array([np.polynomial.legendre.Legendre.basis(k)(nodes) for k in range(n)])
    cheb = np.array([np.polynomial.chebyshev.Chebyshev.basis(k)(nodes) for k in range(n)])
    conds = []
    for basis in (mono, leg, cheb):
        G = (basis * weights) @ basis.T            # Gram 矩陣 Gᵢⱼ = ∫ bᵢbⱼ dx
        G = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))   # 先正規化再比條件數
        conds.append(condition_number(G))
    print(f"{n:>4} | {conds[0]:>18.4g} | {conds[1]:>18.4g} | {conds[2]:>19.4g}")

n = 8
mono = np.array([nodes ** k for k in range(n)])
G_mono = (mono * weights) @ mono.T
show_matrix("單項式的 Gram 矩陣（n = 8，∫₋₁¹ xⁱ⁺ʲ dx）", G_mono[:4, :4])
H_like = np.array([[2.0 / (i + j + 1) if (i + j) % 2 == 0 else 0.0
                    for j in range(n)] for i in range(n)])
check("Gram 矩陣 = ∫₋₁¹ xⁱ⁺ʲdx（偶數次才非零）", G_mono, H_like, tol=1e-10)
print(f"  κ = {condition_number(G_mono):.4g} ← 近乎奇異，x⁷ 幾乎是低次項的組合")

leg = np.array([np.polynomial.legendre.Legendre.basis(k)(nodes) for k in range(n)])
G_leg = (leg * weights) @ leg.T
show_matrix("Legendre 的 Gram 矩陣（對角！）", G_leg[:4, :4])
check("Legendre 多項式兩兩正交", G_leg, np.diag(np.diag(G_leg)), tol=1e-10)
check("‖Pₙ‖² = 2/(2n+1)", np.diag(G_leg),
      np.array([2 / (2 * k + 1) for k in range(n)]), tol=1e-10)
print(f"  κ（正規化後）= 1 ← 完美的基底")

# Chebyshev 在它自己的權重下正交
cheb_nodes = np.cos((np.arange(1, 2001) - 0.5) * np.pi / 2000)      # Chebyshev–Gauss
cheb_w = np.full(2000, np.pi / 2000)
cheb = np.array([np.polynomial.chebyshev.Chebyshev.basis(k)(cheb_nodes) for k in range(6)])
G_cheb = (cheb * cheb_w) @ cheb.T
check("Chebyshev 在權重 1/√(1−x²) 下正交", G_cheb, np.diag(np.diag(G_cheb)), tol=1e-8)
print(f"  ‖T₀‖² = π，其餘 ‖Tₙ‖² = π/2：{np.round(np.diag(G_cheb), 6)}")

# 逼近實驗：同樣次數下誤差差多少
section("10.5 逼近 f(x) = 1/(1+25x²)（Runge 函數）")

f = lambda x: 1.0 / (1 + 25 * x ** 2)
fine = np.linspace(-1, 1, 2001)
print(f"{'次數':>6} | {'單項式（等距點）':>20} | {'Chebyshev 點':>18} | {'Legendre 投影':>18}")
print("-" * 70)
for deg in [4, 8, 12, 16]:
    # (a) 等距點 + 單項式基底（Runge 現象）
    xe = np.linspace(-1, 1, deg + 1)
    c_mono = np.linalg.solve(np.vander(xe, deg + 1, increasing=True), f(xe))
    err_mono = np.max(np.abs(np.vander(fine, deg + 1, increasing=True) @ c_mono - f(fine)))
    # (b) Chebyshev 點 + 內插
    xc = np.cos(np.pi * np.arange(deg + 1) / deg)
    c_cheb = np.polynomial.chebyshev.Chebyshev.fit(xc, f(xc), deg)
    err_cheb = np.max(np.abs(c_cheb(fine) - f(fine)))
    # (c) Legendre 最小平方投影（用求積點）
    legb = np.array([np.polynomial.legendre.Legendre.basis(k)(nodes) for k in range(deg + 1)])
    coefs = [(weights * legb[k] * f(nodes)).sum() / (2 / (2 * k + 1)) for k in range(deg + 1)]
    approx = sum(coefs[k] * np.polynomial.legendre.Legendre.basis(k)(fine)
                 for k in range(deg + 1))
    err_leg = np.max(np.abs(approx - f(fine)))
    print(f"{deg:>6} | {err_mono:>20.4e} | {err_cheb:>18.4e} | {err_leg:>18.4e}")
check("Chebyshev 點的誤差隨次數下降（等距點會發散）", err_cheb < err_mono)
print("  ⇒ Runge 現象：等距點 + 高次多項式會在端點劇烈振盪；換基底／換節點就解決了")

fig, ax = new_axes("Runge function: basis and nodes matter", figsize=(6.4, 4.2),
                   equal=False)
ax.plot(fine, f(fine), "k-", lw=2, label="f(x) = 1/(1+25x^2)")
deg = 12
xe = np.linspace(-1, 1, deg + 1)
c_mono = np.linalg.solve(np.vander(xe, deg + 1, increasing=True), f(xe))
ax.plot(fine, np.vander(fine, deg + 1, increasing=True) @ c_mono, "C3--", lw=1.4,
        label=f"equispaced, degree {deg}")
xc = np.cos(np.pi * np.arange(deg + 1) / deg)
ax.plot(fine, np.polynomial.chebyshev.Chebyshev.fit(xc, f(xc), deg)(fine), "C0", lw=1.4,
        label=f"Chebyshev nodes, degree {deg}")
ax.set_ylim(-0.6, 1.4)
ax.set_xlabel("x")
ax.legend(fontsize=8)
finish(fig, "ch10_runge")

# %% [markdown]
# ## 動手練習
#
# 1. 判斷下列是否線性：$T(A)=A^{\mathsf T}$、$T(A)=A^2$、$T(A)=\operatorname{trace}(A)$、
#    $T(A)=\det(A)$。（在矩陣空間上）
# 2. 寫出「對 $x$ 微分兩次」在基底 $1,x,x^2,x^3,x^4$ 下的矩陣，確認它是第 1 題微分矩陣的平方。
# 3. 給定兩組 $\mathbb R^2$ 的基底，算出基底變換矩陣，並驗證同一向量的兩組座標關係。
# 4. 找一個 $3\times3$ 矩陣，它有 $\lambda=5,5,5$ 但只有一個特徵向量。寫出它的 Jordan 形式。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1：矩陣空間上的轉換
print("練習 1：")
rng = np.random.default_rng(2)


def is_linear_matrix(T, n=3, trials=100):
    for _ in range(trials):
        A1, A2 = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        c, d = rng.standard_normal(2)
        if not np.allclose(np.atleast_1d(T(c * A1 + d * A2)),
                           c * np.atleast_1d(T(A1)) + d * np.atleast_1d(T(A2))):
            return False
    return True


for name, T, expect in [("T(A) = Aᵀ", lambda M: M.T, True),
                        ("T(A) = A²", lambda M: M @ M, False),
                        ("T(A) = trace(A)", lambda M: np.trace(M), True),
                        ("T(A) = det(A)", lambda M: np.linalg.det(M), False)]:
    got = is_linear_matrix(T)
    print(f"  {name:<18} 線性：{got}（預期 {expect}）")
    check(f"  判定正確", got == expect)

# 練習 2
D5 = np.zeros((4, 5))
for k in range(1, 5):
    D5[k - 1, k] = k                      # d/dx: xᵏ → k xᵏ⁻¹
D5_2 = np.zeros((3, 5))
for k in range(2, 5):
    D5_2[k - 2, k] = k * (k - 1)          # d²/dx²
print(f"\n練習 2：")
show_matrix("  d/dx 矩陣（5→4）", D5)
show_matrix("  d²/dx² 矩陣（5→3）", D5_2)
check("  d²/dx² = (d/dx) 的兩次合成", D5[:3, :4] @ D5, D5_2)
c_test = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
check("  對 1−2x+3x²+0.5x³−x⁴ 微分兩次正確",
      D5_2 @ c_test, np.array([6.0, 3.0, -12.0]))

# 練習 3
V2 = np.array([[1.0, 1.0], [0.0, 1.0]])
W2 = np.array([[2.0, 0.0], [1.0, 3.0]])
B2 = np.linalg.inv(W2) @ V2
u2 = np.array([5.0, 2.0])
print(f"\n練習 3：B = W⁻¹V =\n{np.round(B2, 4)}")
check("  W·B = V", W2 @ B2, V2)
check("  d = B·c", np.linalg.solve(W2, u2), B2 @ np.linalg.solve(V2, u2))

# 練習 4
J3 = np.array([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [0.0, 0.0, 5.0]])
print(f"\n練習 4：J =\n{J3}")
print(f"  特徵值 = {np.linalg.eigvals(J3)}（都是 5）")
gm = 3 - np.linalg.matrix_rank(J3 - 5 * np.eye(3), tol=1e-10)
print(f"  幾何重數 = {gm}（只有一個特徵向量 (1,0,0)）")
check("  AM = 3, GM = 1 ⇒ 一個 3x3 的 Jordan 塊", gm == 1)
check("  (J−5I)³ = 0（冪零）", np.linalg.matrix_power(J3 - 5 * np.eye(3), 3),
      np.zeros((3, 3)))
EJ3 = matrix_exp_series(J3, 1.0, 60)
print(f"  e^J = e⁵·[[1, 1, 1/2], [0, 1, 1], [0, 0, 1]]：")
print(f"  {np.round(EJ3 / np.exp(5.0), 6)}")
check("  e^J 的右上角 = t²/2 · e^{5t}（t=1）", EJ3[0, 2], 0.5 * np.exp(5.0))

# %% [markdown]
# ## 本章重點回顧
#
# * 線性轉換只要求 $T(c\mathbf v+d\mathbf w)=cT(\mathbf v)+dT(\mathbf w)$；
#   平移是 affine 而非線性。長度、平方、乘積都不線性。
# * **基底決定一切**：知道 $T$ 對基底的作用就知道全部。
#   選定輸入／輸出基底後，$T$ 的矩陣的第 $j$ 欄就是 $T(\mathbf v_j)$ 的展開係數。
# * 基底變換矩陣 $B=W^{-1}V$。基底變大，座標變小。
# * **矩陣乘法 $AB$ 對應轉換合成 $TS$** ── 這才是矩陣乘法定義的真正來源
#   （順手證明了二倍角公式）。
# * 微分／積分是線性轉換；它們的矩陣互為偽逆，$A_{d}A_{\int}=I$ 但 $A_{\int}A_{d}\ne I$
#   （常數函數是 kernel）── 微積分基本定理的線性代數版。
# * 好基底：特徵向量 → $\Lambda$；奇異向量 → $\Sigma$（永遠可行）；
#   廣義特徵向量 → Jordan 形 $J$（每個缺少的特徵向量對應一個 1）；Fourier → 對角化循環矩陣。
# * 函數空間：單項式 $1,x,x^2,\dots$ 的 Gram 矩陣是 Hilbert 矩陣，極度病態；
#   Legendre／Chebyshev／Fourier 才是好基底（Runge 現象的解藥）。
#
# 下一章：最佳化中的線性代數 ── 梯度、Hessian、牛頓法、Lagrange 乘子與對偶。
