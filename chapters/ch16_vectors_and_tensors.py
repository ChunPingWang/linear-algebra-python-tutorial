# %% [markdown]
# # 第 16 章　向量代數、叉積與張量
#
# > 對應 Riley 第 7 章（7.1–7.9 向量代數、叉積、三重積、直線與平面、倒向量）
# > 與第 26 章（26.1–26.17 張量：座標變換、Cartesian 張量、$\delta_{ij}$ 與 $\epsilon_{ijk}$、
# > 等向張量、偽張量、對偶張量、物理應用、度規張量），
# > 並對照 Strang 第 5 章（行列式與體積）、附錄 6（張量）。
#
# 張量的核心問題：**座標變換下什麼東西不變？**
#
# | 階 | 名稱 | 變換律 | 例子 |
# |---|---|---|---|
# | 0 | 純量 | $\phi'=\phi$ | 溫度、質量 |
# | 1 | 向量 | $v_i'=L_{ij}v_j$ | 位置、速度、力 |
# | 2 | 二階張量 | $T_{ij}'=L_{ik}L_{jl}T_{kl}$ | 慣性張量、應力、導電率 |
# | 3 | 三階張量 | $T_{ijk}'=L_{il}L_{jm}L_{kn}T_{lmn}$ | 壓電係數、$\epsilon_{ijk}$ |
#
# 其中 $L$ 是正交矩陣（旋轉）。**不是每個有下標的東西都是張量** ──
# 必須滿足變換律才算。
#
# ```bash
# python chapters/ch16_vectors_and_tensors.py
# python tools/build_notebooks.py ch16
# ```

# %%
import pathlib
import sys
from itertools import permutations, product

_r = pathlib.Path(globals().get("__file__", "_")).resolve().parent.parent
if not (_r / "linalg_tutorial").is_dir():
    _r = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "linalg_tutorial").is_dir())
sys.path.insert(0, str(_r))

import numpy as np

from linalg_tutorial.determinant import det_lu, volume
from linalg_tutorial.eigen import is_positive_definite
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 16.1　叉積的矩陣面貌
#
# 叉積可以寫成**反對稱矩陣**乘法：
#
# $$\mathbf a\times\mathbf b=[\mathbf a]_{\times}\mathbf b,\qquad
# [\mathbf a]_{\times}=\begin{bmatrix}0&-a_3&a_2\\a_3&0&-a_1\\-a_2&a_1&0\end{bmatrix}$$
#
# 這個對應是「$3\times3$ 反對稱矩陣 $\leftrightarrow$ 三維向量」的同構
# （Riley 26.11 的**對偶張量**），也是剛體角速度 $\boldsymbol\omega$ 與
# $\dot R=[\boldsymbol\omega]_{\times}R$ 的關係。
#
# 三重積就是行列式：
#
# $$\mathbf a\cdot(\mathbf b\times\mathbf c)=\det[\mathbf a\ \mathbf b\ \mathbf c]
# =\text{平行六面體的（有號）體積}$$

# %%
section("16.1 叉積 = 反對稱矩陣乘法")


def cross_matrix(a):
    """[a]ₓ：滿足 [a]ₓ b = a × b 的反對稱矩陣。"""
    a1, a2, a3 = a
    return np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]])


a = np.array([1.0, 2.0, 3.0])
b = np.array([-2.0, 1.0, 4.0])
c = np.array([0.5, -1.0, 2.0])
Ax = cross_matrix(a)
show_matrix("[a]ₓ", Ax)
check("[a]ₓ b = a × b", Ax @ b, np.cross(a, b))
check("[a]ₓ 反對稱", Ax.T, -Ax)
check("[a]ₓ a = 0（叉積自己 = 0）⇒ a 在零空間", Ax @ a, np.zeros(3))
check("rank([a]ₓ) = 2（零空間就是 a 的方向）", np.linalg.matrix_rank(Ax), 2)
check("反對稱 ⇒ 特徵值為 0, ±i‖a‖",
      np.sort(np.abs(np.linalg.eigvals(Ax))),
      np.sort(np.array([0.0, np.linalg.norm(a), np.linalg.norm(a)])))
check("a × b = −b × a（反交換）", np.cross(a, b), -np.cross(b, a))
check("a × (b × c) ≠ (a × b) × c（不結合）",
      not np.allclose(np.cross(a, np.cross(b, c)), np.cross(np.cross(a, b), c)))
check("Jacobi 恆等式 a×(b×c) + b×(c×a) + c×(a×b) = 0",
      np.cross(a, np.cross(b, c)) + np.cross(b, np.cross(c, a))
      + np.cross(c, np.cross(a, b)), np.zeros(3))

print("\n三重積 = 行列式 = 體積：")
E = np.column_stack([a, b, c])
triple = a @ np.cross(b, c)
print(f"  a·(b×c) = {triple:.6f}")
print(f"  det[a b c] = {det_lu(E):.6f}")
check("a·(b×c) = det[a b c]", triple, det_lu(E))
check("= 平行六面體體積（取絕對值）", abs(triple), volume(E))
check("循環對稱 a·(b×c) = b·(c×a) = c·(a×b)",
      (a @ np.cross(b, c), b @ np.cross(c, a)),
      (c @ np.cross(a, b), c @ np.cross(a, b)))
check("向量三重積 a×(b×c) = (a·c)b − (a·b)c",
      np.cross(a, np.cross(b, c)), (a @ c) * b - (a @ b) * c)
check("Lagrange 恆等式 ‖a×b‖² = ‖a‖²‖b‖² − (a·b)²",
      np.linalg.norm(np.cross(a, b)) ** 2,
      (a @ a) * (b @ b) - (a @ b) ** 2)

# 角速度：剛體旋轉的生成元
print("\n剛體旋轉：Ṙ = [ω]ₓR（反對稱矩陣是旋轉群的生成元）")
from linalg_tutorial.eigen import matrix_exp_series
omega = np.array([0.0, 0.0, 1.0])            # 繞 z 軸
for t in [0.0, np.pi / 4, np.pi / 2]:
    R_t = matrix_exp_series(cross_matrix(omega), t, 60)
    theory = np.array([[np.cos(t), -np.sin(t), 0.0],
                       [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    check(f"  exp(t[ω]ₓ) = 繞 ω 轉 t 的旋轉矩陣（t = {t:.4f}）", R_t, theory, tol=1e-9)
check("exp(反對稱) 必為正交矩陣",
      matrix_exp_series(cross_matrix(a), 0.7, 60).T
      @ matrix_exp_series(cross_matrix(a), 0.7, 60), np.eye(3), tol=1e-10)
print("  v = ω × r 的矩陣寫法就是 v = [ω]ₓ r（第 8 章的 e^{At} 在這裡變成旋轉）")

# %% [markdown]
# ### 倒向量 = 對偶基底 = $E^{-1}$ 的列
#
# Riley 的「倒向量」$\mathbf a^{*},\mathbf b^{*},\mathbf c^{*}$ 滿足
# $\mathbf a^{*}\cdot\mathbf a=1$、$\mathbf a^{*}\cdot\mathbf b=0$ 等，即
#
# $$\mathbf a^{*}=\frac{\mathbf b\times\mathbf c}{\mathbf a\cdot(\mathbf b\times\mathbf c)}$$
#
# 線性代數的說法：**倒基底就是 $E^{-1}$ 的列**（$E$ 的欄是原基底）。
# 晶體學的倒晶格、張量分析的 contravariant 基底，都是同一件事。

# %%
section("16.1 倒向量 = E⁻¹ 的列")

a_star = np.cross(b, c) / triple
b_star = np.cross(c, a) / triple
c_star = np.cross(a, b) / triple
Estar = np.vstack([a_star, b_star, c_star])
show_matrix("倒基底（列）", Estar)
show_matrix("E⁻¹", np.linalg.inv(E))
check("倒基底 = E⁻¹ 的列", Estar, np.linalg.inv(E))
check("a*·a = 1", a_star @ a, 1.0)
check("a*·b = 0", a_star @ b, 0.0)
check("a*·c = 0", a_star @ c, 0.0)
check("對偶關係 E⁻¹E = I（就是全部 9 條關係）", Estar @ E, np.eye(3))
v = np.array([3.0, -1.0, 2.0])
coords = Estar @ v
check("用倒基底讀出座標：v = (a*·v)a + (b*·v)b + (c*·v)c",
      coords[0] * a + coords[1] * b + coords[2] * c, v)
print("  ⇒ 非正交基底下要用倒基底取座標；正交歸一時倒基底 = 原基底（E⁻¹ = Eᵀ）")
Q, _ = np.linalg.qr(E)
check("正交基底的倒基底 = 自己", np.linalg.inv(Q), Q.T)

# %% [markdown]
# ## 16.2　張量的定義：變換律才是本質
#
# 一組有下標的數字是不是張量，要看它在旋轉下的行為：
#
# $$T_{ij}'=L_{ik}L_{jl}T_{kl}\qquad\text{（矩陣寫法：}T'=LTL^{\mathsf T}\text{）}$$
#
# 注意這**不是**相似變換 $S^{-1}TS$ ── 但因為 $L$ 正交，
# $L^{\mathsf T}=L^{-1}$，所以兩者在 Cartesian 張量下一致。

# %%
section("16.2 哪些量是張量？")


def rotation_matrix(axis, theta):
    """繞指定軸轉 θ 的旋轉矩陣（Rodrigues 公式）。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = cross_matrix(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


L = rotation_matrix([1.0, 1.0, 1.0], 0.7)
check("L 正交（旋轉）", L.T @ L, np.eye(3))
check("det L = +1（真旋轉，非反射）", det_lu(L), 1.0)

# Riley 式的測驗：哪些 (v₁,v₂,v₃) 是一階張量？
print("\n一階張量的測驗（在 x₁x₂ 平面上轉 θ）：")
theta = 0.6
L2 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
x_pts = np.array([1.3, -0.7])
x_new = L2 @ x_pts
candidates = {
    "(x₂, −x₁)": (lambda x: np.array([x[1], -x[0]])),
    "(x₂, x₁)": (lambda x: np.array([x[1], x[0]])),
    "(x₁², x₂²)": (lambda x: np.array([x[0] ** 2, x[1] ** 2])),
}
for name, f in candidates.items():
    v_old = f(x_pts)                      # 在舊座標算出分量，再依張量律轉換
    by_law = L2 @ v_old
    by_formula = f(x_new)                 # 直接用新座標代公式
    is_tensor = np.allclose(by_law, by_formula)
    print(f"  {name:<12} 張量律給 {np.round(by_law, 4)}，"
          f"公式直接算給 {np.round(by_formula, 4)} → {'是張量' if is_tensor else '不是張量'}")
check("(x₂, −x₁) 是一階張量", np.allclose(
    L2 @ candidates["(x₂, −x₁)"](x_pts), candidates["(x₂, −x₁)"](x_new)))
check("(x₂, x₁) 不是張量", not np.allclose(
    L2 @ candidates["(x₂, x₁)"](x_pts), candidates["(x₂, x₁)"](x_new)))
check("(x₁², x₂²) 不是張量", not np.allclose(
    L2 @ candidates["(x₁², x₂²)"](x_pts), candidates["(x₁², x₂²)"](x_new)))

# 二階張量的例子：外積、δ、慣性張量
print("\n二階張量的例子：")
T_outer = np.outer(a, b)
T_rot = L @ T_outer @ L.T
check("外積 aᵢbⱼ 是二階張量（T' = LTLᵀ）", T_rot, np.outer(L @ a, L @ b))
check("δ_ij 是等向二階張量（任何旋轉下不變）", L @ np.eye(3) @ L.T, np.eye(3))
check("純量（如 a·b）不變", (L @ a) @ (L @ b), a @ b)
check("trace 不變（一階不變量）", np.trace(T_rot), np.trace(T_outer))
check("det 不變（三階不變量）", det_lu(T_rot), det_lu(T_outer), tol=1e-9)
second_inv = lambda T: 0.5 * (np.trace(T) ** 2 - np.trace(T @ T))
check("第二不變量 ½[(trT)² − tr(T²)] 不變",
      second_inv(T_rot), second_inv(T_outer), tol=1e-9)
print("  三個不變量（trace、第二不變量、det）= 特徵多項式的係數")
check("特徵值不變", np.sort(np.real(np.linalg.eigvals(T_rot))),
      np.sort(np.real(np.linalg.eigvals(T_outer))), tol=1e-9)

# %% [markdown]
# ## 16.3　$\delta_{ij}$ 與 $\epsilon_{ijk}$：張量代數的兩個主角
#
# $$\delta_{ij}=\begin{cases}1&i=j\\0&i\ne j\end{cases},\qquad
# \epsilon_{ijk}=\begin{cases}+1&(i,j,k)\text{ 為偶置換}\\
# -1&\text{奇置換}\\0&\text{有重複}\end{cases}$$
#
# 它們把向量代數全部編碼起來：
#
# $$(\mathbf a\times\mathbf b)_i=\epsilon_{ijk}a_jb_k,\qquad
# \det A=\epsilon_{ijk}A_{1i}A_{2j}A_{3k}$$
#
# 最有用的恆等式（所有向量恆等式的來源）：
#
# $$\boxed{\epsilon_{ijk}\epsilon_{ilm}=\delta_{jl}\delta_{km}-\delta_{jm}\delta_{kl}}$$

# %%
section("16.3 ε 與 δ 的恆等式")

eps = np.zeros((3, 3, 3))
for p in permutations(range(3)):
    inv = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    eps[p] = (-1.0) ** inv
print("ε_ijk 的非零元素：")
for idx in product(range(3), repeat=3):
    if eps[idx] != 0:
        print(f"  ε{''.join(str(i + 1) for i in idx)} = {eps[idx]:+.0f}", end="")
print()

delta = np.eye(3)
check("(a×b)ᵢ = ε_ijk aⱼ bₖ", np.einsum("ijk,j,k->i", eps, a, b), np.cross(a, b))
check("det A = ε_ijk A₁ᵢA₂ⱼA₃ₖ",
      np.einsum("ijk,i,j,k->", eps, E[0, :], E[1, :], E[2, :]), det_lu(E))
check("ε_ijk ε_ilm = δ_jl δ_km − δ_jm δ_kl",
      np.einsum("ijk,ilm->jklm", eps, eps),
      np.einsum("jl,km->jklm", delta, delta) - np.einsum("jm,kl->jklm", delta, delta))
check("ε_ijk ε_ijm = 2δ_km", np.einsum("ijk,ijm->km", eps, eps), 2 * delta)
check("ε_ijk ε_ijk = 6", np.einsum("ijk,ijk->", eps, eps), 6.0)
check("δ_ii = 3", np.trace(delta), 3.0)

# 用 ε-δ 恆等式證明向量三重積
print("\n用 ε–δ 恆等式推導 a×(b×c) = (a·c)b − (a·b)c：")
lhs = np.einsum("ijk,j,klm,l,m->i", eps, a, eps, b, c)
check("  ε_ijk aⱼ ε_klm bₗ cₘ = (a·c)b − (a·b)c", lhs, (a @ c) * b - (a @ b) * c)
check("  也等於 np.cross(a, np.cross(b, c))", lhs, np.cross(a, np.cross(b, c)))

# ε 是等向三階張量（偽張量）
eps_rot = np.einsum("il,jm,kn,lmn->ijk", L, L, L, eps)
check("ε 在真旋轉下不變（等向三階張量）", eps_rot, eps, tol=1e-10)
Parity = -np.eye(3)                        # 反演（improper，det = −1）
eps_parity = np.einsum("il,jm,kn,lmn->ijk", Parity, Parity, Parity, eps)
print(f"\n反演 det(P) = {det_lu(Parity):.0f}（improper）下：ε → {-1 if np.allclose(eps_parity, -eps) else '?'}·ε")
check("ε 在反演下變號 ⇒ 它是「偽張量」", eps_parity, -eps)

# 偽向量：叉積在反演下不變號！
print("\n偽向量（pseudovector）：")
print(f"  真向量：a → Pa = {Parity @ a}")
print(f"  叉積：  a×b → (Pa)×(Pb) = {np.cross(Parity @ a, Parity @ b)}")
print(f"  但若 a×b 是真向量，應該變成 P(a×b) = {Parity @ np.cross(a, b)}")
check("(Pa)×(Pb) = +a×b（不變號）⇒ 叉積是偽向量",
      np.cross(Parity @ a, Parity @ b), np.cross(a, b))
check("真向量在反演下變號", Parity @ a, -a)
print("  物理意義：角動量、磁場、角速度都是偽向量（鏡像下行為與真向量不同）")

# 對偶張量：反對稱 3x3 ↔ 向量
print("\n對偶張量：反對稱矩陣 ↔ 向量（Riley 26.11）")
A_skew = cross_matrix(a)
dual_vec = -0.5 * np.einsum("ijk,jk->i", eps, A_skew)
check("由反對稱矩陣還原向量：aᵢ = −½ε_ijk A_jk", dual_vec, a)
check("反過來：A_jk = −ε_jki aᵢ", -np.einsum("jki,i->jk", eps, a), A_skew)
print(f"  3x3 反對稱矩陣有 3 個自由度，正好對應一個三維向量")
check("反對稱矩陣空間的維度 = 3", 3 * (3 - 1) // 2, 3)

# %% [markdown]
# ## 16.4　等向張量：旋轉平均的威力
#
# **等向**（isotropic）= 在所有旋轉下都不變。結論非常強：
#
# | 階 | 所有等向張量 |
# |---|---|
# | 0 | 任意純量 |
# | 1 | 只有 $\mathbf 0$ |
# | 2 | $\lambda\delta_{ij}$ |
# | 3 | $\lambda\epsilon_{ijk}$ |
# | 4 | $\alpha\delta_{ij}\delta_{kl}+\beta\delta_{ik}\delta_{jl}+\gamma\delta_{il}\delta_{jk}$ |
#
# 四階的結果就是**等向彈性體只有兩個獨立彈性常數**（Lamé 常數）的原因！
#
# 下面用「對所有旋轉取平均」的數值實驗驗證：任何張量的旋轉平均
# 一定落在等向張量的空間裡。

# %%
section("16.4 旋轉平均 ⇒ 等向張量")

rng = np.random.default_rng(0)


def random_rotation(rng):
    """用 QR 產生均勻分布的隨機旋轉矩陣（det = +1）。"""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if det_lu(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# 一階：旋轉平均必為 0
v_any = np.array([1.0, 2.0, -1.0])
avg1 = np.mean([random_rotation(rng) @ v_any for _ in range(200000)], axis=0)
print(f"一階張量的旋轉平均 = {np.round(avg1, 4)}（理論 0）")
check("等向一階張量只有 0", np.linalg.norm(avg1) < 0.01)

# 二階：旋轉平均 = (trace/3)·δ
T_any = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0], [2.0, 1.0, 4.0]])
avg2 = np.mean([(lambda R: R @ T_any @ R.T)(random_rotation(rng))
                for _ in range(200000)], axis=0)
show_matrix("二階張量的旋轉平均", avg2)
print(f"理論 = (tr T/3)·δ = {np.trace(T_any) / 3:.4f}·δ")
check("等向二階張量 = λδ_ij", avg2, np.trace(T_any) / 3 * np.eye(3), tol=0.02)
check("  λ = tr(T)/3", np.trace(avg2) / 3, np.trace(T_any) / 3, tol=0.01)

# 三階：旋轉平均 ∝ ε
T3 = rng.standard_normal((3, 3, 3))
avg3 = np.mean([(lambda R: np.einsum("il,jm,kn,lmn->ijk", R, R, R, T3))(
    random_rotation(rng)) for _ in range(100000)], axis=0)
coef = np.einsum("ijk,ijk->", avg3, eps) / 6
print(f"\n三階張量的旋轉平均 ∝ ε，係數 = {coef:.6f}")
check("等向三階張量 = λ ε_ijk", avg3, coef * eps, tol=0.02)
check("  λ = (T·ε)/6", coef, np.einsum("ijk,ijk->", T3, eps) / 6, tol=1e-2)

# 四階：三個獨立常數 ⇒ 等向彈性只有兩個（加上對稱性）
print("\n四階等向張量的空間維度：")
basis4 = [np.einsum("ij,kl->ijkl", delta, delta),
          np.einsum("ik,jl->ijkl", delta, delta),
          np.einsum("il,jk->ijkl", delta, delta)]
Bmat = np.array([B.ravel() for B in basis4])
print(f"  δ_ijδ_kl、δ_ikδ_jl、δ_ilδ_jk 三個基底，秩 = {np.linalg.matrix_rank(Bmat)}")
check("四階等向張量空間維度 = 3", np.linalg.matrix_rank(Bmat), 3)
T4 = rng.standard_normal((3, 3, 3, 3))
avg4 = np.mean([(lambda R: np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, T4))(
    random_rotation(rng)) for _ in range(20000)], axis=0)
coef4, *_ = np.linalg.lstsq(Bmat.T, avg4.ravel(), rcond=None)
resid = np.linalg.norm(Bmat.T @ coef4 - avg4.ravel()) / np.linalg.norm(avg4)
print(f"  旋轉平均落在這三個基底張出的空間裡（相對殘差 = {resid:.3%}）")
check("四階旋轉平均 = 三個 δδ 項的組合", resid < 0.05)
print("  ⇒ 等向彈性體的彈性張量只有 2 個獨立常數（再用對稱性去掉一個）")
print("     這就是 Lamé 常數 λ 與 μ（或 Young 模數與 Poisson 比）的來源")

# %% [markdown]
# ## 16.5　物理應用：主軸與不變量
#
# 二階張量的物理意義一律是「**向量 → 向量**的線性關係」：
#
# | 關係 | 張量 | 說明 |
# |---|---|---|
# | $\mathbf J=\sigma\mathbf E$ | 導電率 $\sigma_{ij}$ | 電流不一定平行電場！ |
# | $\mathbf L=I\boldsymbol\omega$ | 慣性 $I_{ij}$ | 角動量不一定平行角速度 |
# | $\mathbf t=\tau\hat{\mathbf n}$ | 應力 $\tau_{ij}$ | 面上的力與法向量 |
# | $\mathbf p=\alpha\mathbf E$ | 極化率 $\alpha_{ij}$ | 晶體的雙折射 |
#
# **對稱張量 $\Rightarrow$ 必有三個互相垂直的主軸**（譜定理，第 7 章），
# 在主軸座標下張量變成對角 ── 物理上就是「本徵方向」。

# %%
section("16.5 導電率張量：電流為何不平行電場")

# 各向異性晶體的導電率
sigma = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
check("導電率張量對稱（Onsager 倒易關係）", sigma, sigma.T)
check("正定（功率 EᵀσE > 0，被動介質）", is_positive_definite(sigma))

E_field = np.array([1.0, 0.0, 0.0])
J = sigma @ E_field
angle = np.degrees(np.arccos(J @ E_field / (np.linalg.norm(J) * np.linalg.norm(E_field))))
print(f"E = {E_field} ⇒ J = {J}")
print(f"  J 與 E 的夾角 = {angle:.2f}° ≠ 0 ⇒ 電流不平行電場！")
check("J 與 E 不平行", angle > 1e-6)

sigma_principal, axes = np.linalg.eigh(sigma)
print(f"\n主導電率 = {sigma_principal}")
print(f"主軸（正交）=\n{np.round(axes, 4)}")
check("主軸正交", axes.T @ axes, np.eye(3))
check("主軸座標下張量對角", axes.T @ sigma @ axes, np.diag(sigma_principal), tol=1e-10)
for k in range(3):
    n_ax = axes[:, k]
    J_ax = sigma @ n_ax
    check(f"  沿主軸 {k + 1} 施加 E ⇒ J ∥ E",
          np.linalg.norm(np.cross(J_ax, n_ax)), 0.0)
print("  只有沿主軸施加電場時，電流才與電場平行")

power = lambda Ev: Ev @ sigma @ Ev
print(f"\n功率耗散 P = EᵀσE（二次式）：")
print(f"  最大 = λ_max = {sigma_principal[-1]:.4f}（沿主軸 3）")
print(f"  最小 = λ_min = {sigma_principal[0]:.4f}（沿主軸 1）")
rng = np.random.default_rng(1)
dirs = rng.standard_normal((20000, 3))
dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
vals = np.einsum("ij,jk,ik->i", dirs, sigma, dirs)
check("單位電場的功率落在 [λ_min, λ_max]",
      vals.min() >= sigma_principal[0] - 1e-9 and vals.max() <= sigma_principal[-1] + 1e-9)

# 應力張量：主應力與最大剪應力
section("16.5 應力張量：主應力與最大剪應力")

tau = np.array([[50.0, 30.0, 0.0], [30.0, -20.0, 0.0], [0.0, 0.0, 10.0]])
show_matrix("應力張量 τ (MPa)", tau)
principal, dirs_p = np.linalg.eigh(tau)
print(f"主應力 = {principal} MPa")
check("主應力 = 特徵值", np.sort(principal), np.sort(np.linalg.eigvalsh(tau)))
print(f"最大剪應力 = (σ_max − σ_min)/2 = "
      f"{(principal[-1] - principal[0]) / 2:.4f} MPa（Mohr 圓的半徑）")
tr1, tr2, tr3 = np.trace(tau), second_inv(tau), det_lu(tau)
print(f"\n三個不變量：I₁ = {tr1:.4f}，I₂ = {tr2:.4f}，I₃ = {tr3:.4f}")
L_rand = random_rotation(np.random.default_rng(3))
tau_rot = L_rand @ tau @ L_rand.T
check("I₁ = tr τ 旋轉不變", np.trace(tau_rot), tr1, tol=1e-9)
check("I₂ 旋轉不變", second_inv(tau_rot), tr2, tol=1e-8)
check("I₃ = det τ 旋轉不變", det_lu(tau_rot), tr3, tol=1e-8)
print("  ⇒ 破壞準則（von Mises 等）只能用不變量寫，否則會依賴座標選擇")
dev = tau - np.trace(tau) / 3 * np.eye(3)
von_mises = np.sqrt(1.5 * np.sum(dev * dev))
check("von Mises 應力也是旋轉不變量",
      np.sqrt(1.5 * np.sum((tau_rot - np.trace(tau_rot) / 3 * np.eye(3)) ** 2)),
      von_mises, tol=1e-8)
print(f"  von Mises 應力 = {von_mises:.4f} MPa")
check("偏應力的 trace = 0（純剪切部分）", np.trace(dev), 0.0)

fig, ax = new_axes("Stress: the ellipsoid of t = tau n for unit n",
                   figsize=(5.6, 4.6), equal=False, d3=True)
u_ang = np.linspace(0, 2 * np.pi, 60)
v_ang = np.linspace(0, np.pi, 30)
X = np.outer(np.cos(u_ang), np.sin(v_ang))
Y = np.outer(np.sin(u_ang), np.sin(v_ang))
Z = np.outer(np.ones_like(u_ang), np.cos(v_ang))
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
mapped = tau @ pts
ax.plot_surface(mapped[0].reshape(X.shape), mapped[1].reshape(X.shape),
                mapped[2].reshape(X.shape), alpha=0.3, color="C0")
for k in range(3):
    d = dirs_p[:, k] * principal[k]
    ax.plot([0, d[0]], [0, d[1]], [0, d[2]], "C3", lw=2.2)
ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_zlabel("t3")
finish(fig, "ch16_stress_ellipsoid")

# %% [markdown]
# ## 16.6　非 Cartesian 座標與度規張量
#
# 座標不正交（或不是直線）時，必須區分兩種分量：
#
# $$\text{covariant }v_i=\mathbf v\cdot\mathbf e_i,\qquad
# \text{contravariant }v^i=\mathbf v\cdot\mathbf e^{\,i}$$
#
# 兩者由**度規張量**互相轉換（升／降指標）：
#
# $$g_{ij}=\mathbf e_i\cdot\mathbf e_j,\qquad
# v_i=g_{ij}v^{j},\qquad v^{i}=g^{ij}v_j,\qquad g^{ij}=(g^{-1})_{ij}$$
#
# 弧長與體積元素都由 $g$ 決定：
#
# $$ds^2=g_{ij}\,dx^i dx^j,\qquad dV=\sqrt{\det g}\;dx^1dx^2dx^3$$

# %%
section("16.6 度規張量：升降指標與弧長")

# 斜交基底
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.array([1.0, 1.0, 0.0])
e3 = np.array([0.0, 1.0, 2.0])
Ebasis = np.column_stack([e1, e2, e3])
g = Ebasis.T @ Ebasis
show_matrix("度規 g_ij = eᵢ·eⱼ", g)
g_inv = np.linalg.inv(g)
show_matrix("g^ij = (g⁻¹)_ij", g_inv)
check("g 對稱正定", is_positive_definite(g))
check("g^ij g_jk = δ^i_k", g_inv @ g, np.eye(3))

v_phys = np.array([2.0, -1.0, 3.0])
v_contra = np.linalg.solve(Ebasis, v_phys)         # v = v^i eᵢ
v_cov = Ebasis.T @ v_phys                          # vᵢ = v·eᵢ
print(f"\n物理向量 v = {v_phys}")
print(f"  contravariant 分量 v^i = {np.round(v_contra, 4)}（係數）")
print(f"  covariant 分量 vᵢ    = {np.round(v_cov, 4)}（投影）")
check("vᵢ = g_ij v^j（降指標）", v_cov, g @ v_contra)
check("v^i = g^ij vⱼ（升指標）", v_contra, g_inv @ v_cov)
check("長度 ‖v‖² = v^i vᵢ = g_ij v^i v^j",
      v_contra @ v_cov, v_phys @ v_phys)
check("  也等於 g_ij v^i v^j", v_contra @ g @ v_contra, v_phys @ v_phys)
print("  ⇒ 只有正交歸一基底（g = I）時，上下指標才沒有差別")
check("倒基底 = contravariant 基底 = E⁻¹ 的列",
      np.linalg.inv(Ebasis), (g_inv @ Ebasis.T))

# 體積元素
check("dV = √det(g)（平行六面體體積）", np.sqrt(det_lu(g)), volume(Ebasis), tol=1e-10)
print(f"  √det g = {np.sqrt(det_lu(g)):.6f} = |det E| = {volume(Ebasis):.6f}")

# 曲線座標：極座標與球座標的度規
section("16.6 曲線座標的度規")

print("極座標 (r, θ)：x = r cosθ, y = r sinθ")
for r0, th0 in [(1.0, 0.3), (2.5, 1.2)]:
    J = np.array([[np.cos(th0), -r0 * np.sin(th0)],
                  [np.sin(th0), r0 * np.cos(th0)]])       # ∂(x,y)/∂(r,θ)
    g_polar = J.T @ J
    print(f"  r = {r0}, θ = {th0}：g = {np.round(g_polar, 6).tolist()}"
          f"，√det g = {np.sqrt(det_lu(g_polar)):.6f}")
    check(f"  g = diag(1, r²)", g_polar, np.diag([1.0, r0 ** 2]), tol=1e-10)
    check(f"  √det g = r（面積元素 dA = r dr dθ）", np.sqrt(det_lu(g_polar)), r0)

print("\n球座標 (r, θ, φ)：")
r0, th0, ph0 = 2.0, 0.8, 1.1
J3 = np.array([
    [np.sin(th0) * np.cos(ph0), r0 * np.cos(th0) * np.cos(ph0),
     -r0 * np.sin(th0) * np.sin(ph0)],
    [np.sin(th0) * np.sin(ph0), r0 * np.cos(th0) * np.sin(ph0),
     r0 * np.sin(th0) * np.cos(ph0)],
    [np.cos(th0), -r0 * np.sin(th0), 0.0]])
g_sph = J3.T @ J3
show_matrix("球座標的度規 g", g_sph)
check("g = diag(1, r², r²sin²θ)", g_sph,
      np.diag([1.0, r0 ** 2, (r0 * np.sin(th0)) ** 2]), tol=1e-10)
check("√det g = r²sinθ（體積元素 dV = r²sinθ dr dθ dφ）",
      np.sqrt(det_lu(g_sph)), r0 ** 2 * np.sin(th0))
print(f"  √det g = {np.sqrt(det_lu(g_sph)):.6f} = r²sinθ = "
      f"{r0 ** 2 * np.sin(th0):.6f}")
print("  ⇒ Jacobian（第 5 章）與度規的行列式是同一回事")

# Christoffel 符號：基底向量隨位置改變
print("\nChristoffel 符號 Γ：曲線座標下基底向量會變化")
print("  極座標：∂e_r/∂θ = e_θ/r，∂e_θ/∂θ = −r e_r")
h = 1e-6


def polar_basis(r, th):
    """極座標的（非歸一）基底向量 e_r = ∂x/∂r, e_θ = ∂x/∂θ。"""
    return (np.array([np.cos(th), np.sin(th)]),
            np.array([-r * np.sin(th), r * np.cos(th)]))


r0, th0 = 1.7, 0.6
er, eth = polar_basis(r0, th0)
der_dth = (polar_basis(r0, th0 + h)[0] - polar_basis(r0, th0 - h)[0]) / (2 * h)
deth_dth = (polar_basis(r0, th0 + h)[1] - polar_basis(r0, th0 - h)[1]) / (2 * h)
check("∂e_r/∂θ = (1/r)·e_θ", der_dth, eth / r0, tol=1e-6)
check("∂e_θ/∂θ = −r·e_r", deth_dth, -r0 * er, tol=1e-6)
print(f"  Γ^θ_rθ = 1/r = {1 / r0:.6f}，Γ^r_θθ = −r = {-r0:.6f}")
print("  ⇒ 這些 Γ 就是為什麼曲線座標下的「導數」要用共變導數（covariant derivative）")

# %% [markdown]
# ## 動手練習
#
# 1. 用 $\epsilon$–$\delta$ 恆等式證明
#    $(\mathbf a\times\mathbf b)\cdot(\mathbf c\times\mathbf d)
#    =(\mathbf a\cdot\mathbf c)(\mathbf b\cdot\mathbf d)-(\mathbf a\cdot\mathbf d)(\mathbf b\cdot\mathbf c)$。
# 2. 判斷 $T_{ij}=x_ix_j$ 與 $T_{ij}=x_i+x_j$ 哪個是二階張量。
# 3. 求均勻長方體（邊長 $a,b,c$、質量 $M$）的慣性張量，確認主軸就是幾何對稱軸。
# 4. 用旋轉平均驗證：任意二階張量的旋轉平均 $=\frac{\operatorname{tr}T}{3}\delta$。
# 5. 算出柱座標 $(\rho,\phi,z)$ 的度規與體積元素。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
d_vec = np.array([2.0, -3.0, 1.0])
lhs1 = np.cross(a, b) @ np.cross(c, d_vec)
rhs1 = (a @ c) * (b @ d_vec) - (a @ d_vec) * (b @ c)
check("練習 1：(a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c)", lhs1, rhs1)
check("  用 ε–δ 直接算也相同",
      np.einsum("ijk,j,k,ilm,l,m->", eps, a, b, eps, c, d_vec), rhs1)

# 練習 2
print("\n練習 2：")
x0 = np.array([1.3, -0.7, 0.5])
x1 = L @ x0
T_a = np.outer(x0, x0)
T_a_new = np.outer(x1, x1)
check("  T_ij = xᵢxⱼ 是二階張量", L @ T_a @ L.T, T_a_new)
# 注意：要用一個「不固定 (1,1,1) 方向」的旋轉才看得出來
L_z = rotation_matrix([0.0, 0.0, 1.0], 0.7)
x1z = L_z @ x0
T_b = x0[:, None] + x0[None, :]
T_b_new = x1z[:, None] + x1z[None, :]
check("  T_ij = xᵢ + xⱼ 不是張量", not np.allclose(L_z @ T_b @ L_z.T, T_b_new))
print("  原因：LTLᵀ = (Lx)(L1)ᵀ + (L1)(Lx)ᵀ，但公式要的是 (Lx)1ᵀ + 1(Lx)ᵀ；")
print("        除非 L1 = 1（旋轉軸恰好是 (1,1,1)）才會巧合相等 —— 張量律必須對所有旋轉成立")
check("  （若旋轉軸正好是 (1,1,1)，L1 = 1 會造成巧合）", L @ np.ones(3), np.ones(3),
      tol=1e-10)

# 練習 3：長方體的慣性張量
print("\n練習 3：均勻長方體的慣性張量")
M_box, a_box, b_box, c_box = 12.0, 2.0, 3.0, 4.0
I_box = M_box / 12 * np.diag([b_box ** 2 + c_box ** 2,
                              a_box ** 2 + c_box ** 2,
                              a_box ** 2 + b_box ** 2])
show_matrix("I（已在對稱軸座標）", I_box)
check("  已是對角矩陣 ⇒ 幾何對稱軸就是主軸", I_box, np.diag(np.diag(I_box)))
# 數值積分驗證
rng = np.random.default_rng(0)
pts = rng.uniform(-0.5, 0.5, size=(400000, 3)) * np.array([a_box, b_box, c_box])
dm = M_box / len(pts)
I_num = sum(dm * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) for r in pts[:4000])
I_num = np.zeros((3, 3))
for idx in range(3):
    for jdx in range(3):
        if idx == jdx:
            I_num[idx, jdx] = dm * np.sum(np.sum(pts ** 2, axis=1) - pts[:, idx] ** 2)
        else:
            I_num[idx, jdx] = -dm * np.sum(pts[:, idx] * pts[:, jdx])
check("  蒙地卡羅積分與公式相符", I_num, I_box, tol=0.05)
L_box = random_rotation(np.random.default_rng(9))
I_rot = L_box @ I_box @ L_box.T
check("  旋轉後不再對角，但 trace 不變", np.trace(I_rot), np.trace(I_box), tol=1e-9)
check("  特徵值（主慣性矩）不變",
      np.sort(np.linalg.eigvalsh(I_rot)), np.sort(np.diag(I_box)), tol=1e-9)

# 練習 4（已在 16.4 做過，這裡用另一個張量再確認）
T_ex = np.array([[2.0, -1.0, 0.5], [3.0, 1.0, 1.0], [0.0, 2.0, -3.0]])
rng = np.random.default_rng(7)
avg_ex = np.mean([(lambda R: R @ T_ex @ R.T)(random_rotation(rng))
                  for _ in range(150000)], axis=0)
print(f"\n練習 4：旋轉平均 ≈ (trT/3)δ = {np.trace(T_ex) / 3:.4f}·δ")
check("  驗證", avg_ex, np.trace(T_ex) / 3 * np.eye(3), tol=0.02)

# 練習 5：柱座標
print("\n練習 5：柱座標 (ρ, φ, z)")
rho0, phi0 = 1.5, 0.9
J_cyl = np.array([[np.cos(phi0), -rho0 * np.sin(phi0), 0.0],
                  [np.sin(phi0), rho0 * np.cos(phi0), 0.0],
                  [0.0, 0.0, 1.0]])
g_cyl = J_cyl.T @ J_cyl
check("  g = diag(1, ρ², 1)", g_cyl, np.diag([1.0, rho0 ** 2, 1.0]), tol=1e-10)
check("  √det g = ρ（dV = ρ dρ dφ dz）", np.sqrt(det_lu(g_cyl)), rho0)
print(f"  ds² = dρ² + ρ²dφ² + dz²，dV = {np.sqrt(det_lu(g_cyl)):.4f} dρ dφ dz")

# %% [markdown]
# ## 本章重點回顧
#
# * 叉積 $=$ 反對稱矩陣乘法 $[\mathbf a]_{\times}\mathbf b$；
#   $3\times3$ 反對稱矩陣 $\leftrightarrow$ 三維向量（對偶張量）；
#   $\exp(t[\boldsymbol\omega]_{\times})$ 就是旋轉矩陣。
# * 三重積 $=$ 行列式 $=$ 體積；倒向量 $=$ 對偶基底 $=E^{-1}$ 的列。
# * **張量由變換律定義**：$T'=LTL^{\mathsf T}$。不是每個有下標的東西都是張量。
#   trace、第二不變量、det、特徵值在旋轉下都不變 ── 它們才是物理量。
# * $\delta_{ij}$ 與 $\epsilon_{ijk}$ 把向量代數全部編碼；
#   $\epsilon_{ijk}\epsilon_{ilm}=\delta_{jl}\delta_{km}-\delta_{jm}\delta_{kl}$
#   一條恆等式就能推出所有向量恆等式。
# * $\epsilon$ 在反演下變號 $\Rightarrow$ 叉積是**偽向量**（角動量、磁場都是）。
# * **等向張量**：一階只有 $\mathbf 0$、二階只有 $\lambda\delta$、三階只有 $\lambda\epsilon$、
#   四階只有三個常數 ── 這就是等向彈性體只有兩個彈性常數的原因。
#   用「旋轉平均」可以數值驗證。
# * 二階對稱張量的物理意義是「向量 → 向量」的線性關係，
#   必有三個垂直的**主軸**；導電率、慣性、應力、極化率全是同一套數學。
# * 非 Cartesian 座標要用**度規張量** $g_{ij}=\mathbf e_i\cdot\mathbf e_j$ 升降指標；
#   $\sqrt{\det g}$ 就是 Jacobian（第 5 章），$\Gamma$ 符號描述基底向量的變化。
#
# 下一章：無窮維線性代數 ── Fourier 級數、正交函數與 Sturm–Liouville 理論。
