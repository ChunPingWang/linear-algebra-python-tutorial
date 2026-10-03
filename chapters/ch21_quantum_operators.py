# %% [markdown]
# # 第 21 章　量子算子：Hermitian 矩陣、交換子與不確定性
#
# > 對應 Riley 第 19 章（19.1 算子形式、交換子、19.2 物理算子的例子：
# > 位置與動量、角動量、梯子算子），並對照 Strang 6.4（複數與 Hermitian）、
# > 本教材第 7 章（譜定理）、第 17 章（自伴算子）。
#
# 量子力學**就是**無窮維的線性代數：
#
# | 量子力學 | 線性代數 |
# |---|---|
# | 狀態 $|\psi\rangle$ | 單位向量（$\langle\psi|\psi\rangle=1$）|
# | 可觀測量 $A$ | **Hermitian 矩陣** |
# | 可能的測量值 | **特徵值**（必為實數）|
# | 測到 $a_n$ 的機率 | $|c_n|^2=|\langle a_n|\psi\rangle|^2$ |
# | 期望值 | $\langle\psi|A|\psi\rangle$（Rayleigh 商！）|
# | 完備性 | $\sum_n|a_n\rangle\langle a_n|=I$（$=QQ^{\mathsf H}$）|
# | 時間演化 | **unitary** $U=e^{-iHt/\hbar}$ |
# | 同時可測 | $[A,B]=0$（可交換 ⇒ 共用特徵向量）|
# | 不確定性原理 | $\Delta A\,\Delta B\ge\frac12|\langle[A,B]\rangle|$ |
#
# ```bash
# python chapters/ch21_quantum_operators.py
# python tools/build_notebooks.py ch21
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

from linalg_tutorial.eigen import matrix_exp_series
from linalg_tutorial.elimination import second_difference_matrix
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# Dirac 記號的實作
bra_ket = lambda psi, phi: np.vdot(psi, phi)                 # ⟨ψ|φ⟩
expect = lambda psi, A: np.real(np.vdot(psi, A @ psi))       # ⟨ψ|A|ψ⟩
dagger = lambda A: A.conj().T                                # A†
is_hermitian = lambda A: np.allclose(dagger(A), A)
comm = lambda A, B: A @ B - B @ A                            # [A, B]

# %% [markdown]
# ## 21.1　可觀測量 = Hermitian 矩陣
#
# 為什麼物理的可觀測量一定是 Hermitian？因為**測量結果必須是實數**，
# 而 Hermitian 矩陣的特徵值必為實數（第 7 章的譜定理）。
#
# 三行證明（Dirac 記號）：
#
# $$A|a\rangle=a|a\rangle
# \Rightarrow\langle a|A|a\rangle=a\langle a|a\rangle;\qquad
# \langle a|A^{\dagger}=a^{*}\langle a|
# \Rightarrow\langle a|A|a\rangle=a^{*}\langle a|a\rangle$$
#
# 兩式相減：$(a-a^{*})\langle a|a\rangle=0$，而 $\langle a|a\rangle\ne0$，故 $a=a^{*}$。

# %%
section("21.1 Hermitian 算子的特徵值必為實數")

H = np.array([[2.0, 1 - 1j, 0.0],
              [1 + 1j, 3.0, 2j],
              [0.0, -2j, -1.0]])
show_matrix("Hamiltonian H", H)
check("H† = H（Hermitian）", dagger(H), H)
E, states = np.linalg.eigh(H)
print(f"能階 E = {E}（全為實數）")
check("特徵值全為實數", np.all(np.isreal(E)))
check("特徵態正交歸一：⟨aₘ|aₙ⟩ = δₘₙ", dagger(states) @ states,
      np.eye(3, dtype=complex))
for n in range(3):
    check(f"  H|a{n}⟩ = E{n}|a{n}⟩", H @ states[:, n], E[n] * states[:, n])
    check(f"  ⟨a{n}|H|a{n}⟩ = E{n}（期望值 = 特徵值）",
          expect(states[:, n], H), E[n])

# 非 Hermitian 的反例
A_nonh = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
print(f"\n非 Hermitian 的反例 [[0,1],[0,0]]：λ = {np.linalg.eigvals(A_nonh)}")
A_cplx = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
print(f"反 Hermitian 的例子 [[0,1],[−1,0]]：λ = {np.linalg.eigvals(A_cplx)}"
      f"（純虛數 ⇒ 不能當可觀測量）")
check("反 Hermitian 的 λ 是純虛數",
      np.allclose(np.real(np.linalg.eigvals(A_cplx)), 0))

# %% [markdown]
# ### 測量：機率與期望值
#
# 把狀態用特徵態展開 $|\psi\rangle=\sum_nc_n|a_n\rangle$，則
#
# $$P(a_n)=|c_n|^2,\qquad
# \sum_n|c_n|^2=1,\qquad
# \langle A\rangle=\langle\psi|A|\psi\rangle=\sum_n|c_n|^2a_n$$
#
# 這就是第 4 章「用正交基底展開」加上第 7 章「Rayleigh 商」。

# %%
section("21.1 測量的機率與期望值")

rng = np.random.default_rng(0)
psi = rng.standard_normal(3) + 1j * rng.standard_normal(3)
psi = psi / np.linalg.norm(psi)
print(f"狀態 |ψ⟩ = {np.round(psi, 4)}")
check("歸一化 ⟨ψ|ψ⟩ = 1", bra_ket(psi, psi).real, 1.0)

c = dagger(states) @ psi                      # cₙ = ⟨aₙ|ψ⟩
probs = np.abs(c) ** 2
print(f"\n展開係數 cₙ = ⟨aₙ|ψ⟩ = {np.round(c, 4)}")
print(f"測量機率 |cₙ|² = {np.round(probs, 6)}")
check("機率總和 = 1（Parseval）", float(probs.sum()), 1.0)
check("|ψ⟩ = Σcₙ|aₙ⟩（完整還原）", states @ c, psi)
print(f"\n期望值 ⟨A⟩：")
print(f"  ⟨ψ|H|ψ⟩      = {expect(psi, H):.10f}")
print(f"  Σ|cₙ|²Eₙ     = {np.sum(probs * E):.10f}")
check("⟨ψ|H|ψ⟩ = Σ|cₙ|²Eₙ", expect(psi, H), float(np.sum(probs * E)))
check("期望值落在 [E_min, E_max]（Rayleigh 商的範圍）",
      E[0] - 1e-12 <= expect(psi, H) <= E[-1] + 1e-12)

# 完備性 = 投影算子之和 = QQᴴ = I
proj = [np.outer(states[:, n], states[:, n].conj()) for n in range(3)]
check("Σ|aₙ⟩⟨aₙ| = I（完備性）", sum(proj), np.eye(3, dtype=complex))
for n in range(3):
    check(f"  P{n}² = P{n}（投影算子）", proj[n] @ proj[n], proj[n])
    check(f"  P{n}† = P{n}", dagger(proj[n]), proj[n])
check("PₘPₙ = 0（m ≠ n，正交投影）", proj[0] @ proj[1],
      np.zeros((3, 3), dtype=complex))
check("H = ΣEₙ|aₙ⟩⟨aₙ|（譜分解）", sum(E[n] * proj[n] for n in range(3)), H)
print("  ⇒ 「完備性關係」就是第 7 章的譜分解 S = QΛQᴴ")

# 測量後的塌縮
n_collapse = 1
psi_after = proj[n_collapse] @ psi / np.linalg.norm(proj[n_collapse] @ psi)
check("測到 E₁ 後塌縮成特徵態（相差一個相位）",
      np.abs(np.abs(bra_ket(psi_after, states[:, n_collapse])) - 1.0) < 1e-10)
check("塌縮後再測一定得到同一個值",
      expect(psi_after, H), E[n_collapse])

# %% [markdown]
# ## 21.2　交換子：能不能「同時測量」
#
# $$[A,B]=AB-BA$$
#
# **核心定理**（第 6 章就提過）：
#
# $$[A,B]=0\iff A,B\text{ 可以同時對角化}\iff\text{共用一組完整的特徵基底}$$
#
# 物理意義：$[A,B]=0$ 才能同時精確測量兩個量。
#
# 交換子的代數性質：
# $[A,B]=-[B,A]$、$[A,BC]=[A,B]C+B[A,C]$、
# Jacobi 恆等式 $[A,[B,C]]+[B,[C,A]]+[C,[A,B]]=0$（與第 16 章的叉積同構！）

# %%
section("21.2 交換子與同時對角化")

# 可交換的例子
A_c = np.diag([1.0, 2.0, 3.0]).astype(complex)
B_c = np.diag([4.0, -1.0, 0.5]).astype(complex)
U_rand, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
A_com = U_rand @ A_c @ dagger(U_rand)
B_com = U_rand @ B_c @ dagger(U_rand)
check("A, B 都是 Hermitian", is_hermitian(A_com) and is_hermitian(B_com))
check("[A,B] = 0（可交換）", comm(A_com, B_com), np.zeros((3, 3), dtype=complex))
_, V_A = np.linalg.eigh(A_com)
check("A 的特徵向量也是 B 的特徵向量（同時對角化）",
      dagger(V_A) @ B_com @ V_A,
      np.diag(np.diag(dagger(V_A) @ B_com @ V_A)), tol=1e-9)

# 不可交換的例子
A_nc = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)      # σ_z
B_nc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)       # σ_x
show_matrix("[σ_z, σ_x] 的實部", np.real(comm(A_nc, B_nc)))
check("[σ_z, σ_x] ≠ 0", not np.allclose(comm(A_nc, B_nc), 0))
check("[σ_z, σ_x] = 2iσ_y", comm(A_nc, B_nc),
      2j * np.array([[0.0, -1j], [1j, 0.0]]))
_, V_z = np.linalg.eigh(A_nc)
print(f"σ_z 的特徵向量 = {np.round(V_z, 4).tolist()}")
print(f"σ_x 作用在 σ_z 的特徵向量上：{np.round(B_nc @ V_z[:, 0], 4)}"
      f"（不是倍數 ⇒ 不是共同特徵向量）")
check("σ_z 的特徵向量不是 σ_x 的特徵向量",
      not np.allclose(np.cross(np.append(np.real(B_nc @ V_z[:, 0]), 0),
                               np.append(np.real(V_z[:, 0]), 0)), 0))

# 交換子的代數性質
A_t = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
B_t = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
C_t = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
check("[A,B] = −[B,A]", comm(A_t, B_t), -comm(B_t, A_t))
check("[A,BC] = [A,B]C + B[A,C]（Leibniz 律）",
      comm(A_t, B_t @ C_t), comm(A_t, B_t) @ C_t + B_t @ comm(A_t, C_t))
check("Jacobi 恆等式（與叉積同構！）",
      comm(A_t, comm(B_t, C_t)) + comm(B_t, comm(C_t, A_t))
      + comm(C_t, comm(A_t, B_t)), np.zeros((4, 4), dtype=complex))
check("A, B Hermitian ⇒ [A,B] 是反 Hermitian",
      dagger(comm(A_nc, B_nc)), -comm(A_nc, B_nc))
print("  ⇒ 所以 i[A,B] 才是 Hermitian（可觀測量）── 不確定性原理用的就是它")

# %% [markdown]
# ### 位置與動量：$[x,p]=i\hbar$
#
# 離散化後 $\hat x=\operatorname{diag}(x_j)$、
# $\hat p=-i\hbar D$（$D$ 是中央差分矩陣）。
#
# 連續情形 $[\hat x,\hat p]=i\hbar I$ 在有限維**不可能精確成立**
# （因為 $\operatorname{tr}[A,B]=0$ 但 $\operatorname{tr}(i\hbar I)=i\hbar n\ne0$）──
# 這是「位置與動量都有連續譜」的線性代數證據：**它們需要無窮維空間**。

# %%
section("21.2 [x, p] = iℏ 在有限維不可能成立")

hbar = 1.0
n = 200
L = 20.0
h = L / n
x_grid = (np.arange(n) - n / 2) * h
X_op = np.diag(x_grid).astype(complex)

# 中央差分的動量算子（週期邊界 ⇒ 反對稱 ⇒ p Hermitian）
D = np.zeros((n, n), dtype=complex)
for i in range(n):
    D[i, (i + 1) % n] = 1 / (2 * h)
    D[i, (i - 1) % n] = -1 / (2 * h)
P_op = -1j * hbar * D
check("x̂ 是 Hermitian", is_hermitian(X_op))
check("p̂ 是 Hermitian（D 反對稱 ⇒ −iℏD Hermitian）", is_hermitian(P_op))

C_xp = comm(X_op, P_op)
print(f"trace[x̂, p̂] = {np.trace(C_xp):.3e}（任何有限維的交換子 trace 都是 0）")
check("trace[A,B] = 0（恆成立）", np.trace(C_xp), 0.0, tol=1e-10)
print(f"但 trace(iℏI) = iℏn = {1j * hbar * n}")
check("所以 [x̂,p̂] = iℏI 在有限維不可能成立",
      not np.allclose(C_xp, 1j * hbar * np.eye(n)))

# 不過「作用在平滑的波函數上」時它仍然近似成立
print(f"\n[x̂,p̂] 的非零元素都在次對角線上：[x̂,p̂]ⱼ,ⱼ₊₁ = "
      f"{C_xp[n // 2, n // 2 + 1]:.4f}（= iℏ/2）")
check("次對角線 = iℏ/2", C_xp[n // 2, n // 2 + 1], 0.5j * hbar, tol=1e-12)
check("每列的元素和 = iℏ（弱意義下的正則關係）",
      C_xp[n // 2, :].sum(), 1j * hbar, tol=1e-12)
psi_smooth = np.exp(-x_grid ** 2 / 8).astype(complex)
lhs_cr = (C_xp @ psi_smooth)[n // 4:3 * n // 4]
rhs_cr = (1j * hbar * psi_smooth)[n // 4:3 * n // 4]
rel_cr = np.max(np.abs(lhs_cr - rhs_cr)) / np.max(np.abs(rhs_cr))
print(f"作用在平滑高斯波包上：‖[x̂,p̂]ψ − iℏψ‖/‖iℏψ‖ = {rel_cr:.3%}（O(h²) 誤差）")
check("[x̂,p̂]ψ ≈ iℏψ（平滑函數、遠離邊界）", rel_cr < 0.02)
print("  ⇒ 正則交換關係只能在無窮維成立；有限維離散化只能「弱意義」地近似它")

# %% [markdown]
# ## 21.3　不確定性原理
#
# 定義不確定度 $\Delta A=\sqrt{\langle A^2\rangle-\langle A\rangle^2}$，則
# **Robertson 不等式**
#
# $$\boxed{\Delta A\,\Delta B\ge\frac12\left|\langle[A,B]\rangle\right|}$$
#
# 證明只用 Schwarz 不等式（第 1 章！）：
# 令 $|f\rangle=(A-\langle A\rangle)|\psi\rangle$、$|g\rangle=(B-\langle B\rangle)|\psi\rangle$，
# 則 $\Delta A^2\Delta B^2=\langle f|f\rangle\langle g|g\rangle\ge|\langle f|g\rangle|^2
# \ge(\operatorname{Im}\langle f|g\rangle)^2=\frac14|\langle[A,B]\rangle|^2$。

# %%
section("21.3 不確定性原理 = Schwarz 不等式")


def uncertainty(psi, A):
    """ΔA = √(⟨A²⟩ − ⟨A⟩²)。"""
    mean = expect(psi, A)
    mean_sq = expect(psi, A @ A)
    return np.sqrt(max(mean_sq - mean ** 2, 0.0))


# 用 2x2 的自旋系統（Pauli 矩陣）
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1j], [1j, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
print("Pauli 矩陣的交換關係 [σᵢ,σⱼ] = 2iε_ijk σₖ：")
for (i, A_s), (j, B_s), (k, C_s) in [((0, sigma_x), (1, sigma_y), (2, sigma_z)),
                                     ((1, sigma_y), (2, sigma_z), (0, sigma_x)),
                                     ((2, sigma_z), (0, sigma_x), (1, sigma_y))]:
    check(f"  [σ{'xyz'[i]}, σ{'xyz'[j]}] = 2iσ{'xyz'[k]}", comm(A_s, B_s), 2j * C_s)
check("σᵢ² = I", sigma_x @ sigma_x, np.eye(2, dtype=complex))
check("σᵢσⱼ = δᵢⱼI + iε_ijk σₖ（i=x, j=y）", sigma_x @ sigma_y, 1j * sigma_z)
check("Pauli 矩陣都是 Hermitian 且 unitary",
      is_hermitian(sigma_x) and np.allclose(dagger(sigma_x) @ sigma_x, np.eye(2)))
check("trace σᵢ = 0（無跡 ⇒ 特徵值 ±1）", np.trace(sigma_x), 0.0)
check("特徵值 = ±1", np.sort(np.real(np.linalg.eigvalsh(sigma_x))),
      np.array([-1.0, 1.0]))

print(f"\n{'狀態':<28} | {'Δσx':>8} | {'Δσz':>8} | {'ΔσxΔσz':>10} | "
      f"{'|⟨[σx,σz]⟩|/2':>14}")
print("-" * 80)
test_states = {
    "|↑⟩ = (1,0)（σz 的特徵態）": np.array([1.0, 0.0], dtype=complex),
    "|→⟩ = (1,1)/√2（σx 的特徵態）": np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
    "(1, i)/√2（σy 的特徵態）": np.array([1.0, 1j]) / np.sqrt(2),
    "(2, 1)/√5（一般狀態）": np.array([2.0, 1.0], dtype=complex) / np.sqrt(5),
}
for name, st in test_states.items():
    dx, dz = uncertainty(st, sigma_x), uncertainty(st, sigma_z)
    bound = abs(np.vdot(st, comm(sigma_x, sigma_z) @ st)) / 2
    print(f"{name:<28} | {dx:>8.4f} | {dz:>8.4f} | {dx * dz:>10.4f} | {bound:>14.4f}")
    check(f"  {name}：ΔσxΔσz ≥ |⟨[σx,σz]⟩|/2", dx * dz >= bound - 1e-12)
check("σz 的特徵態：Δσz = 0（精確），但 Δσx = 1（完全不確定）",
      (uncertainty(np.array([1.0, 0.0], dtype=complex), sigma_z),
       uncertainty(np.array([1.0, 0.0], dtype=complex), sigma_x)), (0.0, 1.0))

# 隨機狀態的大規模驗證
print("\n隨機狀態的大規模驗證（Robertson 不等式）：")
violations = 0
min_slack = np.inf
for _ in range(20000):
    st = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    st = st / np.linalg.norm(st)
    dx, dz = uncertainty(st, sigma_x), uncertainty(st, sigma_z)
    bound = abs(np.vdot(st, comm(sigma_x, sigma_z) @ st)) / 2
    slack = dx * dz - bound
    min_slack = min(min_slack, slack)
    if slack < -1e-12:
        violations += 1
print(f"  20000 個隨機狀態：違反次數 = {violations}，最小餘裕 = {min_slack:.3e}")
check("沒有任何狀態違反不確定性原理", violations == 0)

# 用 Schwarz 不等式逐步驗證證明
st = test_states["(2, 1)/√5（一般狀態）"]
f_vec = (sigma_x - expect(st, sigma_x) * np.eye(2)) @ st
g_vec = (sigma_z - expect(st, sigma_z) * np.eye(2)) @ st
print("\n證明的每一步（Schwarz 不等式）：")
check("  ΔA² = ⟨f|f⟩", uncertainty(st, sigma_x) ** 2, np.vdot(f_vec, f_vec).real)
check("  ΔB² = ⟨g|g⟩", uncertainty(st, sigma_z) ** 2, np.vdot(g_vec, g_vec).real)
check("  Schwarz：⟨f|f⟩⟨g|g⟩ ≥ |⟨f|g⟩|²",
      np.vdot(f_vec, f_vec).real * np.vdot(g_vec, g_vec).real
      >= abs(np.vdot(f_vec, g_vec)) ** 2 - 1e-12)
check("  |⟨f|g⟩|² ≥ (Im⟨f|g⟩)²",
      abs(np.vdot(f_vec, g_vec)) ** 2 >= np.imag(np.vdot(f_vec, g_vec)) ** 2 - 1e-12)
check("  2·Im⟨f|g⟩ = ⟨[A,B]⟩/i（即 Im 部分就是交換子）",
      2 * np.imag(np.vdot(f_vec, g_vec)),
      np.imag(np.vdot(st, comm(sigma_x, sigma_z) @ st) / 1j) * 0
      + np.real(np.vdot(st, comm(sigma_x, sigma_z) @ st) / 1j), tol=1e-10)

# 位置–動量的不確定性（高斯態達到下界）
section("21.3 高斯波包達到不確定性的下界")

def gaussian_uncertainty(sigma, n_grid, L_box=24.0):
    """用中央差分的 p̂ 計算高斯波包的 Δx·Δp。"""
    h_g = L_box / n_grid
    xg = (np.arange(n_grid) - n_grid / 2) * h_g
    Xg = np.diag(xg).astype(complex)
    Dg = np.zeros((n_grid, n_grid), dtype=complex)
    for i in range(1, n_grid - 1):
        Dg[i, i + 1] = 1 / (2 * h_g)
        Dg[i, i - 1] = -1 / (2 * h_g)
    Pg = -1j * Dg                                   # ℏ = 1
    psi_g = np.exp(-xg ** 2 / (4 * sigma ** 2)).astype(complex)
    psi_g = psi_g / np.linalg.norm(psi_g)
    return uncertainty(psi_g, Xg), uncertainty(psi_g, Pg), h_g


print("注意：中央差分的 p̂ 會低估高頻成分（Δp 偏小 O(h²)），")
print("      所以「離散的」ΔxΔp 可能略低於 ℏ/2；網格變細就會收斂回 1/2。")
print(f"\n{'σ':>6} | {'n':>6} | {'h/σ':>8} | {'Δx':>10} | {'Δp':>10} | "
      f"{'ΔxΔp':>10} | {'偏差':>10}")
print("-" * 72)
for sigma in [1.0, 2.0]:
    for n_grid in [200, 400, 800]:
        dx, dp, h_g = gaussian_uncertainty(sigma, n_grid)
        print(f"{sigma:>6} | {n_grid:>6} | {h_g / sigma:>8.4f} | {dx:>10.6f} | "
              f"{dp:>10.6f} | {dx * dp:>10.6f} | {dx * dp - 0.5:>10.2e}")
    check(f"  σ = {sigma}：網格變細時 ΔxΔp → ℏ/2",
          gaussian_uncertainty(sigma, 800)[0] * gaussian_uncertainty(sigma, 800)[1],
          0.5, tol=2e-3)
    check(f"  σ = {sigma}：Δx = σ（精確）", gaussian_uncertainty(sigma, 800)[0],
          sigma, tol=sigma * 1e-3)
    err_coarse = abs(gaussian_uncertainty(sigma, 200)[0]
                     * gaussian_uncertainty(sigma, 200)[1] - 0.5)
    err_fine = abs(gaussian_uncertainty(sigma, 800)[0]
                   * gaussian_uncertainty(sigma, 800)[1] - 0.5)
    check(f"  σ = {sigma}：誤差隨 h² 下降（細網格誤差小 10 倍以上）",
          err_fine < err_coarse / 10)
print("  ⇒ 高斯波包是「最小不確定態」ΔxΔp = ℏ/2（squeezed state 的出發點）")

# %% [markdown]
# ## 21.4　梯子算子與諧振子
#
# $$a=\frac{1}{\sqrt{2}}(\hat x+i\hat p),\qquad
# a^{\dagger}=\frac{1}{\sqrt2}(\hat x-i\hat p),\qquad
# [a,a^{\dagger}]=1$$
#
# 數算子 $N=a^{\dagger}a$ 的特徵值是 $0,1,2,\dots$，而
#
# $$H=\hbar\omega\left(N+\tfrac12\right),\qquad
# a^{\dagger}|n\rangle=\sqrt{n+1}|n+1\rangle,\qquad
# a|n\rangle=\sqrt n|n-1\rangle$$
#
# 在**有限維截斷**下（$n<N_{\max}$）這些關係幾乎精確成立 ──
# 只有最後一行會出現截斷誤差，這是數值量子力學的標準做法。

# %%
section("21.4 梯子算子（有限維截斷）")

N_max = 12
# 在數基底下：a 的矩陣元素 ⟨n−1|a|n⟩ = √n
a_op = np.diag(np.sqrt(np.arange(1, N_max)), 1).astype(complex)
a_dag = dagger(a_op)
N_op = a_dag @ a_op
show_matrix("a（降算子，上對角線 = √n）", np.real(a_op[:5, :5]))
show_matrix("N = a†a（數算子）", np.real(N_op[:5, :5]))
check("N 是 Hermitian", is_hermitian(N_op))
check("N 的特徵值 = 0,1,2,...", np.real(np.diag(N_op)),
      np.arange(N_max, dtype=float))

C_aa = comm(a_op, a_dag)
print(f"\n[a, a†] 的對角線 = {np.round(np.real(np.diag(C_aa)), 6)}")
check("[a,a†] = I（除了最後一個截斷的位置）",
      np.real(np.diag(C_aa))[:-1], np.ones(N_max - 1))
print(f"  最後一個元素是 {np.real(C_aa[-1, -1]):.0f}（截斷造成的誤差，"
      f"因為有限維的 trace[a,a†] 必須為 0）")
check("trace[a,a†] = 0（有限維的必然結果）", np.trace(C_aa), 0.0, tol=1e-10)

check("[N, a] = −a", comm(N_op, a_op), -a_op, tol=1e-10)
check("[N, a†] = a†", comm(N_op, a_dag), a_dag, tol=1e-10)
print("  ⇒ 這兩條交換關係就是「梯子」的來源：a 降一階、a† 升一階")

# 梯子的作用
for n_level in [0, 1, 3]:
    ket_n = np.zeros(N_max, dtype=complex)
    ket_n[n_level] = 1.0
    up = a_dag @ ket_n
    down = a_op @ ket_n
    check(f"  a†|{n_level}⟩ = √{n_level+1}|{n_level+1}⟩",
          up, np.sqrt(n_level + 1) * np.eye(N_max)[:, n_level + 1])
    if n_level > 0:
        check(f"  a|{n_level}⟩ = √{n_level}|{n_level-1}⟩",
              down, np.sqrt(n_level) * np.eye(N_max)[:, n_level - 1])
ket_0 = np.eye(N_max)[:, 0].astype(complex)
check("a|0⟩ = 0（基態無法再降）", a_op @ ket_0, np.zeros(N_max, dtype=complex))

# Hamiltonian 的能階
H_osc = N_op + 0.5 * np.eye(N_max)
E_osc = np.real(np.linalg.eigvalsh(H_osc))
print(f"\n能階 E = ℏω(n + ½) = {np.round(E_osc[:6], 4)}")
check("E_n = n + 1/2", E_osc, np.arange(N_max) + 0.5)
print("  （與第 17 章用 Sturm–Liouville 離散化得到的 2n+1 一致，差個因子 2）")

# x 與 p 用梯子算子表示
X_lad = (a_op + a_dag) / np.sqrt(2)
P_lad = 1j * (a_dag - a_op) / np.sqrt(2)
check("x̂ = (a + a†)/√2 是 Hermitian", is_hermitian(X_lad))
check("p̂ = i(a† − a)/√2 是 Hermitian", is_hermitian(P_lad))
check("H = (x̂² + p̂²)/2（截斷誤差只在邊緣）",
      np.real((X_lad @ X_lad + P_lad @ P_lad) / 2)[:N_max - 2, :N_max - 2],
      np.real(H_osc)[:N_max - 2, :N_max - 2], tol=1e-10)
print("\n各能階的不確定度（ΔxΔp）：")
for n_level in [0, 1, 2, 5]:
    ket_n = np.eye(N_max)[:, n_level].astype(complex)
    dx = uncertainty(ket_n, X_lad)
    dp = uncertainty(ket_n, P_lad)
    print(f"  |{n_level}⟩：Δx = {dx:.6f}，Δp = {dp:.6f}，ΔxΔp = {dx * dp:.6f}"
          f"（理論 n + ½ = {n_level + 0.5}）")
    check(f"  |{n_level}⟩：ΔxΔp = n + ½", dx * dp, n_level + 0.5, tol=1e-9)
check("基態達到最小不確定度 1/2",
      uncertainty(ket_0, X_lad) * uncertainty(ket_0, P_lad), 0.5)

# %% [markdown]
# ## 21.5　時間演化 = unitary 矩陣
#
# Schrödinger 方程 $i\hbar\,\partial_t|\psi\rangle=H|\psi\rangle$ 的解
#
# $$|\psi(t)\rangle=U(t)|\psi(0)\rangle,\qquad U(t)=e^{-iHt/\hbar}$$
#
# $H$ Hermitian $\Rightarrow$ $U$ **unitary** $\Rightarrow$ **機率守恆**
# （$\|\psi\|$ 不變）── 這就是第 8 章「反對稱矩陣的指數是正交矩陣」的複數版。

# %%
section("21.5 時間演化算子是 unitary")

H_t = np.array([[1.0, 0.5 - 0.2j], [0.5 + 0.2j, -1.0]])
check("H Hermitian", is_hermitian(H_t))
for t in [0.0, 0.5, 1.7, 3.0]:
    U = matrix_exp_series(-1j * H_t, t, 80)
    check(f"  t = {t}：U†U = I（unitary ⇒ 機率守恆）",
          dagger(U) @ U, np.eye(2, dtype=complex), tol=1e-10)
    check(f"  t = {t}：|det U| = 1", abs(np.linalg.det(U)), 1.0, tol=1e-10)

psi0 = np.array([1.0, 0.0], dtype=complex)
print(f"\n初始態 |ψ(0)⟩ = {psi0}")
print(f"{'t':>6} | {'|c₁|²':>10} | {'|c₂|²':>10} | {'總機率':>10} | {'⟨H⟩':>10}")
print("-" * 54)
for t in [0.0, 0.5, 1.0, 2.0, 3.0]:
    U = matrix_exp_series(-1j * H_t, t, 80)
    psi_t = U @ psi0
    probs_t = np.abs(psi_t) ** 2
    print(f"{t:>6} | {probs_t[0]:>10.6f} | {probs_t[1]:>10.6f} | "
          f"{probs_t.sum():>10.6f} | {expect(psi_t, H_t):>10.6f}")
    check(f"  t = {t}：機率守恆", float(probs_t.sum()), 1.0, tol=1e-10)
    check(f"  t = {t}：⟨H⟩ 守恆（能量守恆）", expect(psi_t, H_t),
          expect(psi0, H_t), tol=1e-10)

# Rabi 振盪：自旋在磁場中的進動
print("\nRabi 振盪（σx 作為 Hamiltonian，初始態 |↑⟩）：")
U_rabi = lambda t: matrix_exp_series(-1j * sigma_x, t, 80)
for t in [0.0, np.pi / 4, np.pi / 2, np.pi]:
    psi_t = U_rabi(t) @ np.array([1.0, 0.0], dtype=complex)
    p_down = abs(psi_t[1]) ** 2
    print(f"  t = {t:.4f}：P(↓) = {p_down:.6f}（理論 sin²t = {np.sin(t) ** 2:.6f}）")
    check(f"  t = {t:.4f}：P(↓) = sin²t", p_down, np.sin(t) ** 2, tol=1e-10)
check("t = π/2 時完全翻轉（P(↓) = 1）",
      abs((U_rabi(np.pi / 2) @ np.array([1.0, 0.0], dtype=complex))[1]) ** 2, 1.0,
      tol=1e-9)

# 守恆量 = 與 H 可交換的算子
print("\n守恆量的判準：[A, H] = 0 ⇒ ⟨A⟩ 不隨時間改變")
for name, A_obs in [("H 自己", H_t), ("σx（與 σx-Hamiltonian 可交換）", sigma_x),
                    ("σz（不可交換）", sigma_z)]:
    commutes = np.allclose(comm(A_obs, sigma_x), 0)
    vals = [expect(U_rabi(t) @ np.array([1.0, 0.5], dtype=complex)
                   / np.linalg.norm(np.array([1.0, 0.5])), A_obs)
            for t in [0.0, 0.7, 1.4, 2.1]]
    conserved = np.allclose(vals, vals[0], atol=1e-9)
    print(f"  {name:<34} [A,H]=0：{str(commutes):<5} ⟨A⟩ 守恆：{conserved}")
    check(f"  {name}：[A,H] = 0 ⟺ ⟨A⟩ 守恆", commutes == conserved)

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
ts = np.linspace(0, 3 * np.pi, 300)
p_up, p_dn, exp_z = [], [], []
for t in ts:
    psi_t = U_rabi(t) @ np.array([1.0, 0.0], dtype=complex)
    p_up.append(abs(psi_t[0]) ** 2)
    p_dn.append(abs(psi_t[1]) ** 2)
    exp_z.append(expect(psi_t, sigma_z))
ax.plot(ts, p_up, "C0", lw=1.6, label="P(up)")
ax.plot(ts, p_dn, "C3", lw=1.6, label="P(down)")
ax.plot(ts, np.array(p_up) + np.array(p_dn), "k--", lw=1, label="total = 1")
ax.set_title("Rabi oscillation (unitary evolution)", fontsize=9)
ax.set_xlabel("t")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
ns = np.arange(6)
x_plot = np.linspace(-5, 5, 400)
for n_level in range(4):
    # 諧振子波函數：Hermite 多項式 × 高斯
    Hn = np.polynomial.hermite.Hermite.basis(n_level)(x_plot)
    psi_n = Hn * np.exp(-x_plot ** 2 / 2)
    psi_n = psi_n / np.max(np.abs(psi_n))
    ax.plot(x_plot, psi_n + 2.5 * n_level, lw=1.5,
            label=f"n = {n_level}, E = {n_level + 0.5}")
    ax.axhline(2.5 * n_level, color="k", lw=0.4, alpha=0.3)
ax.plot(x_plot, x_plot ** 2 / 4, "k--", lw=1, alpha=0.5, label="V = x^2/2 (scaled)")
ax.set_xlim(-5, 5)
ax.set_ylim(-1.5, 10)
ax.set_title("Harmonic oscillator eigenstates (ladder)", fontsize=9)
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
finish(fig, "ch21_quantum")

# %% [markdown]
# ## 21.6　密度矩陣：混合態與純態
#
# $$\rho=\sum_ip_i|\psi_i\rangle\langle\psi_i|$$
#
# | 性質 | 線性代數 |
# |---|---|
# | $\rho^{\dagger}=\rho$ | Hermitian |
# | $\rho\succeq0$ | 半正定 |
# | $\operatorname{tr}\rho=1$ | 跡為 1 |
# | **純態** | $\rho^2=\rho$（投影！）$\iff$ $\operatorname{rank}\rho=1$ |
# | 期望值 | $\langle A\rangle=\operatorname{tr}(\rho A)$ |
#
# 純態是秩一投影矩陣（第 4 章），混合態是它們的凸組合。

# %%
section("21.6 密度矩陣")

psi_a = np.array([1.0, 0.0], dtype=complex)
psi_b = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
rho_pure = np.outer(psi_a, psi_a.conj())
rho_mixed = 0.7 * np.outer(psi_a, psi_a.conj()) + 0.3 * np.outer(psi_b, psi_b.conj())
show_matrix("純態的 ρ = |ψ⟩⟨ψ|", np.real(rho_pure))
show_matrix("混合態的 ρ（0.7|a⟩⟨a| + 0.3|b⟩⟨b|）", np.real(rho_mixed))

for name, rho in [("純態", rho_pure), ("混合態", rho_mixed)]:
    print(f"\n{name}：")
    check("  ρ† = ρ", dagger(rho), rho)
    check("  trace ρ = 1", np.trace(rho).real, 1.0)
    check("  ρ 半正定", np.all(np.linalg.eigvalsh(rho) > -1e-12))
    purity = np.trace(rho @ rho).real
    print(f"  純度 tr(ρ²) = {purity:.6f}（純態 = 1，最大混合 = 1/2）")
    if name == "純態":
        check("  ρ² = ρ（投影矩陣）", rho @ rho, rho)
        check("  rank ρ = 1", np.linalg.matrix_rank(rho, tol=1e-10), 1)
        check("  tr(ρ²) = 1", purity, 1.0)
    else:
        check("  ρ² ≠ ρ（非投影）", not np.allclose(rho @ rho, rho))
        check("  tr(ρ²) < 1", purity < 1.0)

# 期望值 = trace(ρA)
print("\n期望值的兩種算法：")
check("純態：tr(ρA) = ⟨ψ|A|ψ⟩", np.trace(rho_pure @ sigma_z).real,
      expect(psi_a, sigma_z))
mixed_expect = 0.7 * expect(psi_a, sigma_z) + 0.3 * expect(psi_b, sigma_z)
check("混合態：tr(ρA) = Σpᵢ⟨ψᵢ|A|ψᵢ⟩", np.trace(rho_mixed @ sigma_z).real,
      mixed_expect)
print(f"  ⟨σz⟩ = {np.trace(rho_mixed @ sigma_z).real:.6f}")

# 最大混合態 = I/2
rho_max = np.eye(2, dtype=complex) / 2
print(f"\n最大混合態 ρ = I/2：純度 = {np.trace(rho_max @ rho_max).real:.4f}")
check("最大混合態的所有 ⟨σᵢ⟩ = 0（完全無資訊）",
      [np.trace(rho_max @ s).real for s in (sigma_x, sigma_y, sigma_z)],
      [0.0, 0.0, 0.0])
check("von Neumann 熵最大：λ = (1/2, 1/2)",
      np.sort(np.linalg.eigvalsh(rho_max)), np.array([0.5, 0.5]))
entropy = lambda rho: -sum(l * np.log2(l) for l in np.linalg.eigvalsh(rho) if l > 1e-15)
print(f"{'狀態':<16} | {'純度 tr(ρ²)':>12} | {'von Neumann 熵':>16}")
print("-" * 50)
for name, rho in [("純態", rho_pure), ("混合態", rho_mixed), ("最大混合", rho_max)]:
    print(f"{name:<16} | {np.trace(rho @ rho).real:>12.6f} | {entropy(rho):>16.6f}")
check("純態的熵 = 0", entropy(rho_pure), 0.0, tol=1e-12)
check("最大混合態的熵 = 1 bit", entropy(rho_max), 1.0)

# Bloch 球表示
print("\nBloch 球：ρ = (I + r·σ)/2，|r| ≤ 1（純態在球面上）")
for name, rho in [("純態", rho_pure), ("混合態", rho_mixed), ("最大混合", rho_max)]:
    r_vec = np.array([np.trace(rho @ s).real for s in (sigma_x, sigma_y, sigma_z)])
    check(f"  {name}：ρ = (I + r·σ)/2",
          (np.eye(2) + sum(r_vec[i] * s for i, s in
                           enumerate((sigma_x, sigma_y, sigma_z)))) / 2, rho)
    print(f"  {name:<10} r = {np.round(r_vec, 4)}，|r| = {np.linalg.norm(r_vec):.6f}")
    check(f"  {name}：|r| ≤ 1", np.linalg.norm(r_vec) <= 1 + 1e-12)
check("純態 ⟺ |r| = 1（在球面上）",
      np.linalg.norm([np.trace(rho_pure @ s).real
                      for s in (sigma_x, sigma_y, sigma_z)]), 1.0)

# %% [markdown]
# ## 動手練習
#
# 1. 證明兩個 Hermitian 算子的乘積 $AB$ 一般**不是** Hermitian，但 $\frac12(AB+BA)$ 是。
# 2. 驗證 $e^{i\theta\sigma_x}=\cos\theta\,I+i\sin\theta\,\sigma_x$（Euler 公式的矩陣版）。
# 3. 求角動量 $J_z$ 與 $J^2$ 的共同特徵基底（$j=1$，$3\times3$ 矩陣）。
# 4. 驗證「$[A,H]=0$ 的算子個數」= $H$ 的交換子代數的維度（與能階退化有關）。
# 5. 對兩量子位元的糾纏態 $(|00\rangle+|11\rangle)/\sqrt2$，計算單一量子位元的
#    約化密度矩陣，確認它是最大混合態。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
A_h = np.array([[1.0, 1j], [-1j, 2.0]])
B_h = np.array([[0.0, 2.0], [2.0, 3.0]], dtype=complex)
check("練習 1：A, B Hermitian", is_hermitian(A_h) and is_hermitian(B_h))
check("  AB 不是 Hermitian", not is_hermitian(A_h @ B_h))
check("  (AB + BA)/2 是 Hermitian（對稱化）",
      is_hermitian((A_h @ B_h + B_h @ A_h) / 2))
check("  i[A,B] 也是 Hermitian", is_hermitian(1j * comm(A_h, B_h)))
print("  ⇒ 量子力學裡要把古典的乘積 xp 量子化成 (x̂p̂ + p̂x̂)/2（Weyl 排序）")

# 練習 2
print("\n練習 2：矩陣的 Euler 公式")
for theta in [0.3, 1.0, np.pi / 2]:
    lhs = matrix_exp_series(1j * theta * sigma_x, 1.0, 80)
    rhs = np.cos(theta) * np.eye(2) + 1j * np.sin(theta) * sigma_x
    check(f"  θ = {theta:.4f}：e^{{iθσx}} = cosθ·I + i sinθ·σx", lhs, rhs, tol=1e-12)
print("  原因：σx² = I，所以冪級數可以分成偶次（I）與奇次（σx）兩組")

# 練習 3：j = 1 的角動量
print("\n練習 3：j = 1 的角動量矩陣")
Jz = np.diag([1.0, 0.0, -1.0]).astype(complex)
Jp = np.sqrt(2) * np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                           dtype=complex)
Jm = dagger(Jp)
Jx = (Jp + Jm) / 2
Jy = (Jp - Jm) / (2j)
J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
show_matrix("J² 的實部", np.real(J2))
check("  J² = j(j+1)I = 2I", J2, 2 * np.eye(3, dtype=complex))
check("  [J², Jz] = 0（可同時測量）", comm(J2, Jz),
      np.zeros((3, 3), dtype=complex))
check("  [Jx, Jy] = iJz", comm(Jx, Jy), 1j * Jz)
check("  [Jy, Jz] = iJx", comm(Jy, Jz), 1j * Jx)
check("  Jz 的特徵值 = m = 1, 0, −1", np.sort(np.real(np.linalg.eigvalsh(Jz))),
      np.array([-1.0, 0.0, 1.0]))
check("  J₊|m=0⟩ = √2|m=1⟩（梯子）",
      Jp @ np.array([0.0, 1.0, 0.0], dtype=complex),
      np.sqrt(2) * np.array([1.0, 0.0, 0.0], dtype=complex))

# 練習 4
print("\n練習 4：與 H 可交換的算子空間維度")
for name, H_ex, expect_dim in [("無退化 diag(1,2,3)", np.diag([1.0, 2.0, 3.0]), 3),
                               ("二重退化 diag(1,1,3)", np.diag([1.0, 1.0, 3.0]), 5),
                               ("完全退化 I", np.eye(3), 9)]:
    # 解 [A, H] = 0 ⇒ (I⊗H − Hᵀ⊗I)vec(A) = 0
    Sys = np.kron(np.eye(3), H_ex) - np.kron(H_ex.T, np.eye(3))
    dim = 9 - np.linalg.matrix_rank(Sys, tol=1e-10)
    print(f"  {name:<22} 可交換算子空間維度 = {dim}（理論 Σmᵢ² = {expect_dim}）")
    check(f"  {name}：維度 = Σmᵢ²", dim, expect_dim)
print("  ⇒ 退化越嚴重，守恆量越多（對稱性越高）—— 這是第 23 章群論的入口")

# 練習 5：糾纏與約化密度矩陣
print("\n練習 5：Bell 態的約化密度矩陣")
bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)   # (|00⟩+|11⟩)/√2
rho_bell = np.outer(bell, bell.conj())
check("  Bell 態是純態（ρ² = ρ）", rho_bell @ rho_bell, rho_bell)
check("  trace = 1", np.trace(rho_bell).real, 1.0)
# 對第二個量子位元取跡
rho_A = np.zeros((2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        rho_A[i, j] = sum(rho_bell[2 * i + k, 2 * j + k] for k in range(2))
show_matrix("  約化密度矩陣 ρ_A 的實部", np.real(rho_A))
check("  ρ_A = I/2（最大混合！）", rho_A, np.eye(2, dtype=complex) / 2)
check("  純度 tr(ρ_A²) = 1/2", np.trace(rho_A @ rho_A).real, 0.5)
check("  糾纏熵 = 1 bit", entropy(rho_A), 1.0)
print("  ⇒ 整體是純態、子系統卻是最大混合 —— 這就是量子糾纏")
print("     線性代數的說法：Bell 態的 2x2 係數矩陣的奇異值是 (1/√2, 1/√2)，")
print("     Schmidt 秩 = 2 > 1 ⇒ 不可分解（第 9 章 SVD 的應用）")
coef_matrix = bell.reshape(2, 2)
sv_bell = np.linalg.svd(coef_matrix, compute_uv=False)
check("  Schmidt 係數（奇異值）= (1/√2, 1/√2)", np.sort(sv_bell),
      np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
check("  Schmidt 秩 = 2 ⇒ 糾纏", np.linalg.matrix_rank(coef_matrix, tol=1e-10), 2)
prod_state = np.kron(np.array([1.0, 0.0]), np.array([1.0, 1.0]) / np.sqrt(2))
check("  可分離態的 Schmidt 秩 = 1",
      np.linalg.matrix_rank(prod_state.reshape(2, 2), tol=1e-10), 1)

# %% [markdown]
# ## 本章重點回顧
#
# * 量子力學的公設逐條對應線性代數：狀態 = 單位向量、可觀測量 = **Hermitian 矩陣**、
#   測量值 = 特徵值（必為實數）、機率 = $|c_n|^2$、期望值 = Rayleigh 商。
# * **完備性關係** $\sum_n|a_n\rangle\langle a_n|=I$ 就是第 7 章的譜分解
#   $S=Q\Lambda Q^{\mathsf H}$；每個 $|a_n\rangle\langle a_n|$ 是秩一投影。
# * **交換子** $[A,B]=0\iff$ 可同時對角化 $\iff$ 共用特徵基底 $\iff$ 可同時精確測量。
#   交換子滿足 Jacobi 恆等式（與叉積同構）。
# * $[\hat x,\hat p]=i\hbar I$ 在**任何有限維空間都不可能成立**
#   （因為 $\operatorname{tr}[A,B]=0$）── 量子力學必須用無窮維 Hilbert 空間。
# * **不確定性原理就是 Schwarz 不等式**：
#   $\Delta A\Delta B\ge\frac12|\langle[A,B]\rangle|$。高斯波包達到下界。
# * 梯子算子 $[a,a^{\dagger}]=1$、$N=a^{\dagger}a$ 的特徵值 $0,1,2,\dots$、
#   $\Delta x\Delta p=n+\frac12$；有限維截斷下只有最後一列有誤差。
# * 時間演化 $U=e^{-iHt/\hbar}$ 是 **unitary** ⇒ 機率守恆
#   （第 8 章「反對稱的指數是正交」的複數版）；$[A,H]=0\iff\langle A\rangle$ 守恆。
# * **密度矩陣**：Hermitian、半正定、跡為 1；純態 $\iff\rho^2=\rho\iff\operatorname{rank}=1$。
#   糾纏 $\iff$ 係數矩陣的 **Schmidt 秩**（= 奇異值個數）$>1$。
#
# 下一章：變分法、積分方程與無窮維線性代數的收尾。
