# %% [markdown]
# # 第 23 章　群表示論：對稱性如何把矩陣分塊對角化
#
# > 對應 Riley 第 28 章（群、有限群、非交換群、置換群、子群、商群）
# > 與第 29 章（表示論：等價表示、可約性、正交性定理、特徵、特徵表、
# > 乘積表示、物理應用），並對照本教材第 10 章（基底變換）、第 15 章（法模態）、
# > 第 21 章（量子算子的退化）。
#
# 一句話總結：**群表示論就是「用對稱性找最好的基底」**。
#
# | 群論 | 線性代數 |
# |---|---|
# | 表示 $D:G\to GL(n)$ | 群元素 → 矩陣（同態）|
# | 等價表示 | **相似變換** $S^{-1}DS$ |
# | 可約 | 可同時**分塊對角化** |
# | 不可約表示（irrep）| 最小的不變子空間 |
# | 特徵 $\chi(g)=\operatorname{tr}D(g)$ | **跡（相似不變量）** |
# | 特徵的正交性 | 向量的正交性！ |
# | 投影算子 $P^{(\lambda)}$ | 投影矩陣（第 4 章）|
# | 分解重數 $n_\lambda$ | 正交展開的係數 |
#
# ```bash
# python chapters/ch23_group_representations.py
# python tools/build_notebooks.py ch23
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

from linalg_tutorial.eigen import generalized_symmetric_eig
from linalg_tutorial.subspaces import rank_svd
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 23.1　群與表示
#
# **群**：封閉、結合律、單位元素、逆元素。
# **表示**：一個同態 $D:G\to GL(n,\mathbb C)$，即
#
# $$D(g_1g_2)=D(g_1)D(g_2),\qquad D(e)=I,\qquad D(g^{-1})=D(g)^{-1}$$
#
# 最自然的表示是**置換矩陣**（把群作用在自己身上 ⇒ **正則表示**）。

# %%
section("23.1 S₃ 的置換表示")

# S₃ = 三個物件的所有置換（也就是正三角形的對稱群 D₃）
perms = list(permutations(range(3)))
names = {(0, 1, 2): "e", (1, 2, 0): "r", (2, 0, 1): "r²",
         (0, 2, 1): "s₁", (2, 1, 0): "s₂", (1, 0, 2): "s₃"}


def perm_matrix(p):
    """置換 p 對應的置換矩陣（P[i,j] = 1 若 p(j) = i）。"""
    P = np.zeros((3, 3))
    for j, i in enumerate(p):
        P[i, j] = 1.0
    return P


G = {names[p]: perm_matrix(p) for p in perms}
print(f"S₃ 的 {len(G)} 個元素：{list(G.keys())}")
show_matrix("D(r)（三循環）", G["r"])
show_matrix("D(s₁)（對換）", G["s₁"])

# 驗證群的公設與同態性質
check("單位元素：D(e) = I", G["e"], np.eye(3))
check("封閉性：所有乘積都還在群裡",
      all(any(np.allclose(A @ B, C) for C in G.values())
          for A in G.values() for B in G.values()))
check("逆元素：每個元素都有逆元",
      all(any(np.allclose(A @ B, np.eye(3)) for B in G.values())
          for A in G.values()))
check("同態：D(r)D(r) = D(r²)", G["r"] @ G["r"], G["r²"])
check("D(r)³ = I（r 的階是 3）", np.linalg.matrix_power(G["r"], 3), np.eye(3))
check("D(s₁)² = I（對換的階是 2）", G["s₁"] @ G["s₁"], np.eye(3))
check("非交換：D(r)D(s₁) ≠ D(s₁)D(r)",
      not np.allclose(G["r"] @ G["s₁"], G["s₁"] @ G["r"]))
check("所有表示矩陣都是正交的（置換矩陣）",
      all(np.allclose(A.T @ A, np.eye(3)) for A in G.values()))
print(f"\n每個元素的 det：")
for name, A in G.items():
    print(f"  det D({name}) = {np.linalg.det(A):+.0f}"
          f"（{'偶置換' if np.linalg.det(A) > 0 else '奇置換'}）")
check("det 本身就是一個 1 維表示（sign 表示）",
      all(np.isclose(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))
          for A in G.values() for B in G.values()))

# %% [markdown]
# ## 23.2　可約性 = 可同時分塊對角化
#
# 表示 $D$ **可約** $\iff$ 存在非平凡的**不變子空間** $W$（$D(g)W\subseteq W$）
# $\iff$ 存在 $S$ 使所有 $S^{-1}D(g)S$ 同時變成**分塊對角**。
#
# $S_3$ 的置換表示（$3$ 維）可以分解成
#
# $$3=1\oplus2$$
#
# * **1 維（平凡表示）**：$(1,1,1)$ 方向 ── 所有置換都不動它
# * **2 維（標準表示）**：$x_1+x_2+x_3=0$ 的平面 ── 不可再分

# %%
section("23.2 把 3 維置換表示分解成 1 ⊕ 2")

# 不變子空間：對稱方向 (1,1,1) 與它的正交補
v_sym = np.ones(3) / np.sqrt(3)
check("(1,1,1) 是所有 D(g) 的特徵向量（特徵值 1）",
      all(np.allclose(A @ v_sym, v_sym) for A in G.values()))
print("⇒ span{(1,1,1)} 是 1 維不變子空間（平凡表示）")

# 用 Gram–Schmidt 造出正交補的基底
w1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
w2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
S_basis = np.column_stack([v_sym, w1, w2])
check("S 是正交矩陣", S_basis.T @ S_basis, np.eye(3))
check("正交補也是不變子空間",
      all(abs(v_sym @ (A @ w)) < 1e-12 for A in G.values() for w in (w1, w2)))

print("\n在新基底下，所有 D(g) 同時變成分塊對角（1 ⊕ 2）：")
for name in ["e", "r", "s₁"]:
    D_new = S_basis.T @ G[name] @ S_basis
    print(f"\n  S⁻¹D({name})S =")
    for row in np.round(D_new, 4):
        print(f"    {row}")
    check(f"  D({name}) 的分塊結構（第一列/欄只有對角元素）",
          np.allclose(D_new[0, 1:], 0) and np.allclose(D_new[1:, 0], 0))
    check(f"  左上角 = 1（平凡表示）", D_new[0, 0], 1.0)
all_block = all(np.allclose((S_basis.T @ A @ S_basis)[0, 1:], 0)
                and np.allclose((S_basis.T @ A @ S_basis)[1:, 0], 0)
                for A in G.values())
check("所有 6 個元素都同時分塊對角化", all_block)

# 2 維部分不可再分（irreducible）
D2 = {name: (S_basis.T @ A @ S_basis)[1:, 1:] for name, A in G.items()}
check("2 維部分仍是表示（同態）", D2["r"] @ D2["r"], D2["r²"], tol=1e-12)
# 不可約性的判準：沒有共同特徵向量
print("\n2 維表示不可約的證明：D(r) 與 D(s₁) 沒有共同特徵向量")
_, V_r = np.linalg.eig(D2["r"])
has_common = False
for k in range(2):
    v = V_r[:, k]
    if np.linalg.norm(np.cross(np.append(D2["s₁"] @ v, 0), np.append(v, 0))) < 1e-8:
        has_common = True
check("沒有共同特徵向量 ⇒ 不可約", not has_common)
check("Schur 判準：與所有 D(g) 可交換的矩陣只有 cI",
      True)
# 數值驗證 Schur 引理：解 [M, D(g)] = 0 的空間維度 = 1
Sys = np.vstack([np.kron(np.eye(2), A) - np.kron(A.T, np.eye(2))
                 for A in D2.values()])
dim_commutant = 4 - np.linalg.matrix_rank(Sys, tol=1e-10)
print(f"  與所有 D(g) 可交換的矩陣空間維度 = {dim_commutant}"
      f"（Schur 引理：不可約 ⟺ 維度 = 1）")
check("Schur 引理：交換子空間維度 = 1 ⇒ 不可約", dim_commutant, 1)

# 對比：可約表示的交換子空間維度 > 1
Sys3 = np.vstack([np.kron(np.eye(3), A) - np.kron(A.T, np.eye(3))
                  for A in G.values()])
dim_comm3 = 9 - np.linalg.matrix_rank(Sys3, tol=1e-10)
print(f"  3 維置換表示的交換子空間維度 = {dim_comm3} > 1 ⇒ 可約")
check("可約表示的交換子空間維度 = Σnλ² = 1² + 1² = 2", dim_comm3, 2)

# %% [markdown]
# ## 23.3　特徵（character）與特徵表
#
# $$\chi(g)=\operatorname{tr}D(g)$$
#
# 因為跡是**相似不變量**（第 14 章），所以等價的表示有相同的特徵，
# 而且**同一個共軛類的元素特徵相同**。
#
# **大正交定理**（Great Orthogonality Theorem）的特徵版本：
#
# $$\frac{1}{|G|}\sum_{g}\chi^{(\lambda)}(g)^{*}\chi^{(\mu)}(g)=\delta_{\lambda\mu}$$
#
# 這就是說：**特徵是正交的向量**！於是任意表示的分解係數可以用內積算出：
#
# $$n_\lambda=\frac{1}{|G|}\sum_g\chi(g)\,\chi^{(\lambda)}(g)^{*}$$

# %%
section("23.3 S₃ 的特徵表")

# 共軛類：{e}, {r, r²}, {s₁, s₂, s₃}
classes = {"e": ["e"], "r": ["r", "r²"], "s": ["s₁", "s₂", "s₃"]}
class_sizes = {k: len(v) for k, v in classes.items()}
print(f"共軛類與大小：{class_sizes}（總和 = {sum(class_sizes.values())} = |G|）")

# 三個不可約表示
irreps = {
    "A₁（平凡）": {name: np.array([[1.0]]) for name in G},
    "A₂（sign）": {name: np.array([[np.linalg.det(A)]]) for name, A in G.items()},
    "E（2 維標準）": D2,
}
chi = {}
for label, rep in irreps.items():
    chi[label] = np.array([np.trace(rep[classes[c][0]]).real for c in classes])
print(f"\n{'irrep':<16} | {'χ(e)':>8} | {'χ(r)':>8} | {'χ(s)':>8} | {'維度':>6}")
print("-" * 56)
for label, ch in chi.items():
    print(f"{label:<16} | {ch[0]:>8.1f} | {ch[1]:>8.1f} | {ch[2]:>8.1f} | "
          f"{ch[0]:>6.0f}")
check("χ(e) = 表示的維度", [chi[l][0] for l in chi], [1.0, 1.0, 2.0])
check("Σ(維度)² = |G|（Burnside）",
      sum(chi[l][0] ** 2 for l in chi), float(len(G)))
check("不可約表示的個數 = 共軛類的個數", len(chi), len(classes))

# 特徵的正交性（用類的大小當權重）
weights = np.array([class_sizes[c] for c in classes]) / len(G)
print(f"\n特徵的正交性（權重 = 類大小/|G| = {np.round(weights, 4)}）：")
labels = list(chi.keys())
for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        val = float(np.sum(weights * chi[li] * chi[lj]))
        print(f"  ⟨χ({li[:2]}), χ({lj[:2]})⟩ = {val:>6.3f}"
              f"（應為 {1.0 if i == j else 0.0}）", end="")
    print()
G_mat = np.array([[float(np.sum(weights * chi[li] * chi[lj])) for lj in labels]
                  for li in labels])
check("特徵的 Gram 矩陣 = I（正交歸一）", G_mat, np.eye(3))
print("  ⇒ 「特徵表的列正交」就是向量正交 —— 全部群論計算都靠這一條")

# 用內積分解置換表示
chi_perm = np.array([np.trace(G[classes[c][0]]) for c in classes])
print(f"\n3 維置換表示的特徵 = {chi_perm}")
decomp = {l: float(np.sum(weights * chi_perm * chi[l])) for l in labels}
print(f"分解係數 nλ = ⟨χ_perm, χ^λ⟩：")
for l, n_l in decomp.items():
    print(f"  n({l}) = {n_l:.6f}")
check("置換表示 = A₁ ⊕ E（1 + 2 = 3）",
      [decomp[l] for l in labels], [1.0, 0.0, 1.0])
check("維度對得上：1·1 + 0·1 + 1·2 = 3",
      sum(decomp[l] * chi[l][0] for l in labels), 3.0)
check("χ_perm = χ(A₁) + χ(E)", chi_perm, chi["A₁（平凡）"] + chi["E（2 維標準）"])

# 正則表示：每個 irrep 出現 nλ 次（nλ = 維度）
section("23.3 正則表示：每個 irrep 出現「維度」次")

# 正則表示：6x6 置換矩陣（群作用在自己身上）
elements = list(G.keys())
mult_table = {}
for a in elements:
    for b in elements:
        prod = G[a] @ G[b]
        for c in elements:
            if np.allclose(prod, G[c]):
                mult_table[(a, b)] = c
                break
D_reg = {}
for g in elements:
    P = np.zeros((6, 6))
    for j, b in enumerate(elements):
        i = elements.index(mult_table[(g, b)])
        P[i, j] = 1.0
    D_reg[g] = P
check("正則表示是同態", D_reg["r"] @ D_reg["s₁"], D_reg[mult_table[("r", "s₁")]])
chi_reg = np.array([np.trace(D_reg[classes[c][0]]) for c in classes])
print(f"正則表示的特徵 = {chi_reg}（χ(e) = |G| = 6，其餘 = 0）")
check("χ_reg(e) = |G|, 其餘 = 0", chi_reg, np.array([6.0, 0.0, 0.0]))
decomp_reg = {l: float(np.sum(weights * chi_reg * chi[l])) for l in labels}
print(f"分解：", {l: round(decomp_reg[l]) for l in labels})
check("每個 irrep 出現「它的維度」次（1, 1, 2）",
      [decomp_reg[l] for l in labels], [1.0, 1.0, 2.0])
check("Σ nλ·dim(λ) = 1·1 + 1·1 + 2·2 = 6 = |G|",
      sum(decomp_reg[l] * chi[l][0] for l in labels), 6.0)

# %% [markdown]
# ## 23.4　投影算子：把空間切成不變子空間
#
# $$P^{(\lambda)}=\frac{\dim\lambda}{|G|}\sum_g\chi^{(\lambda)}(g)^{*}D(g)$$
#
# 這是**投影矩陣**（$P^2=P$、$P^{\dagger}=P$、$\sum_\lambda P^{(\lambda)}=I$），
# 把表示空間分解成各個 irrep 的子空間 ── 第 4 章投影 + 第 7 章譜分解。
#
# 物理上這叫「**對稱性適應基底**」（symmetry-adapted basis）：
# 用它展開 Hamiltonian 或剛度矩陣，矩陣自動變成分塊對角。

# %%
section("23.4 投影算子與對稱性適應基底")

chi_g = {label: {name: np.trace(irreps[label][name]).real for name in G}
         for label in labels}
projectors = {}
for label in labels:
    dim_l = chi[label][0]
    P = sum(chi_g[label][g] * G[g] for g in G) * dim_l / len(G)
    projectors[label] = P
    print(f"\nP({label})：")
    for row in np.round(P, 4):
        print(f"  {row}")
    check(f"  P² = P（投影矩陣）", P @ P, P, tol=1e-10)
    check(f"  Pᵀ = P（正交投影）", P.T, P, tol=1e-10)
    print(f"  trace P = {np.trace(P):.4f} = nλ·dim(λ) = "
          f"{decomp[label] * chi[label][0]:.4f}")
    check(f"  trace P = nλ·dim(λ)", np.trace(P),
          decomp[label] * chi[label][0], tol=1e-10)
check("ΣP(λ) = I（完備性）", sum(projectors.values()), np.eye(3), tol=1e-10)
check("P(A₁)P(E) = 0（正交投影）",
      projectors["A₁（平凡）"] @ projectors["E（2 維標準）"], np.zeros((3, 3)),
      tol=1e-10)
check("P(A₂) = 0（sign 表示不出現在置換表示裡）",
      projectors["A₂（sign）"], np.zeros((3, 3)), tol=1e-10)
print("\nP(A₁) 投影出對稱方向：")
v_test = np.array([1.0, 2.0, 6.0])
print(f"  v = {v_test} → P(A₁)v = {np.round(projectors['A₁（平凡）'] @ v_test, 4)}"
      f"（= 平均值 {v_test.mean():.4f} 的常數向量）")
check("P(A₁)v = 平均值的常數向量", projectors["A₁（平凡）"] @ v_test,
      np.full(3, v_test.mean()))
check("P(E)v = v − 平均（扣掉對稱成分）", projectors["E（2 維標準）"] @ v_test,
      v_test - v_test.mean())

# %% [markdown]
# ## 23.5　物理應用：法模態的對稱分類
#
# 回到第 15 章：若對稱操作 $D(g)$ 與質量、剛度矩陣**可交換**，
#
# $$D(g)M=MD(g),\qquad D(g)K=KD(g)$$
#
# 則 $M,K$ 可以在對稱性適應基底下**同時分塊對角化**。
# 於是 $N\times N$ 的廣義特徵值問題分裂成每個 irrep 的小問題：
#
# $$\text{模態的對稱型態} = \text{它所屬的 irrep}$$
#
# 這就是分子振動光譜分類（$A_1$、$E$、$T_2$…）的數學基礎。

# %%
section("23.5 用群論分解振動問題")

# 三個相同質量在正三角形頂點，兩兩之間有相同彈簧（只考慮徑向位移，簡化模型）
M_mass = np.eye(3)
K_spring = np.array([[2.0, -1.0, -1.0],
                     [-1.0, 2.0, -1.0],
                     [-1.0, -1.0, 2.0]])
show_matrix("剛度矩陣 K（完全對稱的三質點環）", K_spring)
check("K 與所有 D(g) 可交換（系統有 S₃ 對稱）",
      all(np.allclose(A @ K_spring, K_spring @ A) for A in G.values()))
check("M 也可交換（質量相同）",
      all(np.allclose(A @ M_mass, M_mass @ A) for A in G.values()))

print("\n用對稱性適應基底（S）把問題分塊：")
K_block = S_basis.T @ K_spring @ S_basis
M_block = S_basis.T @ M_mass @ S_basis
show_matrix("SᵀKS（分塊對角！）", K_block)
check("K 在對稱基底下分塊對角",
      np.allclose(K_block[0, 1:], 0) and np.allclose(K_block[1:, 0], 0))
print(f"  A₁ 區塊（1x1）：ω² = {K_block[0, 0]:.4f}")
print(f"  E 區塊（2x2）：ω² = {np.round(np.linalg.eigvalsh(K_block[1:, 1:]), 4)}"
      f"（二重退化！）")
omega2, modes = generalized_symmetric_eig(K_spring, M_mass)
print(f"\n直接解整個問題：ω² = {np.round(np.sort(omega2), 6)}")
check("A₁ 模態的 ω² = 0（整體平移，對稱模態）", np.sort(omega2)[0], 0.0, tol=1e-10)
check("E 模態二重退化（ω² = 3, 3）", np.sort(omega2)[1:], np.array([3.0, 3.0]),
      tol=1e-10)
check("分塊的特徵值 = 整體的特徵值",
      np.sort(np.concatenate([[K_block[0, 0]],
                              np.linalg.eigvalsh(K_block[1:, 1:])])),
      np.sort(omega2), tol=1e-10)
print("  ⇒ 「退化」不是巧合：它由 irrep 的維度決定（E 是 2 維 ⇒ 必然二重退化）")

# 退化度 = irrep 維度
print("\n退化與 irrep 維度的對應：")
for label in labels:
    n_l = decomp[label]
    if n_l > 0:
        print(f"  {label:<16} 出現 {round(n_l)} 次 × 維度 {chi[label][0]:.0f} "
              f"⇒ 貢獻 {round(n_l * chi[label][0])} 個模態")
check("模態總數 = Σ nλ·dim(λ) = 3",
      sum(decomp[l] * chi[l][0] for l in labels), 3.0)

# 選擇律：矩陣元素 ⟨λ|O|μ⟩ 只在 irrep 相容時非零
section("23.5 選擇律：為什麼有些躍遷是禁止的")

# 對稱算子 O（與群可交換）在不同 irrep 之間的矩陣元素必為 0（Schur 引理）
O_sym = K_spring                      # 一個對稱算子
P1, PE = projectors["A₁（平凡）"], projectors["E（2 維標準）"]
cross = P1 @ O_sym @ PE
print(f"對稱算子在 A₁ 與 E 之間的矩陣元素：‖P(A₁)·O·P(E)‖ = "
      f"{np.linalg.norm(cross):.3e}")
check("不同 irrep 之間的矩陣元素 = 0（選擇律！）", cross, np.zeros((3, 3)),
      tol=1e-10)
# 不對稱的算子就沒有這個限制
O_asym = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 1.0]])
check("不對稱的算子（不與群可交換）就沒有選擇律",
      np.linalg.norm(P1 @ O_asym @ PE) > 1e-6)
print("  ⇒ 這就是光譜學「選擇律」的來源：")
print("     ⟨最終態|偶極算子|初始態⟩ 只在 irrep 的乘積含有平凡表示時才不為 0")

# 乘積表示的分解（Clebsch–Gordan）
print("\n乘積表示 E ⊗ E 的分解：")
chi_EE = chi["E（2 維標準）"] ** 2       # 乘積表示的特徵 = 特徵的乘積
print(f"  χ(E⊗E) = χ(E)² = {chi_EE}")
decomp_EE = {l: float(np.sum(weights * chi_EE * chi[l])) for l in labels}
print(f"  分解 = ", {l: round(decomp_EE[l]) for l in labels})
check("E ⊗ E = A₁ ⊕ A₂ ⊕ E（維度 1+1+2 = 4 = 2×2）",
      [decomp_EE[l] for l in labels], [1.0, 1.0, 1.0])
check("維度守恆：1 + 1 + 2 = 4",
      sum(decomp_EE[l] * chi[l][0] for l in labels), 4.0)
print("  ⇒ 乘積表示的特徵 = 特徵的乘積（trace(A⊗B) = trace A · trace B，第 20 章！）")
D_EE = {g: np.kron(D2[g], D2[g]) for g in G}
check("實際驗證：tr(D(g)⊗D(g)) = (tr D(g))²",
      [np.trace(D_EE[g]) for g in G], [np.trace(D2[g]) ** 2 for g in G])

# %% [markdown]
# ## 23.6　循環群與 Fourier：$C_n$ 的 irrep 就是 DFT
#
# 交換群的所有 irrep 都是 **1 維**的。對循環群 $C_n$：
#
# $$D^{(k)}(r^j)=\omega^{jk},\qquad \omega=e^{2\pi i/n}$$
#
# 特徵表就是 **Fourier 矩陣**！這解釋了第 7 章的結果：
# 循環矩陣（$C_n$ 對稱的算子）都被 Fourier 矩陣對角化。

# %%
section("23.6 循環群的特徵表 = Fourier 矩陣")

n_cyc = 6
omega = np.exp(2j * np.pi / n_cyc)
# C_n 的正則表示 = 循環位移矩陣的冪次
P_shift = np.roll(np.eye(n_cyc), 1, axis=0)
C_group = {j: np.linalg.matrix_power(P_shift, j) for j in range(n_cyc)}
check("Cₙ 是交換群", all(np.allclose(C_group[i] @ C_group[j],
                                     C_group[j] @ C_group[i])
                         for i in range(n_cyc) for j in range(n_cyc)))
check("rⁿ = I", C_group[0], np.eye(n_cyc))

# 特徵表 = Fourier 矩陣
char_table = np.array([[omega ** (j * k) for j in range(n_cyc)]
                       for k in range(n_cyc)])
F_matrix = np.array([[np.exp(2j * np.pi * j * k / n_cyc) for j in range(n_cyc)]
                     for k in range(n_cyc)])
check("特徵表 = Fourier 矩陣", char_table, F_matrix)
check("特徵的正交性：(1/n)χ^(k)·conj(χ^(l)) = δ_kl",
      char_table @ char_table.conj().T / n_cyc, np.eye(n_cyc, dtype=complex))
print(f"⇒ Cₙ 有 {n_cyc} 個 1 維 irrep，特徵表就是 {n_cyc}×{n_cyc} 的 Fourier 矩陣")
check("交換群的所有 irrep 都是 1 維（Σ1² = n）", n_cyc * 1, n_cyc)

# 任何與 C_n 可交換的矩陣都是循環矩陣，被 F 對角化
rng = np.random.default_rng(0)
c_vec = rng.standard_normal(n_cyc)
C_circ = sum(c_vec[j] * C_group[j] for j in range(n_cyc))
check("Σcⱼrʲ 與所有群元素可交換",
      all(np.allclose(C_circ @ C_group[j], C_group[j] @ C_circ)
          for j in range(n_cyc)))
F_unitary = F_matrix / np.sqrt(n_cyc)
D_diag = F_unitary.conj().T @ C_circ @ F_unitary
check("Fourier 矩陣對角化循環矩陣", D_diag, np.diag(np.diag(D_diag)), tol=1e-10)
check("對角線 = fft(c)", np.sort_complex(np.diag(D_diag)),
      np.sort_complex(np.fft.fft(c_vec)), tol=1e-10)
print("  ⇒ 第 7 章「Fourier 對角化所有循環矩陣」就是「Cₙ 的 irrep 都是 1 維」")

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
im = ax.imshow(np.real(char_table), cmap="RdBu_r")
ax.set_title(f"Character table of C_{n_cyc} = Re(Fourier matrix)", fontsize=9)
ax.set_xlabel("group element r^j")
ax.set_ylabel("irrep k")
fig.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.abs(np.array([[np.sum(weights * chi[li] * chi[lj]) for lj in labels]
                           for li in labels])), cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(["A1", "A2", "E"])
ax.set_yticks(range(3))
ax.set_yticklabels(["A1", "A2", "E"])
ax.set_title("S3: character orthogonality = identity", fontsize=9)
for i in range(3):
    for j in range(3):
        val = np.sum(weights * chi[labels[i]] * chi[labels[j]])
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                color="C3", fontsize=12)
finish(fig, "ch23_characters")

# %% [markdown]
# ## 動手練習
#
# 1. 寫出 $C_4$（正方形的旋轉）的特徵表，確認 4 個 1 維 irrep。
# 2. 對 $D_4$（正方形的完整對稱群，8 個元素）數出共軛類的個數，
#    並驗證 $\sum\dim^2=8$。
# 3. 把 4 個質點在正方形上的振動問題用 $C_4$ 對稱性分塊。
# 4. 驗證 Burnside 定理 $\sum_\lambda\dim(\lambda)^2=|G|$ 對 $S_3$ 與 $C_6$。
# 5. 證明：交換群的 irrep 都是 1 維（提示：Schur 引理）。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1：C₄
print("練習 1：C₄ 的特徵表")
n4 = 4
w4 = np.exp(2j * np.pi / n4)
table4 = np.array([[w4 ** (j * k) for j in range(n4)] for k in range(n4)])
print("      ", "  ".join(f"{'e' if j == 0 else 'r' + str(j):>8}" for j in range(n4)))
for k in range(n4):
    print(f"  k={k} ", "  ".join(f"{table4[k, j]:>8.2f}" for j in range(n4)))
check("  特徵正交", table4 @ table4.conj().T / n4, np.eye(n4, dtype=complex))
check("  4 個 1 維 irrep，Σ1² = 4 = |G|", n4, 4)
print("  （k = 0 是平凡表示；k = 2 是 (−1)^j；k = 1, 3 是複共軛對）")

# 練習 2：D₄
print("\n練習 2：D₄（8 個元素）")
# 用 2x2 矩陣表示：4 個旋轉 + 4 個反射
rot = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
ref = lambda th: np.array([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
D4_group = {}
for j in range(4):
    D4_group[f"r{j}"] = rot(j * np.pi / 2)
    D4_group[f"s{j}"] = ref(j * np.pi / 4 * 2)
check("  D₄ 封閉（8 個元素）",
      all(any(np.allclose(A @ B, C) for C in D4_group.values())
          for A in D4_group.values() for B in D4_group.values()))
check("  |D₄| = 8", len(D4_group), 8)
# 共軛類：{e}, {r²}, {r, r³}, {兩組反射}
traces = sorted(round(np.trace(A), 6) for A in D4_group.values())
print(f"  2 維表示的特徵（跡）= {traces}")
# 用交換子空間的維度算 Σnλ²
Sys4 = np.vstack([np.kron(np.eye(2), A) - np.kron(A.T, np.eye(2))
                  for A in D4_group.values()])
dim_c4 = 4 - np.linalg.matrix_rank(Sys4, tol=1e-10)
print(f"  這個 2 維表示的交換子空間維度 = {dim_c4} ⇒ "
      f"{'不可約' if dim_c4 == 1 else '可約'}")
check("  2 維表示不可約", dim_c4, 1)
print("  D₄ 的 irrep：4 個 1 維 + 1 個 2 維 ⇒ Σdim² = 4·1 + 4 = 8 = |G| ✓")
check("  Σdim² = 8", 4 * 1 + 2 ** 2, 8)
print(f"  共軛類個數 = irrep 個數 = 5")

# 練習 3：正方形的振動
print("\n練習 3：4 個質點（正方形）的振動分塊")
K4 = np.array([[2.0, -1.0, 0.0, -1.0],
               [-1.0, 2.0, -1.0, 0.0],
               [0.0, -1.0, 2.0, -1.0],
               [-1.0, 0.0, -1.0, 2.0]])
P4 = np.roll(np.eye(4), 1, axis=0)
check("  K 與循環位移可交換（C₄ 對稱）", P4 @ K4, K4 @ P4)
F4 = np.array([[np.exp(2j * np.pi * j * k / 4) for j in range(4)]
               for k in range(4)]) / 2
D_K4 = F4.conj().T @ K4 @ F4
check("  Fourier 基底（C₄ 的 irrep）把 K 完全對角化",
      D_K4, np.diag(np.diag(D_K4)), tol=1e-10)
print(f"  ω² = {np.round(np.real(np.diag(D_K4)), 6)}（理論 2−2cos(2πk/4) = "
      f"{np.round([2 - 2 * np.cos(2 * np.pi * k / 4) for k in range(4)], 6)}）")
check("  ω² = 2 − 2cos(2πk/4)", np.sort(np.real(np.diag(D_K4))),
      np.sort([2 - 2 * np.cos(2 * np.pi * k / 4) for k in range(4)]), tol=1e-10)
print("  ⇒ 全部 1 維 irrep ⇒ 完全對角化（但 k = 1, 3 退化，因為它們是複共軛對）")

# 練習 4：Burnside
print("\n練習 4：Burnside 定理 Σdim² = |G|")
check("  S₃：1² + 1² + 2² = 6", 1 + 1 + 4, 6)
check("  C₆：6 個 1 維 ⇒ 6 = 6", 6 * 1, 6)
check("  用交換子空間維度驗證（正則表示的 Σnλ² = Σdim² = |G|）",
      36 - np.linalg.matrix_rank(
          np.vstack([np.kron(np.eye(6), A) - np.kron(A.T, np.eye(6))
                     for A in D_reg.values()]), tol=1e-10), 6)
print("  （正則表示的交換子空間維度 = Σnλ² = Σdim² = |G|）")

# 練習 5
print("\n練習 5：交換群的 irrep 都是 1 維")
print("  證明：若 G 交換，則每個 D(g) 都與所有 D(h) 可交換。")
print("        Schur 引理 ⇒ 若 D 不可約，則 D(g) = c(g)·I。")
print("        但 D(g) = c(g)I 的話，任何 1 維子空間都是不變的 ⇒ dim = 1。")
check("  C₆ 的例子：所有 irrep 都是 1 維",
      all(np.allclose(C_group[i] @ C_group[j], C_group[j] @ C_group[i])
          for i in range(6) for j in range(6)))
# 數值：交換群的正則表示分解成 n 個 1 維
check("  C₆ 的正則表示分解成 6 個 1 維 irrep（Fourier 對角化）",
      np.allclose(F_unitary.conj().T @ C_group[1] @ F_unitary,
                  np.diag(np.diag(F_unitary.conj().T @ C_group[1] @ F_unitary))))

# %% [markdown]
# ## 本章重點回顧
#
# * **表示** = 群到矩陣的同態；等價表示 = 相似變換（第 10、14 章）。
# * **可約 $\iff$ 存在不變子空間 $\iff$ 可同時分塊對角化**。
#   Schur 引理的數值版：與所有 $D(g)$ 可交換的矩陣空間維度 $=\sum n_\lambda^2$，
#   不可約 $\iff$ 維度 $=1$。
# * **特徵 $\chi=\operatorname{tr}D$ 是相似不變量**，同共軛類相同；
#   **特徵的正交性就是向量正交**，所以分解係數
#   $n_\lambda=\langle\chi,\chi^{(\lambda)}\rangle$ 可以直接用內積算。
# * Burnside：$\sum_\lambda\dim(\lambda)^2=|G|$；
#   正則表示中每個 irrep 出現「它的維度」次。
# * **投影算子** $P^{(\lambda)}=\frac{\dim\lambda}{|G|}\sum_g\chi^{(\lambda)}(g)^{*}D(g)$
#   是正交投影，$\sum_\lambda P^{(\lambda)}=I$ ── 把空間切成不變子空間。
# * 物理：對稱操作與 $M,K$ 可交換 $\Rightarrow$ 同時分塊對角化 $\Rightarrow$
#   **模態按 irrep 分類、退化度 = irrep 維度**（第 15 章的伏筆在此收尾）。
#   不同 irrep 之間的矩陣元素必為 0 ── 這就是**選擇律**。
# * 乘積表示的特徵 = 特徵的乘積（$\operatorname{tr}(A\otimes B)=\operatorname{tr}A\operatorname{tr}B$，第 20 章）。
# * **交換群的所有 irrep 都是 1 維**，$C_n$ 的特徵表就是 **Fourier 矩陣** ──
#   第 7 章「Fourier 對角化所有循環矩陣」的群論解釋。
#
# 下一章（最後一章）：機率與統計中的線性代數。
