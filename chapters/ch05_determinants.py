# %% [markdown]
# # 第 5 章　行列式：面積、體積與可逆性
#
# > 對應 Strang 第 5 章（5.1 3×3 行列式與餘因子、5.2 計算與應用、5.3 面積與體積），
# > 以及 Riley 8.9（行列式）、6.4（多重積分的 Jacobian）、7.6（叉積）。
#
# 行列式把整個矩陣壓縮成**一個數字**，卻同時回答三件事：
#
# | $\det A$ | 代數意義 | 幾何意義 |
# |---|---|---|
# | $=0$ | 欄相依、不可逆 | 盒子被壓扁、體積 0 |
# | $\ne 0$ | 可逆，$A^{-1}=C^{\mathsf T}/\det A$ | 體積 $=|\det A|$ |
# | 正負號 | 置換的奇偶 | 是否翻轉定向 |
#
# 三種算法，代價天差地別：
#
# | 方法 | 成本 | 適用 |
# |---|---|---|
# | 大公式（$n!$ 項）| $O(n!\,n)$ | 理論、$n\le3$ |
# | 餘因子展開 | $O(n!)$ | 稀疏、小矩陣、符號運算 |
# | $PA=LU$ 乘主元 | $O(n^3/3)$ | **實務唯一選擇** |
#
# ```bash
# python chapters/ch05_determinants.py
# python tools/build_notebooks.py ch05
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

from linalg_tutorial.determinant import (
    big_formula_terms,
    cofactor_matrix,
    cramer,
    det_cofactor,
    det_lu,
    inverse_by_cofactors,
    minor,
    volume,
)
from linalg_tutorial.elimination import plu, second_difference_matrix
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 5.1　從三條性質長出整個行列式
#
# 整套理論只需要三條公設：
#
# 1. $\det I=1$
# 2. **交換兩列 $\Rightarrow$ 變號**
# 3. $\det$ 對**每一列分別線性**
#
# 由此立刻推出：
#
# * 兩列相同 $\Rightarrow\det=0$（交換後不變又該變號）
# * 「某列減去另一列的倍數」**不改變** $\det$ → 消去法可以放心用
# * 三角矩陣的 $\det$ = 對角線相乘
# * $\det A=\pm(\text{主元相乘})$

# %%
section("5.1 三條性質與推論")

A = np.array([[3.0, 1.0, 0.0], [-3.0, 2.0, 1.0], [6.0, 8.0, 4.0]])
check("det I = 1", det_lu(np.eye(4)), 1.0)

swapped = A[[1, 0, 2], :]
check("交換兩列 ⇒ 變號", det_lu(swapped), -det_lu(A))

scaled = A.copy()
scaled[0, :] *= 5
check("某列乘 5 ⇒ det 乘 5（逐列線性）", det_lu(scaled), 5 * det_lu(A))

equal_rows = A.copy()
equal_rows[2, :] = equal_rows[0, :]
check("兩列相同 ⇒ det = 0", det_lu(equal_rows), 0.0)

eliminated = A.copy()
eliminated[1, :] += 3.7 * eliminated[0, :]
check("列減（加）另一列的倍數 ⇒ det 不變", det_lu(eliminated), det_lu(A))

tri = np.triu(A)
check("三角矩陣 det = 對角線相乘", det_lu(tri), float(np.prod(np.diag(tri))))

P, L, U = plu(A)
print(f"\nPA = LU 的主元 = {np.diag(U)}")
print(f"主元相乘 = {np.prod(np.diag(U)):.6g}，det P = {np.linalg.det(P):+.0f}")
check("det A = det(P)⁻¹ · 主元相乘", det_lu(A), np.linalg.det(P) * np.prod(np.diag(U)))
check("det A 與 numpy 一致", det_lu(A), float(np.linalg.det(A)))

# %% [markdown]
# ### 大公式：$n!$ 條「路徑」
#
# $$\det A=\sum_{\sigma}\operatorname{sign}(\sigma)\,
# a_{1\sigma(1)}a_{2\sigma(2)}\cdots a_{n\sigma(n)}$$
#
# 每一項從每列、每欄各取一個元素（一條「下行路徑」），符號由置換的奇偶決定。
# $3\times3$ 有 6 項（就是那個「對角線相乘」口訣的來源），$4\times4$ 有 24 項。

# %%
section("5.1 大公式的每一項")

A3 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
show_matrix("A（−1, 2, −1 三對角矩陣）", A3)
terms = big_formula_terms(A3)
total = 0.0
print(f"{'置換 σ':<14} | {'sign':>5} | {'乘積':>8} | {'貢獻':>8}")
print("-" * 46)
for perm, sign, prod in terms:
    contrib = sign * prod
    total += contrib
    mark = "" if prod != 0 else "  (有 0 元素 → 整項為 0)"
    print(f"{str(tuple(p + 1 for p in perm)):<14} | {sign:>+5.0f} | {prod:>8.4g} "
          f"| {contrib:>8.4g}{mark}")
print(f"\n總和 = {total:.6g}")
check("大公式 = det A", total, det_lu(A3))
print(f"項數 = {len(terms)} = 3!，其中非零項只有 "
      f"{sum(1 for _, _, p in terms if p != 0)} 項")

# %% [markdown]
# ### 餘因子展開與 $A^{-1}=C^{\mathsf T}/\det A$
#
# $$C_{ij}=(-1)^{i+j}\det M_{ij},\qquad
# \det A=\sum_j a_{ij}C_{ij}\quad(\text{沿任一列展開})$$
#
# 最漂亮的結果：$AC^{\mathsf T}=(\det A)I$，於是
#
# $$\boxed{A^{-1}=\frac{C^{\mathsf T}}{\det A}}$$
#
# 這解釋了為什麼 $A^{-1}$ 的每個元素都是「兩個行列式的比值」，
# 也解釋了為什麼 $\det A=0$ 時反矩陣不存在。

# %%
section("5.1 餘因子與反矩陣公式")

A4 = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
show_matrix("A", A4)
C = cofactor_matrix(A4)
show_matrix("餘因子矩陣 C", C)
show_matrix("Cᵀ（= adjugate，此例 det A = 1 故等於 A⁻¹）", C.T)
check("A·Cᵀ = (det A)·I", A4 @ C.T, det_lu(A4) * np.eye(3))
check("A⁻¹ = Cᵀ/det A", inverse_by_cofactors(A4), np.linalg.inv(A4))

print("\n沿不同列展開都得到同一個 det：")
for i in range(3):
    s = sum(A4[i, j] * C[i, j] for j in range(3))
    print(f"  沿第 {i + 1} 列展開 = {s:.6g}")
    check(f"  = det A", s, det_lu(A4))
print(f"\n子矩陣 M₁₂（去掉第 1 列第 2 欄）=\n{minor(A4, 0, 1)}")

# %% [markdown]
# ### 遞迴之美：$-1,2,-1$ 三對角矩陣
#
# 沿第一列做餘因子展開，得到漂亮的遞迴
#
# $$\det K_n=2\det K_{n-1}-\det K_{n-2}
# \quad\Longrightarrow\quad \det K_n=n+1$$
#
# 這個 $K$ 就是第 2 章的二階差分矩陣。
# 行列式永不為零 ⇒ 永遠可逆 ⇒ 離散 Poisson 方程永遠有唯一解。

# %%
section("5.1 二階差分矩陣的行列式 = n + 1")

print(f"{'n':>4} | {'det Kₙ (餘因子)':>16} | {'det Kₙ (LU)':>14} | {'n + 1':>7} | 遞迴驗證")
print("-" * 68)
dets = {}
for n in range(1, 9):
    K = second_difference_matrix(n)
    d_cof = det_cofactor(K) if n <= 6 else float("nan")
    d_lu = det_lu(K)
    dets[n] = d_lu
    rec = ""
    if n >= 3:
        pred = 2 * dets[n - 1] - dets[n - 2]
        rec = f"2·{dets[n-1]:.0f} − {dets[n-2]:.0f} = {pred:.0f} ✓" \
            if abs(pred - d_lu) < 1e-8 else "✗"
    print(f"{n:>4} | {d_cof:>16.6g} | {d_lu:>14.6g} | {n + 1:>7} | {rec}")
check("det Kₙ = n + 1", [dets[n] for n in range(1, 9)],
      [float(n + 1) for n in range(1, 9)], tol=1e-8)
print("\n推論：Kₙ 永遠可逆 ⇒ 固定-固定邊界的離散 Poisson 方程唯一可解")
B = second_difference_matrix(5)
B[0, 0] = B[-1, -1] = 1.0            # 自由-自由
check("自由-自由矩陣 B 的 det = 0（奇異）", det_lu(B), 0.0)

# %% [markdown]
# ## 5.2　行列式的運算律與 Cramer 法則
#
# $$\det A^{\mathsf T}=\det A,\qquad
# \det(AB)=(\det A)(\det B),\qquad
# |\det Q|=1\ (\text{正交矩陣})$$
#
# 注意：$\det(A+B)\ne\det A+\det B$！（看 $I+I$：$2^n\ne 2$）
#
# **Cramer 法則**：$x_j=\dfrac{\det B_j}{\det A}$，$B_j$ 是把 $A$ 的第 $j$ 欄換成 $\mathbf b$。
# 理論漂亮、計算災難（$(n+1)!$ 項）。

# %%
section("5.2 運算律")

rng = np.random.default_rng(0)
A5 = rng.standard_normal((5, 5))
B5 = rng.standard_normal((5, 5))
check("det Aᵀ = det A", det_lu(A5.T), det_lu(A5), tol=1e-8)
check("det(AB) = det A · det B", det_lu(A5 @ B5), det_lu(A5) * det_lu(B5), tol=1e-7)
check("det(A⁻¹) = 1/det A", det_lu(np.linalg.inv(A5)), 1 / det_lu(A5), tol=1e-8)
check("det(cA) = cⁿ det A（n = 5）", det_lu(3 * A5), 3 ** 5 * det_lu(A5), tol=1e-6)
check("det(A + B) ≠ det A + det B",
      not np.isclose(det_lu(A5 + B5), det_lu(A5) + det_lu(B5)))
print(f"  反例：det(I + I) = {det_lu(2 * np.eye(5)):.0f} ≠ "
      f"det I + det I = 2")

Q, _ = np.linalg.qr(A5)
check("|det Q| = 1（正交矩陣）", abs(det_lu(Q)), 1.0, tol=1e-8)
Pm = np.eye(4)[[1, 2, 3, 0], :]
check("置換矩陣 det = ±1", abs(det_lu(Pm)), 1.0)
print(f"  循環置換 (1234)→(2341) 的 det = {det_lu(Pm):+.0f}"
      f"（3 次對換 ⇒ 奇置換）")

# %%
section("5.2 Cramer 法則（以及為什麼不要用它）")

A6 = np.array([[3.0, 4.0], [5.0, 6.0]])
b6 = np.array([2.0, 4.0])
x_cramer = cramer(A6, b6)
print(f"det A = {det_lu(A6):.0f}")
for j in range(2):
    Bj = A6.copy()
    Bj[:, j] = b6
    print(f"  B{j + 1}（第 {j + 1} 欄換成 b）的 det = {det_lu(Bj):.0f}"
          f"  →  x{j + 1} = {det_lu(Bj) / det_lu(A6):.4g}")
check("Cramer 解正確", A6 @ x_cramer, b6)
check("與 numpy.linalg.solve 一致", x_cramer, np.linalg.solve(A6, b6))

print(f"\n{'n':>4} | {'餘因子展開 (s)':>16} | {'LU 法 (s)':>12} | {'倍數':>10}")
print("-" * 52)
for n in [5, 7, 9]:
    M = rng.standard_normal((n, n))
    t0 = time.perf_counter()
    det_cofactor(M)
    t_cof = time.perf_counter() - t0
    t0 = time.perf_counter()
    det_lu(M)
    t_lu = time.perf_counter() - t0
    print(f"{n:>4} | {t_cof:>16.5f} | {t_lu:>12.5f} | {t_cof / t_lu:>9.0f}x")
print("  餘因子展開是 O(n!)；LU 是 O(n³/3)。n = 20 用大公式要算 2.4×10¹⁸ 項。")

# %% [markdown]
# ## 5.3　面積、體積與 $|\det|$ 作為「放大率」
#
# 以 $A$ 的欄為邊的平行多面體：
#
# $$\text{體積}=|\det A|$$
#
# 證明思路（Strang 的漂亮論證）：$A=QR$。$Q$ 是旋轉／反射，**不改變體積**；
# $R$ 是三角矩陣，對角線剛好就是「每個新維度的高」，相乘即體積。
# 於是 $\text{體積}=|\det R|=|\det A|$。
#
# 更一般地：$A$ 把任何形狀的體積乘上 $|\det A|$ ──
# 這正是多重積分換變數時 **Jacobian** 的來源（Riley 6.4）。

# %%
section("5.3 面積 = |ad − bc|")

e1, e2 = np.array([3.0, 1.0]), np.array([1.0, 2.0])
E = np.column_stack([e1, e2])
area = volume(E)
print(f"兩邊 = {e1}, {e2}")
print(f"|det E| = {area:.4g}")
# 用「長方形減六塊」的初等幾何驗證（Strang 圖 5.1）
a, b = e1
c, d = e2
geo = (a + c) * (b + d) - 2 * b * c - a * b - c * d
check("幾何算法（長方形減六塊）= |ad − bc|", geo, area)
# 用叉積驗證
check("= 叉積的 z 分量大小（把兩邊補成 3D）",
      abs(np.cross(np.append(e1, 0.0), np.append(e2, 0.0))[2]), area)
# 用蒙地卡羅驗證
rng = np.random.default_rng(1)
pts = rng.uniform(0, 1, size=(400000, 2))        # 單位正方形取樣
mapped = pts @ E.T                               # 映射後的平行四邊形
hull_area_mc = area                              # 理論值
# 反向檢驗：落在平行四邊形內的比例 × 外接矩形面積
box = np.array([[0, 0], [a, b], [c, d], [a + c, b + d]])
lo, hi = box.min(axis=0), box.max(axis=0)
samp = rng.uniform(lo, hi, size=(400000, 2))
coef = np.linalg.solve(E, samp.T).T
inside = np.all((coef >= 0) & (coef <= 1), axis=1)
mc = inside.mean() * np.prod(hi - lo)
print(f"蒙地卡羅估計面積 = {mc:.4f}（理論 {area:.4f}）")
check("蒙地卡羅誤差 < 1%", abs(mc - area) / area < 0.01)
print(f"三角形面積 = |det|/2 = {area / 2:.4g}")

fig, ax = new_axes("Area of the parallelogram = |det E|", figsize=(5.2, 5.0))
poly = np.array([[0, 0], e1, e1 + e2, e2, [0, 0]])
ax.fill(poly[:, 0], poly[:, 1], alpha=0.25, color="C0")
ax.plot(poly[:, 0], poly[:, 1], "C0", lw=2)
ax.annotate("", xy=e1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C3", lw=2))
ax.annotate("", xy=e2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C2", lw=2))
ax.text(*(e1 * 0.55), " e1", color="C3")
ax.text(*(e2 * 0.55), " e2", color="C2")
ax.text(1.4, 1.6, f"|det| = {area:.0f}", fontsize=12)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 3.5)
finish(fig, "ch05_parallelogram_area")

# %%
section("5.3 體積與 |det| 的放大率")

E3 = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
vol = volume(E3)
print(f"三維盒子體積 = |det E| = {vol:.4g}")
Q, R = np.linalg.qr(E3)
print(f"E = QR：|det Q| = {abs(np.linalg.det(Q)):.4f}（旋轉不改變體積）")
print(f"R 的對角線（每個新維度的高）= {np.abs(np.diag(R))}")
check("|det R| = 對角線相乘 = 體積", abs(np.prod(np.diag(R))), vol, tol=1e-8)

# 放大率：隨機點雲經過 A 之後的「體積」比
rng = np.random.default_rng(2)
cloud = rng.uniform(-1, 1, size=(200000, 3))
unit_ball = np.sum(cloud ** 2, axis=1) <= 1.0
mapped = cloud[unit_ball] @ E3.T
# 球映射成橢球，體積比應為 |det|
print(f"\n單位球經 E 映射成橢球：")
print(f"  理論體積比 = |det E| = {vol:.4f}")
sing = np.linalg.svd(E3, compute_uv=False)
print(f"  橢球半軸 = 奇異值 = {sing}（乘積 = {np.prod(sing):.4f}）")
check("奇異值相乘 = |det|", np.prod(sing), vol, tol=1e-8)

# 與 Jacobian 的連結：極座標
print("\n換變數的 Jacobian（Riley 6.4）：極座標 x = r cosθ, y = r sinθ")
for r0, th0 in [(1.0, 0.3), (2.0, 1.1)]:
    J = np.array([[np.cos(th0), -r0 * np.sin(th0)],
                  [np.sin(th0), r0 * np.cos(th0)]])
    print(f"  r = {r0}, θ = {th0}: det J = {det_lu(J):.6f}（應等於 r）")
    check(f"  det J = r", det_lu(J), r0)
print("  所以 dx dy = r dr dθ —— 這就是行列式在積分裡的身分")

# 定向：det < 0 代表翻轉
Ref = np.array([[1.0, 0.0], [0.0, -1.0]])
print(f"\n反射矩陣 det = {det_lu(Ref):+.0f} < 0 ⇒ 翻轉定向（左手變右手）")
Rot = np.array([[0.0, -1.0], [1.0, 0.0]])
print(f"旋轉矩陣 det = {det_lu(Rot):+.0f} > 0 ⇒ 保持定向")

# %% [markdown]
# ### 行列式與特徵值的橋樑（預告第 6 章）
#
# $$\det(A-\lambda I)=0$$
#
# 這條**特徵方程**正是第 6 章的起點：$A-\lambda I$ 奇異，才會有非零的 $\mathbf x$
# 使 $A\mathbf x=\lambda\mathbf x$。而且
#
# $$\det A=\lambda_1\lambda_2\cdots\lambda_n,\qquad
# \operatorname{trace}A=\lambda_1+\dots+\lambda_n$$

# %%
section("5.3 特徵方程與 det = λ 的乘積")

S = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
lam = np.linalg.eigvalsh(S)
print(f"特徵值 λ = {lam}")
check("det A = λ 的乘積", det_lu(S), float(np.prod(lam)), tol=1e-8)
check("trace A = λ 的總和", np.trace(S), float(np.sum(lam)), tol=1e-8)
print("\n驗證 det(A − λI) = 0：")
for l in lam:
    print(f"  λ = {l:.6f} → det(A − λI) = {det_lu(S - l * np.eye(3)):.3e}")
    check(f"  ≈ 0", abs(det_lu(S - l * np.eye(3))) < 1e-8)

# %% [markdown]
# ## 動手練習
#
# 1. 用三條性質證明：若 $A$ 有一整列為 0，則 $\det A=0$。
# 2. 算出 $4\times4$ 的 $-1,2,-1$ 矩陣的全部 24 項，看看有幾項非零。
# 3. 驗證 $\det\begin{bmatrix}A&B\\0&D\end{bmatrix}=\det A\cdot\det D$（分塊三角矩陣）。
# 4. 用 Vandermonde 矩陣驗證
#    $\det V=\prod_{i<j}(x_j-x_i)$，並說明為何節點重複時行列式為 0。
#
# 參考解答：

# %%
section("練習參考解答")

Zrow = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
check("練習 1：有零列 ⇒ det = 0", det_lu(Zrow), 0.0)
print("  證明：零列 = 0 × 任何列，由逐列線性得 det = 0 × det(...) = 0")

K4 = second_difference_matrix(4)
terms4 = big_formula_terms(K4)
nz = [(p, s, pr) for p, s, pr in terms4 if pr != 0]
print(f"\n練習 2：4x4 共 {len(terms4)} 項，非零的只有 {len(nz)} 項：")
for p, s, pr in nz:
    print(f"    σ = {tuple(i + 1 for i in p)}  sign = {s:+.0f}  乘積 = {pr:+.0f}")
check("  非零項加總 = det K₄ = 5", sum(s * pr for _, s, pr in nz), 5.0)

Ab = np.array([[2.0, 1.0], [0.0, 3.0]])
Bb = np.array([[5.0, 6.0], [7.0, 8.0]])
Db = np.array([[1.0, 2.0], [3.0, 7.0]])
Block = np.block([[Ab, Bb], [np.zeros((2, 2)), Db]])
check("練習 3：分塊三角 det = det A · det D",
      det_lu(Block), det_lu(Ab) * det_lu(Db), tol=1e-8)

xs = np.array([1.0, 2.0, 4.0, 7.0])
V = np.vander(xs, increasing=True)
prod = np.prod([xs[j] - xs[i] for i in range(4) for j in range(i + 1, 4)])
print(f"\n練習 4：det V = {det_lu(V):.6g}，∏(xⱼ − xᵢ) = {prod:.6g}")
check("  兩者相等", det_lu(V), float(prod), tol=1e-8)
xs_dup = np.array([1.0, 2.0, 2.0, 7.0])
check("  節點重複 ⇒ 兩列相同 ⇒ det = 0",
      abs(det_lu(np.vander(xs_dup, increasing=True))) < 1e-10)

# %% [markdown]
# ## 本章重點回顧
#
# * 行列式由三條性質唯一決定：$\det I=1$、交換列變號、逐列線性。
# * 三種公式：大公式（$n!$ 項）、餘因子展開（給出 $A^{-1}=C^{\mathsf T}/\det A$）、
#   主元相乘（實務唯一可行，$O(n^3)$）。
# * $\det A^{\mathsf T}=\det A$、$\det AB=\det A\det B$、$|\det Q|=1$；
#   但 $\det(A+B)\ne\det A+\det B$。
# * Cramer 法則給出解的封閉形式，但計算量是 $(n+1)!$ ── 只適合符號推導。
# * $|\det A|$ 是**體積放大率**：平行多面體體積、橢球半軸乘積、
#   多重積分的 Jacobian，全是同一件事。符號則代表定向是否翻轉。
# * $\det(A-\lambda I)=0$ 把我們直接帶向下一章。
#
# 下一章：特徵值與特徵向量 ── 找出矩陣只拉伸不轉向的方向。
