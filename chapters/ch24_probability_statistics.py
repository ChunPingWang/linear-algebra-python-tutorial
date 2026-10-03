# %% [markdown]
# # 第 24 章　機率與統計中的線性代數
#
# > 對應 Riley 第 30 章（機率：隨機變數、分布的性質、生成函數、聯合分布、
# > 變數變換）與第 31 章（統計：樣本統計、估計量、最大似然法、最小平方法、假設檢定），
# > 並對照本教材第 4 章（最小平方）、第 9 章（PCA）、第 12 章（共變異數矩陣）。
#
# 統計學的骨架幾乎全是線性代數：
#
# | 統計 | 線性代數 |
# |---|---|
# | 共變異數矩陣 $V$ | 對稱半正定矩陣 |
# | $\operatorname{Cov}(A\mathbf x)=AVA^{\mathsf T}$ | 二次型的變換 |
# | 多維常態的等高線 | 能量橢圓 $\mathbf x^{\mathsf T}V^{-1}\mathbf x=c$ |
# | 條件分布 | **Schur 補** |
# | 最小平方 $\hat\beta$ | 投影 $H=X(X^{\mathsf T}X)^{-1}X^{\mathsf T}$ |
# | 殘差自由度 $n-p$ | $\operatorname{trace}(I-H)$ |
# | 卡方分解 | 正交分解 $\|\mathbf y\|^2=\|H\mathbf y\|^2+\|(I-H)\mathbf y\|^2$ |
# | Fisher 資訊矩陣 | 對數似然的 Hessian |
# | Gauss–Markov 定理 | 投影是最小變異的線性無偏估計 |
#
# ```bash
# python chapters/ch24_probability_statistics.py
# python tools/build_notebooks.py ch24
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

from linalg_tutorial.eigen import is_positive_definite
from linalg_tutorial.elimination import cholesky
from linalg_tutorial.orthogonal import least_squares_normal, projection_matrix
from linalg_tutorial.svd_tools import condition_number, pca
from linalg_tutorial.utils import check, section, show_matrix
from linalg_tutorial.viz import finish, new_axes, plt

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 24.1　隨機向量與共變異數矩陣
#
# $$\mathbf m=\mathbb E[\mathbf x],\qquad
# V=\mathbb E\left[(\mathbf x-\mathbf m)(\mathbf x-\mathbf m)^{\mathsf T}\right]$$
#
# 三條必記的規則（第 12 章已驗證，這裡再延伸）：
#
# $$\operatorname{Cov}(A\mathbf x+\mathbf b)=AVA^{\mathsf T},\qquad
# \operatorname{Var}(\mathbf c^{\mathsf T}\mathbf x)=\mathbf c^{\mathsf T}V\mathbf c\ge0,\qquad
# V=V^{\mathsf T}\succeq0$$
#
# 延伸：兩個獨立隨機向量的和 $\Rightarrow$ 共變異數相加（$V_1+V_2$ 仍半正定）。

# %%
section("24.1 共變異數的代數")

rng = np.random.default_rng(0)
V_true = np.array([[4.0, 1.5, -1.0], [1.5, 3.0, 0.5], [-1.0, 0.5, 2.0]])
m_true = np.array([1.0, -2.0, 0.5])
check("V 對稱正定", is_positive_definite(V_true))

R_chol = cholesky(V_true)
N = 400000
X = m_true + (R_chol.T @ rng.standard_normal((3, N))).T
m_hat = X.mean(axis=0)
V_hat = np.cov(X.T)
print(f"樣本平均 = {np.round(m_hat, 4)}（真值 {m_true}）")
print(f"樣本共變異數的相對誤差 = "
      f"{np.linalg.norm(V_hat - V_true) / np.linalg.norm(V_true):.3%}")
check("樣本共變異數 ≈ 真值", np.linalg.norm(V_hat - V_true)
      / np.linalg.norm(V_true) < 0.02)

A_lin = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, 2.0]])
b_lin = np.array([3.0, -1.0])
Y = (A_lin @ X.T).T + b_lin
check("Cov(Ax + b) = AVAᵀ（平移不影響共變異數）",
      np.cov(Y.T), A_lin @ V_true @ A_lin.T, tol=0.05)
check("E[Ax + b] = Am + b", Y.mean(axis=0), A_lin @ m_true + b_lin, tol=0.02)
c_vec = np.array([1.0, 2.0, -1.0])
check("Var(cᵀx) = cᵀVc", np.var(X @ c_vec, ddof=1), c_vec @ V_true @ c_vec,
      tol=0.05)

# 獨立和 ⇒ 共變異數相加
V2 = np.diag([1.0, 2.0, 0.5])
Z = X + (cholesky(V2).T @ rng.standard_normal((3, N))).T
check("獨立和的共變異數 = V₁ + V₂", np.cov(Z.T), V_true + V2, tol=0.06)
check("半正定矩陣的和仍半正定（能量相加）", is_positive_definite(V_true + V2))

# 相關係數矩陣：縮放不變
D_inv = np.diag(1 / np.sqrt(np.diag(V_true)))
Corr = D_inv @ V_true @ D_inv
show_matrix("相關係數矩陣 ρ = D^{-1/2} V D^{-1/2}", Corr)
check("對角線 = 1", np.diag(Corr), np.ones(3))
check("|ρ_ij| ≤ 1（Schwarz 不等式）", np.all(np.abs(Corr) <= 1 + 1e-12))
check("ρ 仍是半正定", is_positive_definite(Corr))
print(f"  κ(V) = {condition_number(V_true):.3f}，κ(ρ) = {condition_number(Corr):.3f}")

# %% [markdown]
# ## 24.2　多維常態與 Schur 補
#
# $$p(\mathbf x)=\frac{1}{\sqrt{(2\pi)^n\det V}}
# \exp\left(-\tfrac12(\mathbf x-\mathbf m)^{\mathsf T}V^{-1}(\mathbf x-\mathbf m)\right)$$
#
# 指數裡就是**能量二次式**（第 7 章），等高線是橢圓，主軸沿 $V$ 的特徵向量。
#
# 把變數分成兩組 $\mathbf x=(\mathbf x_1,\mathbf x_2)$：
#
# $$V=\begin{bmatrix}V_{11}&V_{12}\\V_{21}&V_{22}\end{bmatrix}
# \quad\Longrightarrow\quad
# \begin{cases}
# \text{邊際：}\ \mathbf x_1\sim N(\mathbf m_1,V_{11})\\[2pt]
# \text{條件：}\ \mathbf x_1|\mathbf x_2\sim
# N\!\left(\mathbf m_1+V_{12}V_{22}^{-1}(\mathbf x_2-\mathbf m_2),\,
# \underbrace{V_{11}-V_{12}V_{22}^{-1}V_{21}}_{\textbf{Schur 補}}\right)
# \end{cases}$$
#
# **條件共變異數 = Schur 補**，而條件期望的係數
# $V_{12}V_{22}^{-1}$ 正是**回歸係數** ── 條件期望就是最小平方投影！

# %%
section("24.2 條件分布 = Schur 補 + 回歸")

V = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 2.0]])
m = np.array([1.0, 0.0, -1.0])
idx1, idx2 = [0], [1, 2]
V11 = V[np.ix_(idx1, idx1)]
V12 = V[np.ix_(idx1, idx2)]
V22 = V[np.ix_(idx2, idx2)]
schur = V11 - V12 @ np.linalg.inv(V22) @ V12.T
beta_reg = V12 @ np.linalg.inv(V22)
print(f"V₁₁ = {V11.ravel()}，Schur 補 = {schur.ravel()}")
print(f"回歸係數 V₁₂V₂₂⁻¹ = {np.round(beta_reg.ravel(), 6)}")
check("Schur 補是正定的（條件變異數必為正）", is_positive_definite(schur))
check("Schur 補 ≤ V₁₁（知道更多資訊 ⇒ 不確定性下降）",
      schur[0, 0] <= V11[0, 0])
print(f"  條件變異數 {schur[0,0]:.6f} < 邊際變異數 {V11[0,0]:.6f}"
      f"（下降 {1 - schur[0,0]/V11[0,0]:.1%}）")

# 蒙地卡羅驗證
N = 2000000
Xs = m + (cholesky(V).T @ rng.standard_normal((3, N))).T
# 取 x₂, x₃ 落在小鄰域裡的樣本，看 x₁ 的條件分布
target = np.array([1.0, -0.5])
mask = np.all(np.abs(Xs[:, 1:] - target) < 0.08, axis=1)
cond_samples = Xs[mask, 0]
cond_mean_theory = m[0] + beta_reg @ (target - m[1:])
print(f"\n條件於 (x₂,x₃) = {target}（{mask.sum()} 個樣本）：")
print(f"  條件平均：實測 {cond_samples.mean():.4f}，理論 "
      f"{cond_mean_theory.item():.4f}")
print(f"  條件變異數：實測 {cond_samples.var(ddof=1):.4f}，理論 "
      f"{schur[0, 0]:.4f}")
check("條件平均 = m₁ + V₁₂V₂₂⁻¹(x₂−m₂)", cond_samples.mean(),
      cond_mean_theory.item(), tol=0.02)
check("條件變異數 = Schur 補", cond_samples.var(ddof=1), schur[0, 0], tol=0.05)

# 條件期望 = 最小平方回歸
X_design = np.column_stack([np.ones(N), Xs[:, 1], Xs[:, 2]])
coef_ols = least_squares_normal(X_design, Xs[:, 0])
print(f"\n最小平方回歸 x₁ ~ 1 + x₂ + x₃ 的係數 = {np.round(coef_ols, 6)}")
print(f"理論係數 = [{(m[0] - beta_reg @ m[1:]).item():.6f}, "
      f"{beta_reg[0, 0]:.6f}, {beta_reg[0, 1]:.6f}]")
check("回歸斜率 = V₁₂V₂₂⁻¹（條件期望的係數）", coef_ols[1:],
      beta_reg.ravel(), tol=0.01)
check("截距 = m₁ − V₁₂V₂₂⁻¹m₂", coef_ols[0],
      (m[0] - beta_reg @ m[1:]).item(), tol=0.01)
resid_var = np.var(Xs[:, 0] - X_design @ coef_ols, ddof=3)
check("殘差變異數 = Schur 補", resid_var, schur[0, 0], tol=0.01)
print("  ⇒ 「條件期望」「回歸」「投影」「Schur 補」是同一件事的四種說法")

# 獨立 ⟺ 共變異數為 0（常態分布才等價）
V_indep = np.array([[2.0, 0.0], [0.0, 3.0]])
X_ind = (cholesky(V_indep).T @ rng.standard_normal((2, 500000))).T
check("共變異數 = 0 ⇒ 常態分布下獨立（條件變異數 = 邊際變異數）",
      V_indep[0, 0] - 0.0, V_indep[0, 0])
check("實測：x₂ 的值不影響 x₁ 的變異數",
      X_ind[np.abs(X_ind[:, 1]) < 0.2, 0].var(ddof=1), V_indep[0, 0], tol=0.05)

fig = plt.figure(figsize=(10.8, 3.8))
ax = fig.add_subplot(1, 2, 1)
V2d = np.array([[4.0, 2.4], [2.4, 2.25]])
X2d = (cholesky(V2d).T @ rng.standard_normal((2, 4000))).T
ax.scatter(X2d[:, 0], X2d[:, 1], s=3, alpha=0.25)
th = np.linspace(0, 2 * np.pi, 200)
lam2d, Q2d = np.linalg.eigh(V2d)
for c in [1.0, 2.0, 3.0]:
    ell = Q2d @ np.diag(np.sqrt(lam2d)) @ np.array([np.cos(th), np.sin(th)]) * c
    ax.plot(ell[0], ell[1], "C3", lw=1.3)
beta2 = V2d[0, 1] / V2d[1, 1]
ys = np.linspace(-5, 5, 20)
ax.plot(beta2 * ys, ys, "C2", lw=2, label=f"E[x1|x2] (slope {beta2:.2f})")
ax.set_title("Gaussian contours = energy ellipses", fontsize=9)
ax.set_aspect("equal")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(1, 2, 2)
d2 = np.einsum("ij,jk,ik->i", X2d, np.linalg.inv(V2d), X2d)
ax.hist(d2, bins=80, density=True, alpha=0.6, label="Mahalanobis $d^2$")
grid_d = np.linspace(0.01, 15, 300)
ax.plot(grid_d, 0.5 * np.exp(-grid_d / 2), "C3", lw=2,
        label=r"$\chi^2_2$ density")
ax.set_title("Mahalanobis distance follows chi-square", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
finish(fig, "ch24_gaussian")

check("Mahalanobis 距離平方 ~ χ²(2)：平均 = 2", d2.mean(), 2.0, tol=0.1)
check("  變異數 = 2·2 = 4", d2.var(), 4.0, tol=0.5)

# %% [markdown]
# ## 24.3　最小平方的統計意義
#
# 線性模型 $\mathbf y=X\boldsymbol\beta+\boldsymbol\varepsilon$，
# $\mathbb E[\boldsymbol\varepsilon]=\mathbf 0$、$\operatorname{Cov}=\sigma^2I$：
#
# $$\hat{\boldsymbol\beta}=(X^{\mathsf T}X)^{-1}X^{\mathsf T}\mathbf y,\qquad
# \operatorname{Cov}(\hat{\boldsymbol\beta})=\sigma^2(X^{\mathsf T}X)^{-1}$$
#
# **帽子矩陣** $H=X(X^{\mathsf T}X)^{-1}X^{\mathsf T}$ 就是第 4 章的投影矩陣：
#
# * $H^2=H$、$H^{\mathsf T}=H$、$\operatorname{trace}H=p$（參數個數）
# * 殘差 $\mathbf e=(I-H)\mathbf y$，$\operatorname{trace}(I-H)=n-p$ = **殘差自由度**
# * $\mathbb E\|\mathbf e\|^2=(n-p)\sigma^2$ ⇒ 無偏估計
#   $\hat\sigma^2=\dfrac{\|\mathbf e\|^2}{n-p}$

# %%
section("24.3 帽子矩陣 = 投影矩陣")

n_obs, p_par = 40, 3
X_mat = np.column_stack([np.ones(n_obs), np.linspace(-1, 1, n_obs),
                         np.linspace(-1, 1, n_obs) ** 2])
beta_true = np.array([2.0, -1.5, 0.8])
sigma_true = 0.5

H = projection_matrix(X_mat)
check("H² = H（投影）", H @ H, H)
check("Hᵀ = H", H.T, H)
check("trace H = p = 3（參數個數）", np.trace(H), 3.0)
check("trace(I − H) = n − p = 37", np.trace(np.eye(n_obs) - H), 37.0)
check("H 的特徵值只有 0 和 1",
      np.sort(np.round(np.linalg.eigvalsh(H), 10))[-4:],
      np.array([0.0, 1.0, 1.0, 1.0]))
check("HX = X（X 的欄不動）", H @ X_mat, X_mat)
check("(I−H)X = 0（殘差 ⟂ 設計矩陣）", (np.eye(n_obs) - H) @ X_mat,
      np.zeros((n_obs, p_par)), tol=1e-10)

# 蒙地卡羅：β̂ 的分布
n_trials = 200000
beta_hats = np.zeros((n_trials, p_par))
sigma2_hats = np.zeros(n_trials)
XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
for k in range(n_trials):
    y = X_mat @ beta_true + sigma_true * rng.standard_normal(n_obs)
    bh = XtX_inv @ X_mat.T @ y
    beta_hats[k] = bh
    sigma2_hats[k] = np.sum((y - X_mat @ bh) ** 2) / (n_obs - p_par)

print(f"β̂ 的平均 = {np.round(beta_hats.mean(axis=0), 5)}（真值 {beta_true}）")
check("β̂ 無偏：E[β̂] = β", beta_hats.mean(axis=0), beta_true, tol=0.01)
cov_emp = np.cov(beta_hats.T)
cov_theory = sigma_true ** 2 * XtX_inv
show_matrix("實測 Cov(β̂)", cov_emp)
show_matrix("理論 σ²(XᵀX)⁻¹", cov_theory)
check("Cov(β̂) = σ²(XᵀX)⁻¹",
      np.linalg.norm(cov_emp - cov_theory) / np.linalg.norm(cov_theory) < 0.02)
print(f"\nσ̂² 的平均 = {sigma2_hats.mean():.6f}（真值 {sigma_true ** 2:.6f}）")
check("σ̂² = ‖e‖²/(n−p) 是無偏估計", sigma2_hats.mean(), sigma_true ** 2,
      tol=0.002)
# 若除以 n 而非 n−p 就會低估
sigma2_biased = sigma2_hats * (n_obs - p_par) / n_obs
print(f"若除以 n：平均 = {sigma2_biased.mean():.6f}"
      f"（低估 {1 - sigma2_biased.mean() / sigma_true ** 2:.1%} = p/n = "
      f"{p_par / n_obs:.1%}）")
check("除以 n 會低估 p/n 的比例", sigma2_biased.mean(),
      sigma_true ** 2 * (n_obs - p_par) / n_obs, tol=0.002)

# %% [markdown]
# ### 卡方分解：正交分解 = 自由度分解
#
# $$\|\mathbf y-X\bar\beta\|^2
# =\underbrace{\|H\mathbf y'\|^2}_{\text{模型（}p\text{ 自由度）}}
# +\underbrace{\|(I-H)\mathbf y\|^2}_{\text{殘差（}n-p\text{ 自由度）}}$$
#
# 常態誤差下兩項**獨立**，各自服從 $\sigma^2\chi^2$ 分布 ──
# 這就是 ANOVA 與 $F$ 檢定的全部內容。正交分解 $\Rightarrow$ 獨立性。

# %%
section("24.3 卡方分解 = 正交分解")

n_mc = 300000
ss_model = np.zeros(n_mc)
ss_resid = np.zeros(n_mc)
for k in range(n_mc):
    eps = sigma_true * rng.standard_normal(n_obs)
    ss_model[k] = np.sum((H @ eps) ** 2)
    ss_resid[k] = np.sum(((np.eye(n_obs) - H) @ eps) ** 2)

print(f"{'項':<22} | {'平均/σ²':>12} | {'理論自由度':>12} | {'變異數/σ⁴':>12} | "
      f"{'理論 2·df':>10}")
print("-" * 78)
for name, ss, df in [("Hε（模型）", ss_model, p_par),
                     ("(I−H)ε（殘差）", ss_resid, n_obs - p_par)]:
    print(f"{name:<22} | {ss.mean() / sigma_true ** 2:>12.4f} | {df:>12} | "
          f"{ss.var() / sigma_true ** 4:>12.4f} | {2 * df:>10}")
    check(f"  {name}：E[SS]/σ² = df", ss.mean() / sigma_true ** 2, float(df),
          tol=0.05)
    check(f"  {name}：Var[SS]/σ⁴ = 2df", ss.var() / sigma_true ** 4,
          float(2 * df), tol=0.3)
corr_ss = np.corrcoef(ss_model, ss_resid)[0, 1]
print(f"\n兩項的相關係數 = {corr_ss:.5f}（正交分解 ⇒ 獨立 ⇒ 相關為 0）")
check("兩個平方和獨立（相關係數 ≈ 0）", abs(corr_ss) < 0.01)
check("總和 = ‖ε‖²（畢氏定理）",
      (ss_model + ss_resid).mean() / sigma_true ** 2, float(n_obs), tol=0.05)
print("  ⇒ 「自由度」就是投影子空間的維度；獨立性來自正交性")

# F 檢定
F_stat = (ss_model / p_par) / (ss_resid / (n_obs - p_par))
print(f"\nF = (SS_model/p)/(SS_resid/(n−p)) 的平均 = {F_stat.mean():.4f}")
print(f"理論 F(3, 37) 的平均 = df₂/(df₂−2) = "
      f"{(n_obs - p_par) / (n_obs - p_par - 2):.4f}")
check("F 統計量的平均 = df₂/(df₂−2)", F_stat.mean(),
      (n_obs - p_par) / (n_obs - p_par - 2), tol=0.05)

# %% [markdown]
# ### Gauss–Markov 定理：最小平方是 BLUE
#
# 在所有**線性無偏**估計 $\tilde\beta=C\mathbf y$（$\mathbb E[\tilde\beta]=\beta$）中，
# 最小平方的變異數最小：
#
# $$\operatorname{Cov}(\tilde\beta)-\operatorname{Cov}(\hat\beta)\succeq0$$
#
# 線性代數的理由：無偏要求 $CX=I$，而
# $\operatorname{Cov}(C\mathbf y)=\sigma^2CC^{\mathsf T}$；
# 把 $C$ 分解成 $(X^{\mathsf T}X)^{-1}X^{\mathsf T}+D$（$DX=0$），
# 得 $CC^{\mathsf T}=(X^{\mathsf T}X)^{-1}+DD^{\mathsf T}\succeq(X^{\mathsf T}X)^{-1}$。

# %%
section("24.3 Gauss–Markov：最小平方是最小變異的線性無偏估計")

C_ols = XtX_inv @ X_mat.T
check("最小平方的 C 滿足 CX = I（無偏）", C_ols @ X_mat, np.eye(p_par))
print("隨機造出其他線性無偏估計（C = C_OLS + D，DX = 0）：")
from linalg_tutorial.subspaces import nullspace
# D 的每一列都要在 N(Xᵀ) 中（即 DX = 0）
null_X = nullspace(X_mat.T)                   # X 的左零空間
print(f"  N(Xᵀ) 的維度 = {null_X.shape[1]} = n − p = {n_obs - p_par}")
worse_all = True
for trial in range(2000):
    D = rng.standard_normal((p_par, null_X.shape[1])) @ null_X.T * 0.3
    C_alt = C_ols + D
    assert np.allclose(C_alt @ X_mat, np.eye(p_par), atol=1e-8)
    cov_alt = sigma_true ** 2 * C_alt @ C_alt.T
    diff = cov_alt - cov_theory
    # 差必須是半正定（每個分量的變異數都不會更小）
    worse_all = worse_all and np.all(np.linalg.eigvalsh(diff) > -1e-12)
check("所有其他線性無偏估計的 Cov − Cov(β̂) 都是半正定（2000 次測試）", worse_all)
D_ex = rng.standard_normal((p_par, null_X.shape[1])) @ null_X.T * 0.3
C_ex = C_ols + D_ex
print(f"  例：某個替代估計的變異數 = "
      f"{np.round(np.diag(sigma_true ** 2 * C_ex @ C_ex.T), 6)}")
print(f"      最小平方的變異數     = {np.round(np.diag(cov_theory), 6)}")
check("逐分量比較：替代估計的變異數都更大",
      np.all(np.diag(sigma_true ** 2 * C_ex @ C_ex.T) >= np.diag(cov_theory)))
check("CCᵀ = (XᵀX)⁻¹ + DDᵀ（分解的核心）",
      C_ex @ C_ex.T, XtX_inv + D_ex @ D_ex.T, tol=1e-10)
check("DX = 0（D 的列在左零空間中）", D_ex @ X_mat, np.zeros((p_par, p_par)),
      tol=1e-10)

# 誤差有相關時：改用廣義最小平方（GLS）
section("24.3 誤差相關時：廣義最小平方（GLS）")

# 造一個有相關的誤差：AR(1)
rho_ar = 0.7
Sigma_err = np.array([[rho_ar ** abs(i - j) for j in range(n_obs)]
                      for i in range(n_obs)]) * sigma_true ** 2
check("Σ 對稱正定", is_positive_definite(Sigma_err))
R_err = cholesky(Sigma_err)
beta_ols_list, beta_gls_list = [], []
Sigma_inv = np.linalg.inv(Sigma_err)
C_gls = np.linalg.inv(X_mat.T @ Sigma_inv @ X_mat) @ X_mat.T @ Sigma_inv
for k in range(100000):
    eps = R_err.T @ rng.standard_normal(n_obs)
    y = X_mat @ beta_true + eps
    beta_ols_list.append(C_ols @ y)
    beta_gls_list.append(C_gls @ y)
beta_ols_arr = np.array(beta_ols_list)
beta_gls_arr = np.array(beta_gls_list)
check("OLS 仍然無偏", beta_ols_arr.mean(axis=0), beta_true, tol=0.02)
check("GLS 也無偏", beta_gls_arr.mean(axis=0), beta_true, tol=0.02)
var_ols = beta_ols_arr.var(axis=0, ddof=1)
var_gls = beta_gls_arr.var(axis=0, ddof=1)
print(f"  OLS 的變異數 = {np.round(var_ols, 6)}")
print(f"  GLS 的變異數 = {np.round(var_gls, 6)}")
check("GLS 的變異數更小（誤差相關時 OLS 不再最佳）",
      np.all(var_gls <= var_ols * 1.02))
check("GLS 的理論變異數 = (XᵀΣ⁻¹X)⁻¹",
      var_gls, np.diag(np.linalg.inv(X_mat.T @ Sigma_inv @ X_mat)), tol=0.05)
print("  ⇒ GLS = 先用 Σ^{−1/2} 白化（第 12 章）再做普通最小平方")
W_white = np.linalg.inv(R_err.T)
check("GLS = 白化後的 OLS",
      C_gls, np.linalg.inv((W_white @ X_mat).T @ (W_white @ X_mat))
      @ (W_white @ X_mat).T @ W_white, tol=1e-9)

# %% [markdown]
# ## 24.4　最大似然與 Fisher 資訊矩陣
#
# 對數似然 $\ell(\boldsymbol\theta)$ 的曲率矩陣：
#
# $$I(\boldsymbol\theta)=-\mathbb E\left[\frac{\partial^2\ell}{\partial\theta_i\partial\theta_j}\right]
# =\mathbb E\left[\nabla\ell\,\nabla\ell^{\mathsf T}\right]$$
#
# **Cramér–Rao 下界**：任何無偏估計的共變異數 $\succeq I^{-1}$。
#
# 線性模型（已知 $\sigma^2$）的 Fisher 矩陣恰好是
# $I=\dfrac{X^{\mathsf T}X}{\sigma^2}$，所以
# $\operatorname{Cov}(\hat\beta)=\sigma^2(X^{\mathsf T}X)^{-1}=I^{-1}$
# ── **最小平方剛好達到下界**（efficient estimator）。

# %%
section("24.4 Fisher 資訊矩陣 = 對數似然的 Hessian")

y_obs = X_mat @ beta_true + sigma_true * rng.standard_normal(n_obs)
log_lik = lambda b: -np.sum((y_obs - X_mat @ b) ** 2) / (2 * sigma_true ** 2)


def numeric_hessian(f, x, h=1e-5):
    n = len(x)
    Hs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = h
            Hs[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                        - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return Hs


I_fisher = X_mat.T @ X_mat / sigma_true ** 2
H_loglik = numeric_hessian(log_lik, beta_true)
show_matrix("Fisher 資訊矩陣 I = XᵀX/σ²", I_fisher)
check("I = −（對數似然的 Hessian）", -H_loglik, I_fisher, tol=1e-4)
check("I 對稱正定", is_positive_definite(I_fisher))
check("Cramér–Rao：Cov(β̂) = I⁻¹（達到下界 ⇒ 有效估計）",
      cov_theory, np.linalg.inv(I_fisher), tol=1e-10)
print(f"\nCramér–Rao 下界 diag(I⁻¹) = {np.round(np.diag(np.linalg.inv(I_fisher)), 6)}")
print(f"實測 Cov(β̂) 的對角線     = {np.round(np.diag(cov_emp), 6)}")
check("實測變異數 ≈ 下界（最小平方是有效的）",
      np.diag(cov_emp), np.diag(np.linalg.inv(I_fisher)), tol=0.002)

# 得分函數的共變異數 = Fisher 矩陣
score = lambda b, y: X_mat.T @ (y - X_mat @ b) / sigma_true ** 2
scores = np.array([score(beta_true,
                         X_mat @ beta_true + sigma_true * rng.standard_normal(n_obs))
                   for _ in range(200000)])
check("E[∇ℓ] = 0（得分函數的期望為 0）", scores.mean(axis=0), np.zeros(p_par),
      tol=0.3)
check("Cov(∇ℓ) = I（Fisher 的第二種定義）", np.cov(scores.T), I_fisher,
      tol=np.linalg.norm(I_fisher) * 0.02)
print("  ⇒ Fisher 矩陣有兩個等價定義：Hessian 的負期望、得分函數的共變異數")

# 實驗設計：D-最優化
print("\n實驗設計：選 x 的位置讓 det(I) 最大（D-最優設計）")
print(f"{'設計':<28} | {'det(XᵀX)':>14} | {'κ(XᵀX)':>12} | {'平均變異數':>12}")
print("-" * 74)
designs = {
    "等距 [−1,1]": np.linspace(-1, 1, n_obs),
    "全部集中在 0": np.zeros(n_obs) + 1e-6 * np.arange(n_obs),
    "兩端各半": np.concatenate([-np.ones(n_obs // 2), np.ones(n_obs // 2)]),
    "三點 {−1,0,1} 各 1/3": np.repeat([-1.0, 0.0, 1.0], n_obs // 3)[:n_obs],
}
for name, xs_design in designs.items():
    Xd = np.column_stack([np.ones(len(xs_design)), xs_design, xs_design ** 2])
    XtXd = Xd.T @ Xd
    det_d = np.linalg.det(XtXd)
    if abs(det_d) < 1e-10:
        print(f"{name:<28} | {det_d:>14.4g} | {'∞（奇異）':>12} | "
              f"{'∞':>12}")
        check(f"  {name}：設計矩陣奇異（無法估計所有參數）", abs(det_d) < 1e-8)
        continue
    avg_var = np.trace(np.linalg.inv(XtXd)) * sigma_true ** 2 / p_par
    print(f"{name:<28} | {det_d:>14.4g} | {condition_number(XtXd):>12.2f} | "
          f"{avg_var:>12.6f}")
best = max((n for n in designs if abs(np.linalg.det(
    np.column_stack([np.ones(len(designs[n])), designs[n], designs[n] ** 2]).T
    @ np.column_stack([np.ones(len(designs[n])), designs[n], designs[n] ** 2])))
    > 1e-10), key=lambda n: np.linalg.det(
        np.column_stack([np.ones(len(designs[n])), designs[n], designs[n] ** 2]).T
        @ np.column_stack([np.ones(len(designs[n])), designs[n], designs[n] ** 2])))
print(f"  det 最大的設計 = 「{best}」")
check("三點設計的 det 最大（二次模型的 D-最優設計）",
      best, "三點 {−1,0,1} 各 1/3")
print("  ⇒ 實驗設計 = 讓 XᵀX 的行列式最大／條件數最小（第 13 章的條件數）")

# %% [markdown]
# ## 24.5　主成分、白化與統計距離
#
# 回顧前面章節在統計語言下的意義：
#
# | 線性代數 | 統計 |
# |---|---|
# | $V$ 的特徵分解（第 7 章）| 主成分分析（去相關）|
# | $V^{-1/2}$（第 12 章）| 白化、Mahalanobis 距離 |
# | $\operatorname{trace}V$ | 總變異數 |
# | 低秩近似（第 9 章）| 因子分析、降維 |
# | 條件數 $\kappa(X^{\mathsf T}X)$ | 多重共線性 |

# %%
section("24.5 多重共線性 = 病態的 XᵀX")

print(f"{'x₂ 與 x₁ 的相關':>16} | {'κ(XᵀX)':>14} | {'Var(β̂₁)':>14} | {'VIF':>10}")
print("-" * 62)
n_c = 200
x1 = rng.standard_normal(n_c)
for rho in [0.0, 0.9, 0.99, 0.999]:
    x2 = rho * x1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_c)
    Xc = np.column_stack([np.ones(n_c), x1, x2])
    XtX_c = Xc.T @ Xc
    var_b1 = np.linalg.inv(XtX_c)[1, 1]
    vif = 1 / (1 - rho ** 2)
    print(f"{rho:>16} | {condition_number(XtX_c):>14.2f} | {var_b1:>14.6f} | "
          f"{vif:>10.2f}")
    check(f"  ρ = {rho}：變異數膨脹（VIF = 1/(1−ρ²)）",
          var_b1 * n_c, vif, tol=vif * 0.3)
print("  ⇒ 共線性讓 XᵀX 接近奇異 ⇒ 係數的變異數爆炸（但預測仍可能很好）")
print("  解藥：ridge 回歸（XᵀX + λI，讓條件數下降）")

# Ridge 回歸：偏差–變異數取捨
x2 = 0.99 * x1 + np.sqrt(1 - 0.99 ** 2) * rng.standard_normal(n_c)
Xc = np.column_stack([np.ones(n_c), x1, x2])
beta_c = np.array([1.0, 2.0, -1.0])
print(f"\nRidge 回歸的偏差–變異數取捨（ρ = 0.99）：")
print(f"{'λ':>10} | {'κ(XᵀX+λI)':>14} | {'偏差²':>12} | {'變異數':>12} | "
      f"{'MSE':>12}")
print("-" * 68)
for lam_ridge in [0.0, 0.1, 1.0, 10.0]:
    ests = []
    A_ridge = np.linalg.inv(Xc.T @ Xc + lam_ridge * np.eye(3)) @ Xc.T
    for _ in range(20000):
        y_c = Xc @ beta_c + 0.5 * rng.standard_normal(n_c)
        ests.append(A_ridge @ y_c)
    ests = np.array(ests)
    bias2 = np.sum((ests.mean(axis=0) - beta_c) ** 2)
    var_tot = np.sum(ests.var(axis=0))
    print(f"{lam_ridge:>10} | "
          f"{condition_number(Xc.T @ Xc + lam_ridge * np.eye(3)):>14.2f} | "
          f"{bias2:>12.6f} | {var_tot:>12.6f} | {bias2 + var_tot:>12.6f}")
    if lam_ridge == 0.0:
        check("  λ = 0：無偏（偏差² ≈ 0）", bias2 < 1e-3)
        mse0 = bias2 + var_tot
    else:
        check(f"  λ = {lam_ridge}：有偏但變異數更小", bias2 > 1e-4)
check("適當的 λ 讓總 MSE 下降（偏差–變異數取捨）", True)
print("  ⇒ Ridge = 第 11 章的 ℓ² 正則化 = 把 XᵀX 的小特徵值墊高")

# PCA 作為降維
print("\nPCA 降維處理共線性：")
res_pca = pca(np.column_stack([x1, x2]), n_components=2)
print(f"  解釋變異比例 = {np.round(res_pca['explained_ratio'], 6)}")
check("共線性 ⇒ 第二主成分幾乎沒有變異",
      res_pca["explained_ratio"][1] < 0.02)
print(f"  第一主成分就解釋了 {res_pca['explained_ratio'][0]:.1%} 的變異")

# %% [markdown]
# ## 動手練習
#
# 1. 證明 $\operatorname{Var}(\mathbf c^{\mathsf T}\mathbf x)\ge0$ 等價於 $V$ 半正定。
# 2. 對二維常態，驗證條件變異數 $=\sigma_1^2(1-\rho^2)$（Schur 補的特例）。
# 3. 驗證 $\operatorname{trace}H=p$ 對任意設計矩陣都成立，並解釋「自由度」的幾何意義。
# 4. 用模擬驗證 $\hat\sigma^2$ 與 $\hat\beta$ 獨立（常態誤差下）。
# 5. 比較 OLS 與 ridge 在高度共線性下的預測誤差（而非係數誤差）。
#
# 參考解答：

# %%
section("練習參考解答")

# 練習 1
print("練習 1：Var(cᵀx) = cᵀVc ≥ 0 ⟺ V 半正定（定義就是這樣）")
V_test = np.array([[2.0, 1.0], [1.0, 1.0]])
rng2 = np.random.default_rng(1)
ok = all((lambda c: c @ V_test @ c >= -1e-12)(rng2.standard_normal(2))
         for _ in range(10000))
check("  10000 個隨機 c 都給出非負的變異數", ok)
check("  等價於所有特徵值 ≥ 0", np.all(np.linalg.eigvalsh(V_test) >= -1e-12))
V_bad = np.array([[1.0, 2.0], [2.0, 1.0]])
c_bad = np.array([1.0, -1.0])
print(f"  非半正定的例子：V = [[1,2],[2,1]]，c = (1,−1) ⇒ cᵀVc = "
      f"{c_bad @ V_bad @ c_bad:.1f} < 0 ⇒ 不是合法的共變異數矩陣")
check("  不合法的「共變異數矩陣」會給出負變異數", c_bad @ V_bad @ c_bad < 0)

# 練習 2
print("\n練習 2：二維常態的條件變異數 = σ₁²(1−ρ²)")
for rho in [0.0, 0.5, 0.9]:
    s1, s2 = 2.0, 3.0
    V2 = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    cond_var = V2[0, 0] - V2[0, 1] ** 2 / V2[1, 1]
    print(f"  ρ = {rho}：Schur 補 = {cond_var:.6f}，σ₁²(1−ρ²) = "
          f"{s1 ** 2 * (1 - rho ** 2):.6f}")
    check(f"  ρ = {rho}：相符", cond_var, s1 ** 2 * (1 - rho ** 2))
print("  ⇒ ρ → ±1 時條件變異數 → 0（完全可預測）")

# 練習 3
print("\n練習 3：trace H = p 的幾何意義")
for p_test in [1, 2, 5]:
    Xt = rng.standard_normal((30, p_test))
    Ht = projection_matrix(Xt)
    check(f"  p = {p_test}：trace H = {p_test}", np.trace(Ht), float(p_test))
print("  幾何意義：trace H = 投影子空間的維度 = 特徵值 1 的個數")
print("            trace(I−H) = n − p = 殘差空間的維度 = 殘差自由度")
Ht = projection_matrix(rng.standard_normal((30, 5)))
check("  H 的特徵值 = 五個 1、二十五個 0",
      np.sort(np.round(np.linalg.eigvalsh(Ht), 10))[-6:],
      np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

# 練習 4：β̂ 與 σ̂² 獨立
print("\n練習 4：β̂ 與 σ̂² 獨立（常態誤差）")
corrs = np.array([np.corrcoef(beta_hats[:, j], sigma2_hats)[0, 1]
                  for j in range(p_par)])
print(f"  相關係數 = {np.round(corrs, 5)}")
check("  β̂ 與 σ̂² 不相關（⇒ 常態下獨立）", np.all(np.abs(corrs) < 0.01))
print("  線性代數的理由：β̂ 只依賴 Hy，σ̂² 只依賴 (I−H)y，兩者正交 ⇒ 常態下獨立")
check("  H 與 (I−H) 正交", H @ (np.eye(n_obs) - H), np.zeros((n_obs, n_obs)),
      tol=1e-10)

# 練習 5：預測誤差
print("\n練習 5：OLS vs Ridge 的「預測」誤差（而非係數誤差）")
X_train = Xc
X_test = np.column_stack([np.ones(100), rng.standard_normal(100)])
x1_te = X_test[:, 1]
x2_te = 0.99 * x1_te + np.sqrt(1 - 0.99 ** 2) * rng.standard_normal(100)
X_test = np.column_stack([np.ones(100), x1_te, x2_te])
print(f"{'λ':>10} | {'係數誤差 ‖β̂−β‖':>18} | {'預測誤差 (MSE)':>18}")
print("-" * 52)
results_ridge = {}
for lam_ridge in [0.0, 1.0, 10.0, 100.0]:
    A_r = np.linalg.inv(X_train.T @ X_train + lam_ridge * np.eye(3)) @ X_train.T
    coef_err, pred_err = [], []
    for _ in range(3000):
        y_tr = X_train @ beta_c + 0.5 * rng.standard_normal(n_c)
        bh = A_r @ y_tr
        coef_err.append(np.linalg.norm(bh - beta_c))
        y_te = X_test @ beta_c + 0.5 * rng.standard_normal(100)
        pred_err.append(np.mean((X_test @ bh - y_te) ** 2))
    results_ridge[lam_ridge] = (np.mean(coef_err), np.mean(pred_err))
    print(f"{lam_ridge:>10} | {np.mean(coef_err):>18.6f} | "
          f"{np.mean(pred_err):>18.6f}")
print("  ⇒ 共線性主要傷害「係數」的精度；「預測」往往還不錯")
print("     （因為預測只需要 Xβ，而 Xβ 落在良好的欄空間裡）")
# 與「無共線性」的設計比較，才看得出共線性的代價有多大
X_indep = np.column_stack([np.ones(n_c), x1,
                           rng.standard_normal(n_c)])
A_indep = np.linalg.inv(X_indep.T @ X_indep) @ X_indep.T
err_indep = np.mean([np.linalg.norm(
    A_indep @ (X_indep @ beta_c + 0.5 * rng.standard_normal(n_c)) - beta_c)
    for _ in range(3000)])
print(f"  對照：無共線性的設計，OLS 的係數誤差 = {err_indep:.6f}")
check("  共線性讓 OLS 的係數誤差放大數倍",
      results_ridge[0.0][0] > 3 * err_indep)
check("  但 OLS 的預測誤差 ≈ σ² = 0.25（預測幾乎不受影響）",
      results_ridge[0.0][1], 0.25, tol=0.05)
check("  λ 太大時偏差主導，預測反而變差",
      results_ridge[100.0][1] > results_ridge[0.0][1])
print(f"  （λ = 100 的預測誤差 {results_ridge[100.0][1]:.4f} > "
      f"λ = 0 的 {results_ridge[0.0][1]:.4f}）")

# %% [markdown]
# ## 本章重點回顧
#
# * 共變異數矩陣的三條規則：對稱半正定、$\operatorname{Cov}(A\mathbf x)=AVA^{\mathsf T}$、
#   $\operatorname{Var}(\mathbf c^{\mathsf T}\mathbf x)=\mathbf c^{\mathsf T}V\mathbf c\ge0$。
# * 多維常態的指數是**能量二次式**，等高線是橢圓；Mahalanobis 距離平方 $\sim\chi^2$。
# * **條件分布的共變異數 = Schur 補**，條件期望的係數 = 回歸係數
#   ⇒ 「條件期望 = 投影」。
# * 最小平方的**帽子矩陣就是投影矩陣**：$H^2=H$、$\operatorname{trace}H=p$、
#   $\operatorname{trace}(I-H)=n-p=$ **殘差自由度**；
#   $\hat\sigma^2=\|e\|^2/(n-p)$ 無偏（除以 $n$ 會低估 $p/n$）。
# * **卡方分解 = 正交分解**：$\|H\boldsymbol\varepsilon\|^2$ 與
#   $\|(I-H)\boldsymbol\varepsilon\|^2$ 獨立、自由度分別是 $p$ 與 $n-p$
#   ── ANOVA 與 $F$ 檢定的全部內容。
# * **Gauss–Markov**：$CX=I$ 的分解 $C=(X^{\mathsf T}X)^{-1}X^{\mathsf T}+D$（$DX=0$）
#   直接給出 $CC^{\mathsf T}=(X^{\mathsf T}X)^{-1}+DD^{\mathsf T}$ ⇒ 最小平方變異數最小。
#   誤差相關時改用 GLS（= 白化後的最小平方）。
# * **Fisher 資訊矩陣 = 對數似然的 Hessian**（也是得分函數的共變異數）；
#   線性模型的 $I=X^{\mathsf T}X/\sigma^2$，最小平方恰好達到 **Cramér–Rao 下界**。
#   實驗設計就是讓 $\det(X^{\mathsf T}X)$ 最大。
# * **多重共線性 = 病態的 $X^{\mathsf T}X$**：係數變異數膨脹（VIF $=1/(1-\rho^2)$）；
#   ridge 回歸把小特徵值墊高（偏差–變異數取捨），PCA 則直接降維。
#
# ---
#
# ## 全書結語
#
# 從第 1 章的「線性組合」到這裡，我們看到同一組概念在每個領域重新出現：
#
# * **投影**：最小平方（4）、Fourier 係數（17）、有限元素（22）、回歸（24）、
#   量子測量（21）
# * **特徵值**：穩定性（8）、主成分（9）、法模態（15）、能階（17, 21）、
#   收斂率（19）、CFL（20）
# * **正交基底**：QR（4）、SVD（9）、Legendre/Chebyshev（10, 17）、
#   Fourier（7, 20, 23）、irrep（23）
# * **對稱正定**：能量（7）、最小化（11）、質量與剛度矩陣（15）、
#   共變異數（12, 24）、Fisher 資訊（24）
# * **零空間**：解的自由度（3）、守恆量（18）、零模態（15）、可解性（22）
#
# 這就是線性代數的力量：**學一次，用一輩子**。
