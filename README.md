# 線性代數教材（Python 實作）

以 **Gilbert Strang《Introduction to Linear Algebra》第 6 版** 為主幹、
**Riley, Hobson & Bence《Mathematical Methods for Physics and Engineering》第 3 版**
為延伸的中文線性代數教材。每一章都同時提供：

* **可直接執行的 Python 程式**（`chapters/chNN_*.py`）
* **對應的 Jupyter Notebook**（`notebooks/chNN_*.ipynb`，由同一份原始碼自動產生）
* **從零實作的演算法套件**（`linalg_tutorial/`），並與 NumPy / SciPy 互相驗證

教材中的每一個數學敘述都附帶 `✓` 的程式驗算，執行章節就會看到驗證結果。

---

## 快速開始

```bash
# 1. 建立環境
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 執行任一章（會印出完整講解與驗算結果，並把圖存到 figures/）
.venv/bin/python chapters/ch01_vectors_and_matrices.py

# 3. 產生／更新 Jupyter notebooks
.venv/bin/python tools/build_notebooks.py          # 全部
.venv/bin/python tools/build_notebooks.py ch01     # 指定章節

# 4. 用 Jupyter 開啟
.venv/bin/jupyter lab notebooks/
```

## 專案結構

```
linear-algebra-python-tutorial/
├── chapters/           教材原始碼（單一真實來源，可直接 python 執行）
├── notebooks/          由 chapters/ 自動產生的 .ipynb
├── linalg_tutorial/    隨書程式庫：從零實作的線性代數演算法
│   ├── utils.py        列印、驗算工具
│   ├── elimination.py  消去法、A = LU、PA = LU、Cholesky、差分矩陣
│   ├── subspaces.py    rref、A = CR、四個基本子空間、完整解
│   ├── determinant.py  餘因子展開、LU 求行列式、Cramer 法則
│   ├── orthogonal.py   投影、Gram–Schmidt、Householder QR、最小平方、偽逆
│   ├── eigen.py        冪法、QR 演算法、Jacobi、對角化、矩陣指數
│   ├── svd_tools.py    SVD、低秩近似、PCA、條件數
│   └── viz.py          繪圖輔助（腳本存檔／notebook 直接顯示）
├── tools/              notebook 產生器
├── docs/               課程地圖、原書章節對照表
├── figures/            執行後產生的圖（不納入版本控制）
└── requirements.txt
```

### 設計理念：單一真實來源

每章只寫一份 `chapters/chNN_*.py`，使用與 jupytext 相容的 `# %%` 格式。
`tools/build_notebooks.py` 再把它切成 markdown / code 儲存格產生 notebook，
因此**講解文字與程式碼永遠不會對不上**。

## 章節一覽

| 章 | 主題 | 對應原書 |
|---|---|---|
| 1 | 向量、矩陣與欄空間 | Strang 1；Riley 7.1–7.6, 8.1–8.4 |
| 2 | 消去法、反矩陣與 $A=LU$ | Strang 2；Riley 8.10, 8.18, 27.3 |
| 3 | 四個基本子空間 | Strang 3；Riley 8.1, 8.11 |
| 4 | 正交性、最小平方、$A=QR$、偽逆 | Strang 4；Riley 8.1, 31.6 |
| 5 | 行列式：面積、體積與可逆性 | Strang 5；Riley 8.9, 6.4, 7.6 |
| 6 | 特徵值與對角化 | Strang 6.1–6.2；Riley 8.13–8.16 |
| 7 | 對稱矩陣、正定性與複數／Hermitian 矩陣 | Strang 6.3–6.4；Riley 8.7, 8.12, 8.17 |
| 8 | 線性微分方程與矩陣指數 | Strang 6.5；Riley 15.1, 27.6 |
| 9 | SVD、影像壓縮與 PCA | Strang 7；Riley 8.18, 31.2 |
| 10 | 線性轉換、基底變換與 Jordan 形式 | Strang 8、附錄 5；Riley 8.2, 8.15, 26.2 |
| 11 | 最佳化中的線性代數 | Strang 9；Riley 5.8–5.9 |

> 後續章節（從資料學習、Strang 附錄精選、以及 Riley 各章的線性代數視角補充）
> 持續增補中，詳見 [`docs/00-課程地圖.md`](docs/00-課程地圖.md)。

## 關於參考書

本教材參考下列兩本書的內容編寫，但**兩個 PDF 檔皆為版權素材，不納入版本控制**
（見 `.gitignore`）。若要重現編寫過程，請自備書籍：

* Gilbert Strang, *Introduction to Linear Algebra*, 6th ed., Wellesley–Cambridge Press, 2023.
* K. F. Riley, M. P. Hobson, S. J. Bence, *Mathematical Methods for Physics and
  Engineering*, 3rd ed., Cambridge University Press, 2006.

## 備註

* 圖表中的文字刻意使用英文／數學符號，避免不同系統缺少中文字型而顯示方框；
  中文說明一律放在 markdown 與終端機輸出中。
* 所有隨機範例都固定 `seed`，執行結果可重現。
