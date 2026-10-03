"""linalg_tutorial — 《線性代數教材》隨書程式庫

本套件把教材中每一章用到的演算法「從零實作」一次（只依賴 numpy 的
陣列運算，不呼叫 numpy.linalg 的高階函式），再與 numpy / scipy 的結果
互相驗證。這樣讀者既能看到數學流程，也能確認結果正確。

模組一覽
--------
``utils``        列印、比較、驗算用的小工具
``elimination``  消去法、回代、A = LU、PA = LU、Gauss–Jordan 反矩陣
``subspaces``    rref、A = CR、四個基本子空間、Ax = b 的完整解
``determinant``  餘因子展開、LU 求行列式、Cramer 法則、體積
``orthogonal``   投影、Gram–Schmidt、Householder QR、最小平方、偽逆
``eigen``        特徵值（冪法、反冪法、QR 演算法、Jacobi）、對角化、矩陣指數
``svd_tools``    SVD、低秩近似、PCA
``viz``          繪圖輔助（腳本模式存檔，notebook 模式直接顯示）
"""

from . import utils, elimination, subspaces, determinant, orthogonal, eigen, svd_tools

__all__ = [
    "utils",
    "elimination",
    "subspaces",
    "determinant",
    "orthogonal",
    "eigen",
    "svd_tools",
]

__version__ = "1.0.0"
