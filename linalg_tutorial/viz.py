"""繪圖輔助：同一份程式碼在 notebook 裡顯示圖、在終端機裡存成 PNG。

圖上的文字刻意使用英文/數學符號，避免不同系統缺少中文字型而出現方框。
中文說明一律放在 markdown 或 print 輸出。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


if not _in_notebook() and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")      # 腳本模式：不需要視窗即可畫圖

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "new_axes", "finish", "draw_vector", "FIG_DIR"]

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "font.size": 10,
})


def new_axes(title: str = "", figsize=(5.2, 5.2), equal: bool = True, d3: bool = False):
    """建立一張圖，回傳 (fig, ax)。"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d") if d3 else fig.add_subplot(111)
    if title:
        ax.set_title(title)
    if equal and not d3:
        ax.set_aspect("equal", adjustable="box")
    return fig, ax


def draw_vector(ax, vec, label: str = "", color: str = "C0", origin=(0.0, 0.0), **kw):
    """畫一支從 origin 出發的箭頭（2D）。"""
    ox, oy = origin
    ax.annotate("", xy=(ox + vec[0], oy + vec[1]), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, **kw))
    if label:
        ax.text(ox + vec[0] * 1.06, oy + vec[1] * 1.06, label, color=color, fontsize=11)


def finish(fig, name: str) -> Path:
    """notebook：直接顯示；腳本：存檔到 figures/ 並印出路徑。"""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / (name if name.endswith(".png") else name + ".png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if _in_notebook():
        plt.show()
    else:
        plt.close(fig)
        print(f"  [圖] 已存檔 {path.relative_to(FIG_DIR.parent)}")
    return path
