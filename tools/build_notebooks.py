#!/usr/bin/env python3
"""把 chapters/*.py（percent 格式）轉成 notebooks/*.ipynb。

設計理念：**單一真實來源**。每一章的教材只寫一份 `chapters/chNN_*.py`，
它本身就是可以 `python chapters/chNN_xxx.py` 直接執行的程式；
本工具再把同一份檔案切成 markdown / code 儲存格，產生對應的 Jupyter notebook。
這樣講解文字與程式碼永遠不會對不上。

格式約定（與 jupytext 的 "percent" 格式相容）::

    # %% [markdown]
    # # 標題
    # 說明文字（每行開頭的 "# " 會被移除）

    # %%
    print("這是程式碼儲存格")

用法::

    python tools/build_notebooks.py              # 全部轉換
    python tools/build_notebooks.py ch01 ch02    # 只轉換指定章節
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
NOTEBOOKS = ROOT / "notebooks"

CELL_RE = re.compile(r"^# %%(?P<meta>.*)$")


def split_cells(text: str) -> list[tuple[str, str]]:
    """把 percent 格式的原始碼切成 [(cell_type, source), ...]。"""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            if buf:
                cells.append((kind, buf))
            meta = m.group("meta").strip()
            kind = "markdown" if "markdown" in meta else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))

    out: list[tuple[str, str]] = []
    for kind, lines in cells:
        if kind == "markdown":
            body = "\n".join(_strip_comment(l) for l in lines)
        else:
            body = "\n".join(lines)
        body = body.strip("\n")
        if body.strip():
            out.append((kind, body))
    return out


def _strip_comment(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line.strip() == "#":
        return ""
    return line


def make_notebook(cells: list[tuple[str, str]], title: str) -> dict:
    nb_cells = []
    for kind, body in cells:
        source = [l + "\n" for l in body.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(selected: list[str] | None = None) -> list[Path]:
    NOTEBOOKS.mkdir(exist_ok=True)
    made = []
    for src in sorted(CHAPTERS.glob("ch*.py")):
        if selected and not any(src.name.startswith(s) for s in selected):
            continue
        cells = split_cells(src.read_text(encoding="utf-8"))
        nb = make_notebook(cells, src.stem)
        dst = NOTEBOOKS / (src.stem + ".ipynb")
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        n_md = sum(1 for k, _ in cells if k == "markdown")
        n_code = len(cells) - n_md
        print(f"  {src.name} → {dst.relative_to(ROOT)}  ({n_md} markdown / {n_code} code 儲存格)")
        made.append(dst)
    return made


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    print("建立 Jupyter notebooks：")
    made = build(args or None)
    if not made:
        print("  （沒有符合條件的章節）")
        return 1
    print(f"完成，共 {len(made)} 本 notebook。")
    try:
        import nbformat                                    # 有裝就順手驗證格式
        for p in made:
            nbformat.validate(nbformat.read(p, as_version=4))
        print("nbformat 驗證通過。")
    except ImportError:
        print("（未安裝 nbformat，略過格式驗證）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
