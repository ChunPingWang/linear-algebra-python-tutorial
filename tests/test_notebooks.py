"""確認 notebooks 與 chapters 同步，且格式合法。"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_notebooks import split_cells  # noqa: E402


def test_every_chapter_has_a_notebook():
    for chapter in sorted((ROOT / "chapters").glob("ch*.py")):
        nb = ROOT / "notebooks" / (chapter.stem + ".ipynb")
        assert nb.exists(), f"缺少 {nb.name}（請執行 tools/build_notebooks.py）"


def test_notebooks_are_valid_and_up_to_date():
    for chapter in sorted((ROOT / "chapters").glob("ch*.py")):
        nb_path = ROOT / "notebooks" / (chapter.stem + ".ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert all(c["cell_type"] in ("markdown", "code") for c in nb["cells"])
        # 儲存格數量必須與原始碼切出來的一致（notebook 沒有過期）
        cells = split_cells(chapter.read_text(encoding="utf-8"))
        assert len(nb["cells"]) == len(cells), (
            f"{nb_path.name} 已過期（{len(nb['cells'])} vs {len(cells)} 個儲存格），"
            "請重新執行 tools/build_notebooks.py")


def test_first_cell_is_a_markdown_title():
    for chapter in sorted((ROOT / "chapters").glob("ch*.py")):
        nb = json.loads((ROOT / "notebooks" / (chapter.stem + ".ipynb"))
                        .read_text(encoding="utf-8"))
        first = nb["cells"][0]
        assert first["cell_type"] == "markdown"
        assert first["source"][0].startswith("# 第")
