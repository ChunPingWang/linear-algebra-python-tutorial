"""整合測試：執行每一章教材，確認沒有任何驗算失敗。

因為會真的跑完 24 章（含繪圖與蒙地卡羅模擬），所以標記為 slow：

    pytest                 # 只跑套件的單元測試（快）
    pytest --runslow       # 連同每一章一起跑（慢，數分鐘）
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAPTERS = sorted((ROOT / "chapters").glob("ch*.py"))


def test_chapters_exist():
    assert len(CHAPTERS) >= 24, f"只找到 {len(CHAPTERS)} 章"


@pytest.mark.slow
@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda p: p.stem)
def test_chapter_runs_without_failed_checks(chapter):
    proc = subprocess.run([sys.executable, str(chapter)], cwd=ROOT,
                          capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, f"{chapter.name} 執行失敗：\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    failed = [line for line in proc.stdout.splitlines() if "✗" in line]
    passed = [line for line in proc.stdout.splitlines() if "✓" in line]
    assert not failed, f"{chapter.name} 有 {len(failed)} 項驗算失敗：\n" + "\n".join(failed[:20])
    assert len(passed) > 0, f"{chapter.name} 沒有任何驗算"
