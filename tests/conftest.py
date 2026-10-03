"""pytest 設定：加入專案根目錄到 sys.path，並定義 --runslow 選項。"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="同時執行「跑完每一章」的整合測試（較慢）")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 需要執行整章教材的測試")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --runslow 才會執行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
