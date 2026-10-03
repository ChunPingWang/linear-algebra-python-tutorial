# 線性代數教材：常用指令
PY := .venv/bin/python

.PHONY: help setup notebooks run test test-all clean

help:
	@echo "make setup      建立 .venv 並安裝依賴"
	@echo "make notebooks  由 chapters/*.py 產生 notebooks/*.ipynb"
	@echo "make run        依序執行全部章節（顯示每章的驗算統計）"
	@echo "make test       執行套件的單元測試（快）"
	@echo "make test-all   單元測試 + 跑完每一章的整合測試"
	@echo "make clean      清除產生的圖與快取"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

notebooks:
	$(PY) tools/build_notebooks.py

run:
	@total_ok=0; total_fail=0; \
	for f in chapters/ch*.py; do \
		out=$$($(PY) $$f 2>&1); \
		ok=$$(printf '%s' "$$out" | grep -c '✓'); \
		bad=$$(printf '%s' "$$out" | grep -c '✗'); \
		total_ok=$$((total_ok+ok)); total_fail=$$((total_fail+bad)); \
		printf '%-42s %4d 通過 / %d 失敗\n' "$$(basename $$f)" "$$ok" "$$bad"; \
	done; \
	printf '\n合計：%d 項通過，%d 項失敗\n' "$$total_ok" "$$total_fail"

test:
	$(PY) -m pytest tests/ -q

test-all:
	$(PY) -m pytest tests/ -q --runslow

clean:
	rm -rf figures/*.png __pycache__ */__pycache__ .pytest_cache
