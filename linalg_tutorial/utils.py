"""列印與驗算用的小工具。"""

from __future__ import annotations

import numpy as np

__all__ = [
    "show_matrix",
    "section",
    "check",
    "allclose",
    "round_clean",
    "rand_matrix",
]

_PASS = "✓"
_FAIL = "✗"


def section(title: str, char: str = "=") -> None:
    """印出一個區段標題，讓腳本輸出容易閱讀。"""
    line = char * max(60, len(title) + 4)
    print(f"\n{line}\n{title}\n{line}")


def show_matrix(name: str, A, fmt: str = "{: 9.4g}") -> None:
    """以對齊的方式印出矩陣或向量。"""
    A = np.asarray(A)
    if A.ndim == 0:
        print(f"{name} = {A.item():.6g}")
        return
    if A.ndim == 1:
        A = A.reshape(-1, 1)
        label = f"{name} (直欄向量, {A.shape[0]})"
    else:
        label = f"{name} ({A.shape[0]}x{A.shape[1]})"
    print(f"{label} =")
    for row in A:
        print("  [" + " ".join(fmt.format(float(np.real(x))) if np.isrealobj(A)
                               else f"{x: .4g}" for x in row) + " ]")


def round_clean(A, decimals: int = 10):
    """把 -0.0 與浮點雜訊清掉，方便肉眼比對。"""
    A = np.asarray(A, dtype=float)
    out = np.round(A, decimals)
    out[out == 0] = 0.0
    return out


def allclose(a, b, tol: float = 1e-9) -> bool:
    """容忍浮點誤差的相等判斷（可處理形狀相同的實數/複數陣列）。"""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=tol * 1e3, atol=tol))


def check(label: str, got, expected=None, tol: float = 1e-9) -> bool:
    """驗算一條敘述並印出結果。

    - ``expected`` 為 None 時，``got`` 當成布林值判斷。
    - 否則比較 ``got`` 與 ``expected`` 是否在容差內相等。

    回傳布林值，方便在測試中重複使用。
    """
    if expected is None:
        ok = bool(got)
        detail = ""
    else:
        ok = allclose(got, expected, tol)
        if not ok:
            g, e = np.asarray(got), np.asarray(expected)
            if g.shape == e.shape:
                detail = f"  (最大誤差 {np.max(np.abs(g - e)):.3g})"
            else:
                detail = f"  (形狀不同 {g.shape} vs {e.shape})"
        else:
            detail = ""
    print(f"  {_PASS if ok else _FAIL} {label}{detail}")
    return ok


def rand_matrix(m: int, n: int, seed: int = 0, integer: bool = False, low: int = -5, high: int = 6):
    """產生可重現的隨機矩陣（教材中所有隨機例子都固定 seed）。"""
    rng = np.random.default_rng(seed)
    if integer:
        return rng.integers(low, high, size=(m, n)).astype(float)
    return rng.standard_normal((m, n))
