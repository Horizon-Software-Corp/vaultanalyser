import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
from typing import List, Dict

import math

from enum import StrEnum, auto
from itertools import chain


# ────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────
class Alternative(StrEnum):
    UP = "up"  # H1: SR > SR0
    DOWN = "down"  # H1: SR < SR0
    BOTH = "both"  # H1: SR ≠ SR0


class TestKind(StrEnum):
    PSR = "psr"
    HACt = "hact"  # Lo (2002) HAC-t


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def _clean(ret: np.ndarray) -> np.ndarray:
    """drop NaN and return copy"""
    return ret[~np.isnan(ret)].copy()


def _weights(n: int, LT: int | None) -> np.ndarray:
    """exp-decay weights (newest first) normalised to sum 1"""
    if LT is None or LT <= 0:
        return np.ones(n) / n
    ages = np.arange(n)  # age=0 is newest
    w = np.exp(-ages / LT)
    return w / w.sum()


# ────────────────────────────────────────────────────────────────
# low-level p-value calculators
# ────────────────────────────────────────────────────────────────
def _pvalue_psr(
    ret: np.ndarray,
    SR0: float,
    *,
    alternative: Alternative,
    LT: int | None,
) -> float:
    """Exponentially weighted Probabilistic Sharpe Ratio p-value."""
    ret = _clean(ret)[::-1]  # newest first
    n = len(ret)
    if n < 3:
        return np.nan

    w = _weights(n, LT)
    mu = np.dot(w, ret)
    var = np.dot(w, (ret - mu) ** 2)
    sig = np.sqrt(var)
    S = mu / sig

    g = np.dot(w, (ret - mu) ** 3) / sig**3
    k_ex = np.dot(w, (ret - mu) ** 4) / sig**4 - 3.0
    N_eff = 1.0 / np.sum(w**2)

    denom = np.sqrt((1 - g * S + 0.25 * k_ex * S**2) / (N_eff - 1))
    z = (S - SR0) / denom

    if alternative is Alternative.UP:
        return 1.0 - stats.norm.cdf(z)
    if alternative is Alternative.DOWN:
        return stats.norm.cdf(z)
    return 2.0 * (1.0 - stats.norm.cdf(abs(z)))


def _pvalue_hact(
    ret: np.ndarray,
    SR0: float,
    *,
    alternative: Alternative,
    hac_lags: int | None = None,
) -> float:
    """Lo (2002) HAC-t p-value (equal weights)."""
    ret = _clean(ret)
    n = len(ret)
    if n < 3:
        return np.nan

    mu, sig = ret.mean(), ret.std(ddof=1)
    S = mu / sig
    if hac_lags is None:
        hac_lags = int(np.floor(n ** (1 / 3)))

    rc = ret - mu
    gamma0 = (rc @ rc) / n
    rho = [(rc[:-l] @ rc[l:]) / ((n - l) * gamma0) for l in range(1, hac_lags + 1)]
    w = [1 - l / (hac_lags + 1) for l in range(1, hac_lags + 1)]
    f_L = 1 + 2 * np.sum(np.array(w) * np.array(rho))

    t = (S - SR0) * np.sqrt(n) / np.sqrt(f_L)
    if alternative is Alternative.UP:
        return 1.0 - stats.t.cdf(t, df=n - 1)
    if alternative is Alternative.DOWN:
        return stats.t.cdf(t, df=n - 1)
    return 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────
def p_sharpe(
    ret: np.ndarray,
    SR0: float = 0.1,
    *,
    kind: TestKind = TestKind.PSR,
    alternative: Alternative = Alternative.UP,
    LT: int | None = None,
    hac_lags: int | None = None,
) -> dict[str, float]:
    """
    Return example:
        {"p_psr_up_LT=60": 0.023}
    """
    key = f"p_{kind}_{alternative}_LT={LT}"

    if kind is TestKind.PSR:
        p = _pvalue_psr(ret, SR0, alternative=alternative, LT=LT)
    elif kind is TestKind.HACt:
        if LT is not None:
            raise ValueError("LT is not used for HAC-t test")
        p = _pvalue_hact(ret, SR0, alternative=alternative, hac_lags=hac_lags)
    else:  # safety
        raise ValueError("unknown TestKind")

    return {key: p}


# ────────────────────────────────────────────────────────────────
# block-wise, exp-weighted Fisher combination
# ────────────────────────────────────────────────────────────────
def _weighted_fisher_brown(p: np.ndarray, w: np.ndarray) -> float:
    """Good/Lancaster–Brown weighted Fisher (Gamma approximation)."""
    T = -2.0 * np.sum(w * np.log(p))
    mu = 2.0 * np.sum(w)
    var = 4.0 * np.sum(w**2)
    k = mu**2 / var
    theta = var / mu
    return stats.gamma.sf(T, k, scale=theta)


def block_tests(
    ret: np.ndarray,
    SR0: float = 0.1,
    *,
    kind: TestKind = TestKind.PSR,
    h: int = 30,
    LT: int | None = 60,  # -- sample & block lifetime
    alternative: Alternative = Alternative.UP,
) -> dict[str, float]:
    """
    1. 末尾から h 日ごとに非重複ブロックを作成（最新 → 過去）
    2. 各ブロックで `kind` に応じた p 値を計算
          • PSR  …  サンプルを EWMA( LT ) で重み付け
          • HAC-t…  等重み（Lo 2002 標準形）
    3. ブロック p 値を
          w_i = exp(− age / LT)  （age = i·h 日）で
       Good/Lancaster–Brown Fisher 合成
    """
    ret = ret[~np.isnan(ret)]
    if len(ret) < 3:
        raise ValueError("sample too short")

    p_blocks: list[float] = []
    for start in range(len(ret) - h, -1, -h):  # newest → oldest
        blk = ret[start : start + h]
        if len(blk) < 3:
            break
        if kind is TestKind.PSR:
            p_blocks.append(_pvalue_psr(blk, SR0, alternative=alternative, LT=LT))
        elif kind is TestKind.HACt:
            p_blocks.append(_pvalue_hact(blk, SR0, alternative=alternative))
        else:  # safety
            raise ValueError("unknown TestKind")

    p_arr = np.asarray(p_blocks)
    # block weights (newest block weight = 1)
    w_blocks = (
        np.ones_like(p_arr, dtype=float)
        if LT is None
        else np.exp(-np.arange(len(p_arr)) * h / LT)
    )

    key = f"p_{kind}_h={h}_LT={LT}_{alternative}"
    return {key: _weighted_fisher_brown(p_arr, w_blocks)}


def calculate_sharpe_reliability(rebuilded_pnl: List[float], h=30, SR0=0.1, LT=60):
    """
    Compute Sharpe–ratio p-values on (log-)P&L series.

    Parameters
    ----------
    rebuilded_pnl : list[float]
        Equity curve or accumulated P&L (length ≥ 2).
    h : int
        Block length for the Fisher aggregation.
    SR0 : float
        Sharpe‐ratio benchmark.

    Returns
    -------
    dict
        { "p_<kind>_...": value , ... }
    """
    if len(rebuilded_pnl) < 3:
        raise ValueError("rebuilded_pnl too short")

    # ------------------------- returns -------------------------
    pnl = np.asarray(rebuilded_pnl, dtype=float)
    ret = np.diff(pnl) / pnl[:-1]  # simple daily return
    # -----------------------------------------------------------

    # helper to merge dicts
    def merge(*dicts):
        return dict(chain.from_iterable(d.items() for d in dicts))

    # -------- entire window -----------------------------------
    p_all_up = merge(
        p_sharpe(ret, SR0, kind=TestKind.HACt, alternative=Alternative.UP),
        p_sharpe(ret, SR0, kind=TestKind.PSR, alternative=Alternative.UP, LT=None),
        p_sharpe(ret, SR0, kind=TestKind.PSR, alternative=Alternative.UP, LT=LT),
    )

    p_all_down = merge(
        p_sharpe(ret, SR0, kind=TestKind.HACt, alternative=Alternative.DOWN),
        p_sharpe(ret, SR0, kind=TestKind.PSR, alternative=Alternative.DOWN, LT=None),
        p_sharpe(ret, SR0, kind=TestKind.PSR, alternative=Alternative.DOWN, LT=LT),
    )

    # -------- block-wise aggregation ---------------------------
    p_blk_up = merge(
        block_tests(
            ret, SR0, kind=TestKind.HACt, h=h, LT=None, alternative=Alternative.UP
        ),
        block_tests(
            ret, SR0, kind=TestKind.PSR, h=h, LT=None, alternative=Alternative.UP
        ),
        block_tests(
            ret, SR0, kind=TestKind.PSR, h=h, LT=60, alternative=Alternative.UP
        ),
    )

    p_blk_down = merge(
        block_tests(
            ret, SR0, kind=TestKind.HACt, h=h, LT=None, alternative=Alternative.DOWN
        ),
        block_tests(
            ret, SR0, kind=TestKind.PSR, h=h, LT=None, alternative=Alternative.DOWN
        ),
        block_tests(
            ret, SR0, kind=TestKind.PSR, h=h, LT=60, alternative=Alternative.DOWN
        ),
    )

    # -------- concat & return ---------------------------------
    return merge(p_all_up, p_all_down, p_blk_up, p_blk_down)

    # # Add debug information if requested
    # if is_debug:
    #     debug_data = extract_window_debug_data(test_results, rebuilded_pnl, h)
    #     result["debug_windows"] = debug_data

    # return result
