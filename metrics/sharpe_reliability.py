import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
from typing import List, Dict

import math

from enum import StrEnum
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

    tol = 1e-8  # tolerance for numerical stability
    if np.all(np.abs(ret) <= tol):
        return 0.5  # 分布中心だとみなす

    n = len(ret)
    if n < 3:
        return np.nan

    w = _weights(n, LT)
    mu = np.dot(w, ret)
    var_biased = np.dot(w, (ret - mu) ** 2)
    c = 1.0 - np.sum(w**2)  # <─ 有効自由度比
    var_unbias = var_biased / c
    sig = np.sqrt(var_unbias)

    S = mu / sig

    g = np.dot(w, (ret - mu) ** 3) / sig**3
    k_ex = np.dot(w, (ret - mu) ** 4) / sig**4 - 3.0
    N_eff = 1.0 / np.sum(w**2)

    if abs(-g * S + 0.25 * k_ex * S**2) <= 0.5:
        Delta1 = 1 - g * S + 0.25 * k_ex * S**2
    else:
        # 正規分布周りの展開、1 >> -g * S + 0.25 * k_ex * S**2 が破綻
        Delta1 = 1

    denom = np.sqrt(Delta1 / (N_eff - 1))
    z = (S - SR0) / denom

    if alternative is Alternative.UP:
        p = 1.0 - stats.norm.cdf(z)
    elif alternative is Alternative.DOWN:
        p = stats.norm.cdf(z)
    elif alternative is Alternative.BOTH:
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    if p == 0:
        p = 1e-8  # avoid log(p) = -inf

    return p


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

    tol = 1e-8  # tolerance for numerical stability
    if np.all(np.abs(ret) <= tol):
        return 0.5  # 分布中心だとみなす

    if n < 3:
        return np.nan

    mu, sig = ret.mean(), ret.std(ddof=1)

    S = mu / sig

    if hac_lags is None:
        hac_lags = int(np.floor(n ** (1 / 3)))

    N_eff = n - 1
    rc = ret - mu
    gamma0 = (rc @ rc) / n
    # rho = [(rc[:-l] @ rc[l:]) / ((n - l) * gamma0) for l in range(1, hac_lags + 1)] # 正式な推定量
    rho = [
        (rc[:-l] @ rc[l:]) / (n * gamma0) for l in range(1, hac_lags + 1)
    ]  # n-l -> nでf_Lが正定値になる
    w = [1 - l / (hac_lags + 1) for l in range(1, hac_lags + 1)]
    f_L = max(
        1 + 2 * np.sum(np.array(w) * np.array(rho)), 10e-8
    )  # avoid division by zero

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
        if hac_lags is not None:
            print("Warning: hac_lags is ignored for PSR test, using LT for weights.")
    elif kind is TestKind.HACt:
        if LT is not None:
            print("Warning: LT is ignored for HAC-t test, using equal weights.")
        p = _pvalue_hact(ret, SR0, alternative=alternative, hac_lags=hac_lags)
    else:  # safety
        return {}

    if p == 0:
        print(f"Warning: p-value is 0 for {key}, likely due to numerical issues.")

    return {key: p}


# ────────────────────────────────────────────────────────────────
# block-wise, exp-weighted Fisher combination
# ────────────────────────────────────────────────────────────────
def _weighted_fisher_brown(p_vals: np.ndarray, weights: np.ndarray) -> float:
    """
    Brown–Kost–McDermott 近似で weighted Fisher を Γ 分布に当てはめ、
    combined p-value (右片側) を返す。
    """
    # ---------- 1. test statistic ----------
    if np.any(p_vals <= 0):
        print("p_vals negative", p_vals)
    fisher_stat = -2.0 * np.sum(weights * np.log(p_vals))

    # ---------- 2. null mean & variance ----------
    mean_null = 2.0 * np.sum(weights)  # E[T]
    var_null = 4.0 * np.sum(weights**2)  # Var[T]
    if mean_null <= 0 or var_null <= 0:
        print(f"Warning: mean_null:{mean_null}, weights:{weights}")

    # ---------- 3. gamma parameters ----------
    shape = mean_null**2 / var_null  # k
    scale = var_null / mean_null  # θ

    # ---------- 4. p-value ----------
    return stats.gamma.sf(fisher_stat, shape, scale=scale)


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
    p_blocks: list[float] = []

    # --- 1️⃣ ブロック開始位置のリストを作る ―― newest → oldest --- #
    starts = list(range(len(ret) - h, -1, -h))  # len(ret) >= h のぶん
    if not starts:  # len(ret) < h なら空リスト
        starts = [0]  # 全区間を 1 ブロックとして追加

    # --- 2️⃣ ブロックを取り出して検定 --- #
    for start in starts:
        blk = ret[start : start + h]  # len(ret) < h なら ret[:h] = ret
        if len(blk) < 3:  # サンプル 3 未満はスキップ
            continue

        """PSR か HAC-t かを切り替えて p 値を返す小さなヘルパ."""
        res = p_sharpe(blk, SR0, kind=kind, alternative=alternative, LT=LT)
        p_blocks.append(next(iter(res.values())))

    if p_blocks:
        p_arr = np.asarray(p_blocks)
        # block weights (newest block weight = 1)
        w_blocks = (
            np.ones_like(p_arr, dtype=float)
            if LT is None
            else np.exp(-np.arange(len(p_arr)) * h / LT)
        )

        key = f"p_{kind}_h={h}_LT={LT}_{alternative}"
        return {key: _weighted_fisher_brown(p_arr, w_blocks)}
    else:
        return {}


def calculate_sharpe_reliability(ret: np.ndarray, h=30, SR0=0.1, LT=60):
    """
    Calculate the reliability of the Sharpe ratio using various tests.
    Args:
        ret (np.ndarray): Returns data. Daily returns are expected.
        h (int): block size day
        SR0 (float): Reference Sharpe ratio to test against.
        LT (int | None): Lifetime for exponential weighting, None for equal weights.
    Returns:
        dict: Dictionary containing p-values for different tests.
    """

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
            ret, SR0, kind=TestKind.PSR, h=h, LT=LT, alternative=Alternative.UP
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
            ret, SR0, kind=TestKind.PSR, h=h, LT=LT, alternative=Alternative.DOWN
        ),
    )

    # -------- concat & return ---------------------------------
    return merge(p_all_up, p_blk_up, p_all_down, p_blk_down)

    # # Add debug information if requested
    # if is_debug:
    #     debug_data = extract_window_debug_data(test_results, rebuilded_pnl, h)
    #     result["debug_windows"] = debug_data

    # return result
