import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import math


def rolling_sharpe_tests_indep(df: pd.DataFrame,
                               SR0: float = 0.5,
                               T: int = 30,
                               alpha: float = 0.05) -> pd.DataFrame:
    """
    Rolling Jobson-Korkie/Memmel Z-test & Lo (HAC) t-test
    with Bonferroni correction based on 'independent windows × Meff'.

    Bonferroni factor = ceil(valid_days / T) * Meff
        • ceil(valid_days / T) … 窓の"ほぼ独立"個数
        • Meff … 同じ窓内での JKM と Lo の実効独立度
                 （典型的に 1.2–1.6 程度で安定）
    """
    r = df['returns']
    win = r.rolling(T)

    # -- ローリング平均・標準偏差・Sharpe
    mean_r = win.mean()
    std_r  = win.std(ddof=1)
    sharpe = mean_r / std_r

    # -- Jobson–Korkie / Memmel Z
    var_sr = (1 + 0.5 * SR0**2) / T
    Z_jkm = (sharpe - SR0) / np.sqrt(var_sr)
    p_jkm = 2 * (1 - norm.cdf(np.abs(Z_jkm)))

    # -- Lo (HAC) t
    q = int(np.floor(T ** 0.25))
    demean = r - mean_r
    gamma0 = win.var(ddof=0)
    gamma_sum = 0.0
    for k in range(1, q + 1):
        w = 1 - k / (q + 1)
        cov_k = (demean * demean.shift(k)).rolling(T).mean()
        gamma_sum += w * cov_k
    var_mean_hac = (gamma0 + 2 * gamma_sum) / T
    se_mean_hac  = np.sqrt(var_mean_hac)
    t_lo = (mean_r - SR0 * std_r) / se_mean_hac
    p_lo = 2 * (1 - norm.cdf(np.abs(t_lo)))

    # -- Bonferroni factor
    valid_days  = sharpe.notna().sum()
    Meff = 1.3          # 実効独立度（典型的に 1.2〜1.6 に安定）
    bonf_factor = math.ceil(valid_days / T) * Meff

    p_jkm_bonf = (p_jkm * bonf_factor).clip(0, 1.0)
    p_lo_bonf  = (p_lo  * bonf_factor).clip(0, 1.0)

    rej_jkm = f"reject_jkm_{alpha}"
    rej_lo  = f"reject_lo_{alpha}"

    out = df.copy()
    out['Z_jkm']       = Z_jkm
    out['p_jkm']       = p_jkm
    out['p_jkm_bonf']  = p_jkm_bonf
    out[rej_jkm]       = p_jkm < alpha
    out[f'{rej_jkm}_bonf'] = p_jkm_bonf < alpha

    out['t_lo']        = t_lo
    out['p_lo']        = p_lo
    out['p_lo_bonf']   = p_lo_bonf
    out[rej_lo]        = p_lo < alpha
    out[f'{rej_lo}_bonf']  = p_lo_bonf < alpha

    return out


def minp_fisher_score(df: pd.DataFrame,
                      p_column: str,
                      T: int = 30,
                      eps: float = 1e-12):
    """
    Hybrid Tippett + Fisher test.
        1. Split `p_column` into non-overlapping blocks of length T.
        2. Block statistic = min(p)  (Tippett)
        3. Global Fisher statistic  S = -2 Σ ln(min_p)
           Under H0 and independent blocks:  S ~ χ²(2B).

    -------
    global_p     : float, Fisher p-value across blocks.
    fisher_stat  : float, Fisher statistic across blocks.
    """
    # -------- validation -----------------------------------------------------
    if not p_column.startswith('p_'):
        raise ValueError(f"'{p_column}' は p値列に見えません（'p_' で始まる必要あり）。")
    if p_column not in df.columns:
        raise ValueError(f"DataFrame に列 '{p_column}' が存在しません。")

    p_series = df[p_column].astype(float)
    block_id = np.arange(len(p_series)) // T

    # -------- Tippett per block ---------------------------------------------
    min_p = p_series.groupby(block_id).min().clip(eps, None)
    block_stat = -2 * np.log(min_p)            # Fisher 部分統計

    # -------- Fisher across blocks ------------------------------------------
    fisher_stat  = block_stat.sum()
    df_global    = 2 * len(block_stat)         # 自由度 2B
    global_p     = chi2.sf(fisher_stat, df=df_global)

    return global_p


def calculate_sharpe_reliability(rebuilded_pnl, T=15, SR0=0.5, vault_name="Unknown"):
    """
    Calculate Sharpe ratio reliability metrics from rebuilded PnL data.
    
    :param rebuilded_pnl: List of cumulative PnL values ($).
    :param T: Window size for rolling tests (default: 15).
    :param SR0: Null hypothesis Sharpe ratio (default: 0.5).
    :param vault_name: Name of vault for debugging (default: "Unknown").
    :return: Dictionary with reliability metrics.
    """
    if len(rebuilded_pnl) < T + 1:
        return {
            "Sharpe Reliability": 1.0,  # High p-value indicates low reliability
            "JKM Test P-value": 1.0,
            "Lo Test P-value": 1.0,
            "Fisher Score": 1.0
        }
    
    try:
        # Calculate returns from rebuilded PnL
        returns = [rebuilded_pnl[i] / rebuilded_pnl[i - 1] - 1 
                  for i in range(1, len(rebuilded_pnl)) 
                  if rebuilded_pnl[i - 1] != 0]
        
        if len(returns) < T:
            return {
                "Sharpe Reliability": 1.0,
                "JKM Test P-value": 1.0,
                "Lo Test P-value": 1.0,
                "Fisher Score": 1.0
            }
        
        # Check for NaN or infinite values
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            return {
                "Sharpe Reliability": 1.0,
                "JKM Test P-value": 1.0,
                "Lo Test P-value": 1.0,
                "Fisher Score": 1.0
            }
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'returns': returns})
        
        # Perform rolling Sharpe tests
        test_results = rolling_sharpe_tests_indep(df, SR0=SR0, T=T)
        
        # Calculate Fisher scores for both tests
        fisher_jkm = minp_fisher_score(test_results, 'p_jkm_bonf', T=T)
        fisher_lo = minp_fisher_score(test_results, 'p_lo_bonf', T=T)
        
        # Get average p-values for individual tests (handle NaN)
        avg_jkm_p = test_results['p_jkm_bonf'].mean()
        avg_lo_p = test_results['p_lo_bonf'].mean()
        
        # Handle NaN values
        if pd.isna(avg_jkm_p):
            avg_jkm_p = 1.0
        if pd.isna(avg_lo_p):
            avg_lo_p = 1.0
        if pd.isna(fisher_jkm):
            fisher_jkm = 1.0
        if pd.isna(fisher_lo):
            fisher_lo = 1.0
        
        # Use the more conservative (higher) p-value as the reliability score
        reliability_score = max(fisher_jkm, fisher_lo)
        
        return {
            "Sharpe Reliability": round(reliability_score, 4),
            "JKM Test P-value": round(avg_jkm_p, 4),
            "Lo Test P-value": round(avg_lo_p, 4),
            "Fisher Score": round(min(fisher_jkm, fisher_lo), 4)
        }
        
    except Exception as e:
        # Return default values if calculation fails
        return {
            "Sharpe Reliability": 1.0,
            "JKM Test P-value": 1.0,
            "Lo Test P-value": 1.0,
            "Fisher Score": 1.0
        }
