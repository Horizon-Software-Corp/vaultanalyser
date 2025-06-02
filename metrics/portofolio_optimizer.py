import pandas as pd
import numpy as np
import riskfolio as rf


def optimize_portfolio(
    df_rets: pd.DataFrame,
    *,
    rf_rate: float = 0.0,
    model: str = "Classic",  # ["Classic", "BL","FM", "BL_FM", "NCO"]
    risk_measure: str = "MV",  # ["MV", "CVaR", "CDaR", "EVaR", ...
    objective: str = "Sharpe",  # "Sharpe", "MinRisk", "MaxReturn", …
    is_short: bool = True,
    max_leverage: float = 1,  # total position size limit
    covariance: str = "ledoit",  # "ledoit"(n <~ T), "hist"(n << T), "ewma", …
    solver: str = "ECOS",  # any CVXPY-compatible solver
    l2_reg: float = 0.0,  # λ for ℓ2‐regularisation
    budget: float = 1.0  # Σwᵢ = budget  (use 0 for dollar-neutral)
) -> pd.Series:
    """
    最適ウェイトを返すユーティリティ。

    Parameters
    ----------
    df_rets : pd.DataFrame
        列が資産、行が時系列リターン (相対値)。
    rf_rate : float
        年率換算した無リスク金利。
    model, risk_measure, objective :
        Riskfolio-Lib の引数をそのまま渡す。
    is_short : bool
        True で空売りを許可（ウェイト < 0 可）。デフォルトはロングオンリー。
    max_leverage : float | None
        Σ|wᵢ| の上限。None で無制限。
    weight_bounds : (lb, ub) | None
        個別資産のウェイト下限・上限。None なら is_short に応じて自動設定。
    covariance : str
        共分散推定法。 "ledoit" (Shrinkage) が高速で頑健。
    solver : str
        CVXPY が認識するソルバ名。
    l2_reg : float
        ℓ2 正則化係数。過剰フィッティング抑制用。
    budget : float
        Σwᵢ を何に合わせるか。1 ならフルインベスト、0 ならドルニュートラル等。

    Returns
    -------
    pd.Series
        index = 資産名、values = 最適ウェイト
    """
    # 1) 前処理
    returns = df_rets.dropna(how="all").copy()

    # 2) Portfolio オブジェクト生成
    port = rf.Portfolio(returns=returns)

    # 3) モーメント推定
    port.asset_stats(method_mu="hist", method_cov=covariance)

    # 4) 個別ウェイト境界を決定
    if is_short:
        # 対称的な上下限。必要に応じて調整してください
        weight_bounds = (-max_leverage, max_leverage)
    else:
        weight_bounds = (0.0, max_leverage)

    # 5) 最適化の実行
    w = port.optimization(
        model=model,
        risk_measure=risk_measure,
        obj=objective,
        rf=rf_rate,
        l=l2_reg,
        weight_bounds=weight_bounds,
        short=is_short,  # Riskfolio ≥0.5 では “short” 引数が追加
        budget=budget,
        solver=solver,
    )

    w = w.reindex(df_rets.columns).fillna(0.0)  # 列順を元 DataFrame に合わせる

    # 6) レバレッジ制限（|w| の和）を後段で調整する簡易版
    if max_leverage is not None:
        lev = w.abs().sum()
        if lev > max_leverage:  # 超過していればスケールダウン
            w *= max_leverage / lev

    return w
