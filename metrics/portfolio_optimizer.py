import pandas as pd
import numpy as np
import riskfolio as rf
import cvxpy as cp


def optimize_portfolio(
    df_rets: pd.DataFrame,
    *,
    rf_rate: float = 0.09,
    model: str = "Classic",  # ["Classic", "BL","FM", "BL_FM", "NCO"]
    risk_measure: str = "MV",  # ["MV", "CVaR", "CDaR", "EVaR", ...
    objective: str = "Sharpe",  # "Sharpe", "MinRisk", "MaxReturn", …
    is_short: bool = True,
    max_leverage: float = 1,  # total position size limit
    covariance: str = "ledoit",  # "ledoit"(n <~ T), "hist"(n << T), "ewma", …
    l2_reg: float = 0.0,  # λ for ℓ2‐regularisation
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
    covariance : str
        共分散推定法。 "ledoit" (Shrinkage) が高速で頑健。
    l2_reg : float
        ℓ2 正則化係数。過剰フィッティング抑制用。

    Returns
    -------
    pd.Series
        index = 資産名、values = 最適ウェイト
    """
    print(f"Optimizing portfolio with leverage:{max_leverage}, short: {is_short}")
    rf_rate = rf_rate * ((df_rets.index[1] - df_rets.index[0]) / pd.Timedelta(days=365))

    port = rf.Portfolio(returns=df_rets)

    port.assets_stats(method_mu="hist", method_cov=covariance)
    # cov = df_rets_scaled.cov().values  # NumPy array で共分散行列を取得
    # eigs = np.linalg.eigvalsh(cov)  # 固有値を小→大の順で返す
    # print(f"Eigenvalues: {eigs}")

    if is_short:
        # 1) Riskfolio で推定した μ と Σ を取得
        mu = port.mu.values  # shape = (n,)
        Sigma = port.cov.values  # shape = (n, n)
        n = len(mu)

        # 2) 変数と制約
        w_var = cp.Variable(n)

        constraints = []
        # (a) L¹ ノルム ≤ max_leverage
        if (max_leverage is not None) and np.isfinite(max_leverage):
            constraints += [cp.norm1(w_var) <= max_leverage]

        # (b) リスク指標 (ここでは MV: 分散 ≤ 1)
        if risk_measure.upper() == "MV":
            constraints += [cp.quad_form(w_var, Sigma) <= 1]
        else:
            raise NotImplementedError("cvxpy path supports risk_measure='MV' only.")

        # 3) 目的関数：Sharpe 比最大化 ＋ ℓ2 正則化
        if objective.capitalize() != "Sharpe":
            raise NotImplementedError("cvxpy path supports objective='Sharpe' only.")

        excess_ret = mu - rf_rate
        objective_cvx = cp.Maximize(
            excess_ret @ w_var - 0.5 * l2_reg * cp.sum_squares(w_var)
        )

        # 4) 求解
        prob = cp.Problem(objective_cvx, constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)

        if w_var.value is None:
            raise ValueError(
                "CVXPY optimization failed (problem infeasible or unbounded)"
            )

        w = pd.Series(w_var.value, index=df_rets.columns)
    else:
        port.sht = False  #
        port.upperlng = max_leverage  # デフォルトが1なのでmax_leverage以上にするかNone
        port.uppersht = None  # デフォルトが1なのでmax_leverage以上にするかNone
        port.budget = max_leverage  # デフォルトが1なのでmax_leverage以上にするかNone
        port.budgetsht = None  # デフォルトが0.2なのでmax_leverage以上にするかNone

        # 5) 最適化の実行
        w = port.optimization(
            model=model,
            rm=risk_measure,
            obj=objective,
            kelly=None,  # Kelly criterion は使わない
            rf=rf_rate,
            l=l2_reg,
        )

    w = (
        w.reindex(df_rets.columns).fillna(0.0).squeeze("columns")
    )  # 列順を元 DataFrame に合わせる
    w.name = "Weights"
    cutoff = max_leverage / len(df_rets.columns) * 0.05
    w = prune_small_weights(w, min_abs=cutoff)

    print(f"Optimized weights:\n{w}")

    return w


def prune_small_weights(
    w: pd.Series,
    *,
    min_abs: float | None = None,  # 絶対値がこれ未満なら 0
    top_k: int | None = None,  # 上位 k 本だけ残す
    renormalize: bool = True,  # 合計が元の Σw になるよう再スケール
) -> pd.Series:
    """
    w  : 重みベクトル (index=資産名, values=重み)
    min_abs : float
        例: 0.01 → |w|<1% を 0 に。
    top_k : int
        例: k=10 → |w| が大きい 10 本だけ残す。
        min_abs と同時指定した場合は「まず threshold → その後 top_k」を適用。
    renormalize : bool
        True にすると残ったウェイトを (元の Σw と同じ) 比率でスケールアップ。
    """
    w_new = w.copy()

    # 1) 絶対値しきい値カット
    if min_abs is not None:
        w_new = w_new.where(w_new.abs() >= min_abs, 0.0)

    # 2) 上位 k 本だけ残す
    if top_k is not None and top_k < (w_new != 0).sum():
        idx_keep = w_new.abs().nlargest(top_k).index
        w_new.loc[~w_new.index.isin(idx_keep)] = 0.0

    # 3) （オプション）スケールを元の合計に合わせる
    if renormalize:
        tot_before = w.sum()
        tot_after = w_new.sum()
        if tot_after != 0:
            w_new *= tot_before / tot_after

    return w_new
