# source .venv/bin/activate
# streamlit run main.py

import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from pprint import pprint
from typing import Dict, Tuple
from enum import StrEnum

from hyperliquid.vaults import (
    fetch_vault_details,
    fetch_vaults_data,
    get_all_vault_data,
)
from hyperliquid.users import (
    process_user_addresses,
    get_all_cached_user_data,
    calculate_days_since_start,
    get_user_stats,
    MAX_ADDRESSES_TO_PROCESS,
)
from metrics.sharpe_reliability import calculate_sharpe_reliability
from metrics.metric_styler import MetricsStyler
from metrics.portfolio_optimizer import optimize_portfolio


# ────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────
class DataType(StrEnum):
    VAULT = "Vault"
    USER = "User"


class DataRange(StrEnum):
    ALL_TIME = "allTime"
    MONTH = "month"


# ────────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────────

data_type = DataType.VAULT  # Choose from [DataType.VAULT, DataType.USER]
data_range = DataRange.MONTH  # Choose from [DataRange.ALL_TIME, DataRange.MONTH]
is_debug = False  # Set to True for debugging mode
MAX_ITEMS = 100  # items are filtered based on Sharpe Ratio if more than MAX_ITEMS items are found


# ────────────────────────────────────────────────────────────────
# Parameters for weight calculation
# ────────────────────────────────────────────────────────────────
N_ITEMS = 5
sort_col_weight = "Sharpe Ratio"  # Column to sort by
rf_rate = 0.09  # Annualized risk-free rate
is_short = False  # BTC, ETHなどの価格の系列を入れ、そこだけshort許可してもいいかも
max_leverage = 1


# ────────────────────────────────────────────────────────────────

if data_type == DataType.VAULT:
    page_icon = "📊"
    identifier_name = "Name"
elif data_type == DataType.USER:
    page_icon = "👤"
    identifier_name = "Address"

# Page config
st.set_page_config(
    page_title=f"HyperLiquid {data_type} Analyser", page_icon=page_icon, layout="wide"
)

# Title and description
st.title(f"HyperLiquid {data_type} Analyser")
st.caption(f"🏦 {data_type} Analysis Mode")

if data_type == DataType.VAULT:
    # Update time display
    try:
        with open("./cache/vaults_cache.json", "r") as f:
            cache = json.load(f)
            last_update = datetime.fromisoformat(cache["last_update"])
            st.caption(f"🔄 Last update: {last_update.strftime('%Y-%m-%d %H:%M')} UTC")
    except (FileNotFoundError, KeyError, ValueError):
        st.warning("⚠️ Cache not found. Data will be fetched fresh.")
    st.markdown("---")  # Add a separator line

    DATAFRAME_CACHE_FILE = "./cache/dataframe.pkl"

elif data_type == DataType.USER:
    user_stats = get_user_stats()

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Leaderboard", user_stats["total_addresses"])
    with col2:
        st.metric("Processed Users", user_stats["processed_addresses"])
    with col3:
        st.metric("Failed Addresses", user_stats["failed_addresses"])
    with col4:
        st.metric("Cached Data Files", user_stats["cached_data_files"])
    st.markdown("---")
    DATAFRAME_CACHE_FILE = "./user_cache/user_dataframe.pkl"


# Layout for 3 columns


if data_range == DataRange.ALL_TIME:
    data_index = 3
    sampling_days = 7
    resample_rule = "W-MON"
    sharpe_prefix = "Weekly"
    separation_params = {"h": 8, "SR0": 0.1, "LT": 16}

elif data_range == DataRange.MONTH:
    data_index = 2
    sampling_days = 1
    resample_rule = "D"
    sharpe_prefix = "Daily"
    separation_params = {"h": 8, "SR0": 0.1, "LT": None}
else:
    raise ValueError(f"Invalid data_range: {data_range}. Choose 'allTime' or 'month'.")


def new_process_user_data_for_analysis(user_data_list):

    # Process vault details from cache
    progress_bar = st.progress(0)
    status_text = st.empty()
    indicators = []
    rets = []
    total_steps = len(user_data_list)

    for itr, user_data in enumerate(user_data_list):

        progress_bar.progress((itr + 1) / total_steps)
        status_text.text(
            f"Analyzing user {itr + 1}/{len(user_data_list)}: {user_data[identifier_name.lower()][:15]}..."
        )

        portfolio_data = user_data["portfolio"]

        for period_data in portfolio_data:
            if period_data[0] == data_range:
                if data_type == DataType.VAULT:
                    leader_eq = next(
                        f for f in user_data["followers"] if f["user"] == "Leader"
                    )["vaultEquity"]
                    leader_eq = float(leader_eq)
                    tvl = sum(float(f["vaultEquity"]) for f in user_data["followers"])
                    if tvl == 0:
                        continue
                    leader_fraction = leader_eq / tvl  # 0.0–1.0

                    check_an_identifier = None  # "HLP", ...
                elif data_type == DataType.USER:
                    check_an_identifier = None  # "0x1234...", ...

                identifier = user_data[identifier_name.lower()]

                # Check a vault or an address
                if check_an_identifier and identifier != check_an_identifier:
                    continue

                data_source_pnlHistory = period_data[1].get("pnlHistory", [])
                data_source_accountValueHistory = period_data[1].get(
                    "accountValueHistory", []
                )
                rebuilded_pnl = []
                final_capital_virtuals = []
                used_capitals = []
                returns = []
                bankrupts = []
                transferIns = []
                timestamps = []

                balance = start_balance_amount = 1000000
                nb_rekt = 0
                last_rekt_idx = -10

                # Recalculate the balance without considering deposit movements
                is_print_bunkrupt = False
                is_neglect_bunkrupt = False

                for idx, value in enumerate(data_source_pnlHistory):
                    if idx == 0:
                        final_capital_virtuals.append(0)  # = final_capital
                        used_capitals.append(None)  # = initial_capital
                        returns.append(0)
                        rebuilded_pnl.append(balance)
                        bankrupts.append(0)
                        transferIns.append(0)
                        timestamps.append(data_source_pnlHistory[idx][0])
                        continue

                    # Capital at time T
                    initial_capital = float(
                        data_source_accountValueHistory[idx - 1][1]
                    )  # >= 0
                    final_capital = float(
                        data_source_accountValueHistory[idx][1]
                    )  # >= 0
                    pnl = float(data_source_pnlHistory[idx][1]) - float(
                        data_source_pnlHistory[idx - 1][1]
                    )
                    transferIn = round(final_capital - initial_capital - pnl, 1)
                    transferIns.append(transferIn)
                    used_capital = max(initial_capital, initial_capital + transferIn)
                    used_capitals.append(used_capital)
                    final_capital_virtual = initial_capital + pnl
                    final_capital_virtuals.append(final_capital_virtual)

                    bankrupt = 0
                    if initial_capital == 0:
                        pass
                    elif (
                        final_capital_virtual <= initial_capital * 0.01
                        or final_capital_virtual <= initial_capital * 0.1
                        and final_capital_virtual < 10
                    ):
                        # 　厳密には、initial_capitalよりもだいぶ大きい額をtransferInした直後にちょっと損失が出る場合も含まれてしまうが、無視する
                        is_print_bunkrupt = True
                        if last_rekt_idx + 1 != idx:
                            nb_rekt = nb_rekt + 1
                            balance = start_balance_amount
                            bankrupt = 1
                            if is_neglect_bunkrupt:
                                rebuilded_pnl = [start_balance_amount] * len(balance)
                        last_rekt_idx = idx
                        # continue

                    # Gain/loss ratio
                    if used_capital == 0:
                        return_ = 0
                    else:
                        return_ = max(-1, pnl / used_capital)
                    returns.append(return_)

                    # Verify timestamp consistency
                    if (
                        data_source_pnlHistory[idx][0]
                        != data_source_accountValueHistory[idx][0]
                    ):
                        print("Just to check, normally not happening")
                        exit()

                    # Update the simulated balance
                    if bankrupt:
                        pass
                    else:
                        balance = round(balance * (1 + return_), 2)
                    rebuilded_pnl.append(balance)
                    bankrupts.append(bankrupt)
                    timestamps.append(data_source_pnlHistory[idx][0])

                index_ts = pd.to_datetime(
                    timestamps, unit="ms", origin="unix", utc=True
                )
                ret = pd.Series(
                    returns,
                    index=index_ts,
                    dtype=float,
                )
                is_resample = True
                if is_resample:
                    # daily resampling by close
                    ret = (1 + ret).resample(
                        resample_rule, label="left", closed="left"
                    ).prod() - 1
                    ret.fillna(0, inplace=True)  # スカスカなので

                # pd.set_option("display.max_rows", 1000)  # Show
                # if ret.isna().any():
                #     raise ValueError(
                #         f"Vault {vault['Name']} has NaN values in returns.:{ret}"
                #     )

                if ret.std() == 0:
                    continue

                if index_ts[0] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=28):
                    continue

                bankrupt = ret <= -1
                valid = ~bankrupt
                # warnings.filterwarnings(
                #     "error", message="divide by zero encountered in log1p"
                # )
                log_ret = pd.Series(-np.inf, index=ret.index, dtype=float)
                log_ret.loc[valid] = np.log1p(ret.loc[valid])
                cum_ret = np.exp(log_ret.cumsum())
                dd = cum_ret / np.maximum.accumulate(cum_ret) - 1

                if check_an_identifier:
                    pnls = [float(value[1]) for value in data_source_pnlHistory]
                    df = (
                        pd.DataFrame(
                            {
                                "PnL": pnls,
                                "UsedCap": used_capitals,
                                "TransfIn": transferIns,
                            },
                            index=index_ts,
                        )
                        .resample(resample_rule, label="left", closed="left")
                        .sum()
                    )
                    # print(pnls)

                    df2 = pd.DataFrame(
                        {
                            "Ret %": ret * 100,
                            "CumRet %": cum_ret * 100,
                            "DD %": dd * 100,
                        }
                    )
                    df = pd.concat([df, df2], axis=1)
                    df = df.astype(float)

                    pd.set_option("display.float_format", "{:.2f}".format)
                    pd.set_option("display.max_rows", 1000)  # Show all rows
                    print(df)

                metrics = {
                    f"Total {identifier_name} Value": float(
                        data_source_accountValueHistory[-1][1]
                    ),
                    "Days from Return(Estimate)": len(ret) * sampling_days,
                    f"{sharpe_prefix} Sharpe Ratio": ret.mean() / ret.std(),
                    f"{sharpe_prefix} Sortino Ratio": (
                        ret.mean() / ret[ret > 0].std()
                        if len(ret[ret > 0]) >= 2 and ret[ret > 0].std() != 0
                        else 1000000
                    ),
                    # "Daily Gain %": ret.mean() * 100,
                    "Gain(simple) %": ret.sum() * 100,
                    "Annualized Gain(simple) %/yr": ret.mean()
                    / sampling_days
                    * 365
                    * 100,
                    # "Gain(compound) %": (
                    #     (cum_ret - 1) * 100 if ret.min() > -1 else -100
                    # ),
                    "Annualized Median Gain(compound) %/yr": (
                        ret.mean() - 1 / 2 * ret.var()
                    )
                    / sampling_days
                    * 365
                    * 100,
                    "Max DD %": dd.min() * (-1) * 100,
                    "Rekt": nb_rekt,
                }
                if data_type == DataType.VAULT:
                    nb_followers = sum(
                        1
                        for f in user_data["followers"]
                        if float(f["vaultEquity"]) >= 0.01
                    )
                    metrics.update(
                        {
                            "TVL Leader fraction %": round(leader_fraction * 100, 2),
                            "Act. Followers": nb_followers,
                            "APR(30D) %": float(user_data["apr"]),
                        }
                    )

                elif data_type == DataType.USER:
                    pass

                # Calculate Sharpe reliability metrics

                reliability_metrics = calculate_sharpe_reliability(
                    ret.values, **separation_params  # monthlyは30本しかない
                )
                reliability_metrics_keys = reliability_metrics.keys()

                for key in reliability_metrics_keys:
                    metrics[key] = reliability_metrics[key]
                # Unpacks the metrics dictionary
                metrics[identifier_name] = identifier

                indicators.append(metrics)
                rets.append(ret)
                break

    progress_bar.empty()
    status_text.empty()
    return indicators, rets


cache_used = False
# try:
#     final_df = pd.read_pickle(DATAFRAME_CACHE_FILE)
#     cache_used = True
#     st.info(f"📊 Using cached analysis data ({len(final_df)} users)")
# except (FileNotFoundError, KeyError, ValueError):
#     pass

if not cache_used:
    if data_type == DataType.VAULT:
        vaults = fetch_vaults_data()

        if is_debug:
            vaults = vaults[:100]
            st.title(f"[Debug mode!!!] only process {len(vaults)} {data_type}s")

        user_data_list = get_all_vault_data(vaults)

        st.info(f"🔄 Analyzing {len(user_data_list)} cached users...")
        indicators, rets = new_process_user_data_for_analysis(user_data_list)
        indicators_df = pd.DataFrame(indicators)
        pd.set_option("display.max_rows", 1000)  # Show all rows
        # print(pd.DataFrame(pd.DataFrame(rets).T).loc[:, 0])
        df_rets = pd.DataFrame(rets).T.dropna(axis=0, how="any")

        vaults_df = pd.DataFrame(vaults)
        vaults_df["APR(7D) %"] = vaults_df["APR(7D) %"].astype(float)
        del vaults_df["Leader"]

        final_df = vaults_df.merge(indicators_df, on="Name", how="right")

        final_df["Link"] = final_df["Vault"].apply(
            lambda vault: f"https://app.hyperliquid.xyz/vaults/{vault}"
        )

    elif data_type == DataType.USER:
        # Check if we have any cached user data

        user_data_list = get_all_cached_user_data(st, is_debug=is_debug)

        if is_debug:
            st.info(f"Debug mode! only process {len(user_data_list)} {data_type}s")

        if not user_data_list:
            st.warning("⚠️ No user data found. Please process some users first.")

            # User input and button to process users
            col1, col2 = st.columns([1, 2])
            with col1:
                initial_users_to_process = st.number_input(
                    "Initial users to process",
                    min_value=1,
                    max_value=20000,
                    value=MAX_ADDRESSES_TO_PROCESS,
                    step=1,
                    help="Number of users to process from leaderboard",
                    key="initial_users_input",
                )
            with col2:
                if st.button(
                    f"🔄 Process {initial_users_to_process} Users from Leaderboard"
                ):
                    with st.spinner("Processing users..."):
                        process_user_addresses(
                            max_addresses=initial_users_to_process, show_progress=True
                        )
                    st.rerun()

            st.stop()

        # Process the cached data
        st.info(f"🔄 Processing {len(user_data_list)} cached users...")
        indicators, rets = new_process_user_data_for_analysis(user_data_list)

        # print(pd.DataFrame(pd.DataFrame(rets).T).loc[:, 0])

        if not indicators:
            st.error("❌ No valid user data could be processed.")
            st.stop()

        # Create DataFrame
        final_df = pd.DataFrame(indicators)

        df_rets = pd.DataFrame(rets).T
        # for col in df_rets.columns:
        #     s_col = df_rets[col]
        #     if s_col.isna().any():
        #         pass
        #         # print(f"Warning: {col} has NaN values: {s_col.isna()}")
        df_rets = df_rets.dropna(axis=0, how="any")
        print(df_rets)

        # Add a column with clickable links to HyperLiquid
        final_df["Link"] = final_df["Address"].apply(
            lambda addr: f"https://app.hyperliquid.xyz/trade/{addr}"
        )
        # Display results
        st.subheader(f"Users analysed ({len(final_df)})")

        # Add process more users button with user input
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            # User input for number of users to process
            users_to_process = st.number_input(
                "Users to process",
                min_value=1,
                max_value=20000,
                value=MAX_ADDRESSES_TO_PROCESS,
                step=1,
                help="Number of users to process from leaderboard",
            )
        with col2:
            if st.button(f"➕ Process {users_to_process} More Users"):
                with st.spinner("Processing more users..."):
                    new_data = process_user_addresses(
                        max_addresses=users_to_process, show_progress=True
                    )
                    if new_data:
                        # Clear cache to force reprocessing
                        if os.path.exists(DATAFRAME_CACHE_FILE):
                            os.remove(DATAFRAME_CACHE_FILE)
                        st.rerun()


# Save to cache
final_df.to_pickle(DATAFRAME_CACHE_FILE)
st.toast(f"{data_type} analysis data cached!", icon="✅")

st.subheader(f"{data_type}s  ({len(final_df)})")
st.markdown(
    f"<h3 style='text-align: center;'>Filter by {identifier_name}</h3>",
    unsafe_allow_html=True,
)

# fileter by identifier
filtered_df = final_df.copy()

filter_ = st.text_input(
    f"{identifier_name} Filter",
    "",
    placeholder=f"Enter {identifier_name} separated by ',' to filter",
    key=f"{identifier_name}_filter",
)

# Apply the filter
if filter_.strip():  # Check that the filter is not empty
    list_ = [item.strip() for item in filter_.split(",")]  # List of names to search for
    pattern = "|".join(list_)  # Create a regex pattern with logical "or"
    filtered_df = filtered_df[
        filtered_df[identifier_name].str.contains(
            pattern, case=False, na=False, regex=True
        )
    ]


# Organize sliders into rows of 3
sliders = [
    # from "https://api-ui.hyperliquid.xyz/info"
    {
        "label": f"Min Total {identifier_name} Value",
        "column": f"Total {identifier_name} Value",
        "max": False,
        "default": 100,
        "step": 10,
    },
    {
        "label": "Min Days from Return(Estimate)",
        "column": "Days from Return(Estimate)",
        "max": False,
        "default": 90 if data_range == DataRange.ALL_TIME else 30,
        "step": 1,
    },
    {
        "label": f"Min {sharpe_prefix} Sharpe Ratio",
        "column": f"{sharpe_prefix} Sharpe Ratio",
        "max": False,
        "default": 0.1,
        "step": 0.05,
    },
    {
        "label": f"Min {sharpe_prefix} Sortino Ratio",
        "column": f"{sharpe_prefix} Sortino Ratio",
        "max": False,
        "default": 0.1,
        "step": 0.05,
    },
    {
        "label": "Min Annualized Gain(simple) %/yr",
        "column": "Annualized Gain(simple) %/yr",
        "max": False,
        "default": 20,
        "step": 1,
    },
    {
        "label": "Min Annualized Median Gain(compound) %/yr",
        "column": "Annualized Median Gain(compound) %/yr",
        "max": False,
        "default": 0,
        "step": 1,
    },
    {
        "label": "Max DD %",
        "column": "Max DD %",
        "max": True,
        "default": 50,
        "step": 1,
    },
    {
        "label": "Max Rekt",
        "column": "Rekt",
        "max": True,
        "default": 0,
        "step": 1,
    },
    {
        "label": "Min Followers",
        "column": "Act. Followers",
        "max": False,
        "default": 0,
        "step": 1,
    },
    {
        "label": "Min APR(30D) %",
        "column": "APR(30D) %",
        "max": False,
        "default": 0,
        "step": 1,
    },
    {
        "label": "Min TVL Leader fraction %",
        "column": "TVL Leader fraction %",
        "max": False,
        "default": 0,
        "step": 1,
    },
    # from "https://stats-data.hyperliquid.xyz/Mainnet/vaults"
    {
        "label": "Min TVL",
        "column": "Total Value Locked",
        "max": False,
        "default": 10000,
        "step": 10,
    },
    {
        "label": "Min Days Since",
        "column": "Days Since",
        "max": False,
        "default": 90,
        "step": 1,
    },
    {
        "label": "Min APR(7D)",
        "column": "APR(7D) %",
        "max": False,
        "default": 0,
        "step": 1,
    },
]


def slider_with_label(label, col, min_value, max_value, default_value, step, key):
    """Create a slider with a custom centered title."""
    col.markdown(
        f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True
    )
    if not min_value < max_value:
        col.markdown(
            f"<p style='text-align: center;'>No choice available ({min_value} for all)</p>",
            unsafe_allow_html=True,
        )
        return None

    if default_value < min_value:
        default_value = min_value

    if default_value > max_value:
        default_value = max_value

    return col.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        label_visibility="hidden",
        key=key,
    )


for i in range(0, len(sliders), 3):
    cols = st.columns(3)
    for slider, col in zip(sliders[i : i + 3], cols):
        column = slider["column"]
        if column in filtered_df.columns:
            # 　手で指定した値
            value = slider_with_label(
                slider["label"],
                col,
                min_value=float(filtered_df[column].min()),
                max_value=float(filtered_df[column].max()),
                default_value=float(slider["default"]),
                step=float(slider["step"]),
                key=f"slider_{column}",
            )
            if value is not None:
                pass
                # if slider["max"]:
                #     filtered_df = filtered_df[filtered_df[column] <= value]
                # else:
                #     filtered_df = filtered_df[filtered_df[column] >= value]


# -────────────────────────────────────────────────────────────
# weights calculation
# -────────────────────────────────────────────────────────────
if sort_col_weight == "Sharpe Ratio":
    sort_col_weight = f"{sharpe_prefix} Sharpe Ratio"
    ascending = False
elif sort_col_weight == "Sortino Ratio":
    sort_col_weight = f"{sharpe_prefix} Sortino Ratio"
    ascending = False
else:
    raise ValueError(
        f"Invalid sort_col_weight: {sort_col_weight}. Choose 'Sharpe Ratio' or 'Sortino Ratio'."
    )


filtered_weight_df = filtered_df.sort_values(
    by=sort_col_weight, ascending=ascending
)  # こっちはrowが資産名
# それぞれのretごとに全ての時間のデータを使って計算したsharpe ratio
# print("\n")
# print(filtered_weight_df[sort_col_weight])

idx_weights = filtered_weight_df.index[:N_ITEMS]
df_rets_weight = df_rets.loc[
    :, idx_weights
]  # df_retsは.Tしてあるので、colが資産名 -> 第二スライス

# dropna(axis=0, how="any") したあとのデータを使って計算したsharpe ratio
print("\n")
print("Sharpe Ratio of the top N items:")
print(df_rets_weight.mean() / df_rets_weight.std())  # これでシャープレシオが計算できる
print("\n")

# print("dropna")
# print(df_rets_weight)

weights = optimize_portfolio(
    df_rets_weight, rf_rate=rf_rate, is_short=is_short, max_leverage=max_leverage
)
weights_other = pd.Series(0.0, index=filtered_weight_df.index[N_ITEMS:], name="Weights")
weights_all = pd.concat([weights, weights_other], axis=0)

filtered_df = pd.concat([weights_all, filtered_weight_df], axis=1)

# -────────────────────────────────────────────────────────────
# Sort the DataFrame by Sharpe Ratio

sort_col = f"{sharpe_prefix} Sharpe Ratio"
filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)
orig_len = len(filtered_df)
# Display the table
filtered_df = filtered_df.iloc[:MAX_ITEMS]
if orig_len > MAX_ITEMS:
    st.title(
        f"{data_type} filtered ({orig_len} -> Top {MAX_ITEMS} by '{sort_col}' are shown.) "
    )
else:
    st.title(f"{data_type} filtered ({orig_len}) ")


styler = MetricsStyler(p_th=0.05, zmax=3).generate_style(
    filtered_df, final_df, data_range
)

st.dataframe(
    styler,
    use_container_width=True,
    height=(len(filtered_df) * 35) + 50,
    column_config={
        "Link": st.column_config.LinkColumn(
            f"{data_type} Link", display_text=f"{data_type} Link"
        )
    },
)
