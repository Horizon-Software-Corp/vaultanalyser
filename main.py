# source .venv/bin/activate
# streamlit run main.py

import json
import os
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import streamlit as st
from pprint import pprint
from typing import Dict, Tuple
from enum import StrEnum
import os
import shutil
from pathlib import Path


from utils.vaults import (
    fetch_vaults_data,
    get_all_vault_data,
)
from utils.users import (
    fetch_user_addresses,
    get_all_cached_user_data,
    get_user_stats,
    MAX_ADDRESSES_TO_FETCH,
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
data_type = DataType.USER  # Choose from [DataType.VAULT, DataType.USER]
data_range = DataRange.MONTH  # Choose from [DataRange.ALL_TIME, DataRange.MONTH]
is_debug = False  # Set to True for debugging mode
MAX_ITEMS = 100  # items are filtered based on Sharpe Ratio if more than MAX_ITEMS items are found
is_cache_used = False
is_renew_data = False
relevant_symbols = ["PUMP"]  # futuresはcoin名、spotは@number
is_disctation_trader = True
# ────────────────────────────────────────────────────────────────
# Parameters for weight calculation
# ────────────────────────────────────────────────────────────────
N_ITEMS = 10
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


def get_cache_date(CACHE_DATE_FILE):
    """Get the last cache date from file."""
    if not os.path.exists(CACHE_DATE_FILE):
        # UTC date
        cache_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(CACHE_DATE_FILE, "w") as f:
            f.write(cache_date)
    else:
        with open(CACHE_DATE_FILE, "r") as f:
            cache_date = f.read().strip()

    return cache_date


def copy_entire_dir_with_date(src_dir: str | Path, dst_parent: str | Path) -> None:
    """
    src_dir の内容（ファイル・フォルダを含む全て）を
    dst_parent/<basename_of_src>_YYYY-MM-DD へ丸ごとコピーします。

    Parameters:
      src_dir    コピー元ディレクトリのパス（文字列 or Path）
      dst_parent コピー先の親ディレクトリのパス（文字列 or Path）
    """
    src = Path(src_dir)
    dst_parent = Path(dst_parent)

    if not src.exists() or not src.is_dir():
        raise ValueError(f"src_dir が見つからないかディレクトリではありません: {src!r}")

    # コピー先の親ディレクトリを作成
    dst_parent.mkdir(parents=True, exist_ok=True)

    # コピー先のフルパスを組み立て
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dst = dst_parent / f"{src.name}_{date}"
    print(dst)

    # コピー実行（既存の dst があるとエラーになるので注意）
    shutil.copytree(src, dst)
    print(f"Copied: {src} → {dst}")
    # コピー完了後に元フォルダを削除
    shutil.rmtree(src)
    print(f"Removed original directory: {src}")


if is_renew_data:
    data_type_dir = data_type.lower()
    CACHE_DATE_FILE = f"./cache/{data_type_dir}/cache_date.txt"
    cache_date = get_cache_date(CACHE_DATE_FILE)
    SRC_DIR = f"./cache/{data_type_dir}/"
    DST_DIR = f"./cache_history/{data_type_dir}/"
    DST_PATH = DST_DIR + f"{data_type_dir}_{cache_date}"

    if not os.path.exists(DST_PATH) or pd.Timestamp.now() - pd.Timestamp(
        cache_date
    ) > pd.Timedelta(days=7):

        copy_entire_dir_with_date(SRC_DIR, DST_DIR)
    else:
        # 1日ずれ、みたいなので再取得しないため
        print(f"Cache for {data_type} on {cache_date} already exists at {DST_PATH}.")


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

# elif data_type == DataType.USER:


DATAFRAME_CACHE_FILE = f"./cache/{data_type.lower()}/dataframe.pkl"
DATAFRAME_RETS_CACHE_FILE = f"./cache/{data_type.lower()}/returns.pkl"

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
        identifier = user_data[identifier_name.lower()]

        if data_type == DataType.USER:
            traded_symbols = user_data["fills"]["traded_symbols"]
            if relevant_symbols and (
                set(traded_symbols) & set(relevant_symbols) == set()
            ):
                # print(
                #     f"No relevant symbols:{relevant_symbols} are traded by user {itr + 1}/{len(user_data_list)}: {identifier[:15]} in data_range: {data_range}. Skipping..."
                # )
                continue

        progress_bar.progress((itr + 1) / total_steps)
        status_text.text(
            f"Analyzing user {itr + 1}/{len(user_data_list)}: {identifier[:15]}..."
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
                    f"Total {identifier_name} PnL": float(
                        data_source_pnlHistory[-1][1]
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
                    fills_count = user_data["fills"]["count"]
                    max_fill_count = 2000
                    fills_count_estimated = (
                        max(
                            max_fill_count,
                            int(
                                fills_count
                                * pd.Timedelta(days=30).total_seconds()
                                / user_data["fills"]["last_fill_seconds"]
                            ),
                        )
                        if fills_count == max_fill_count
                        else fills_count
                    )

                    metrics["Estimated Fill Counts (30D)"] = fills_count_estimated

                    metrics["Perp Taker Fee (bips)"] = (
                        float(user_data["fees"]["userAddRate"]) * 10000
                    )
                    metrics["Spot Taker Fee (bips)"] = (
                        float(user_data["fees"]["userSpotAddRate"]) * 10000
                    )
                    metrics["Maker Volume (14D)"] = float(
                        user_data["fees"]["dailyUserVlm_14D"]["userAdd"]
                    )

                    metrics["Taker Volume (14D)"] = float(
                        user_data["fees"]["dailyUserVlm_14D"]["userCross"]
                    )

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


print(st.session_state)
if (is_cache_used or "init_done" in st.session_state) and os.path.exists(
    DATAFRAME_CACHE_FILE
):
    final_df = pd.read_pickle(DATAFRAME_CACHE_FILE)
    df_rets = pd.read_pickle(DATAFRAME_RETS_CACHE_FILE)
    cache_used = True
    st.info(f"📊 Using cached analysis data ({len(final_df)} users)")

else:
    if os.path.exists(DATAFRAME_CACHE_FILE):
        os.remove(DATAFRAME_CACHE_FILE)  # ← ここで一度だけキャッシュを削除
    st.session_state["init_done"] = True  # 以後の再実行では削除しない

    if data_type == DataType.VAULT:
        vaults = fetch_vaults_data()

        if is_debug:
            vaults = vaults[:100]
            st.title(f"[Debug mode!!!] only process {len(vaults)} {data_type}s")

        user_data_list = get_all_vault_data(vaults)

        st.info(f"🔄 Analyzing {len(user_data_list)} cached vaults...")
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
            lambda addr: f"https://app.hyperliquid.xyz/trade/{addr}"
        )

    elif data_type == DataType.USER:
        # Check if we have any cached user data

        user_data_list, failed_data_list = get_all_cached_user_data(
            st, is_debug=is_debug
        )
        # user_data_list = fetch_user_addresses()
        print(f"Number of cached users: {len(user_data_list)}")
        print(f"Number of failed users: {len(failed_data_list)}")

        user_stats = get_user_stats()

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leaderboard", user_stats["total_addresses"])
        with col2:
            st.metric("Fetched Users", user_stats["fetched_addresses"])
        with col3:
            st.metric("Failed Addresses", user_stats["failed_addresses"])
        with col4:
            st.metric("Cached Data Files", user_stats["cached_data_files"])
        st.markdown("---")

        if is_debug:
            st.info(f"Debug mode! only process {len(user_data_list)} {data_type}s")

        if len(user_data_list) < 19000 or len(failed_data_list) > 0:
            st.warning("⚠️ No user data found. Please download some user's data first.")

            # User input and button to process users
            col1, col2 = st.columns([1, 2])
            with col1:
                initial_users_to_fetch = st.number_input(
                    "Initial users to download",
                    min_value=1,
                    max_value=MAX_ADDRESSES_TO_FETCH,
                    value=30000,
                    step=1,
                    help="Number of users to download from leaderboard",
                    key="initial_users_input",
                )
            with col2:
                if st.button(
                    f"🔄 Download {initial_users_to_fetch} Users from Leaderboard"
                ):
                    with st.spinner("Downloading user data..."):
                        fetch_user_addresses(
                            max_addresses=initial_users_to_fetch,
                            addresses_force_fetch=failed_data_list,
                            show_progress=True,
                        )
                    st.rerun()

            st.stop()
            print("download completed")
            user_data_list, _ = get_all_cached_user_data(st, is_debug=is_debug)

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
            lambda addr: f"https://hypurrscan.io/address/{addr}"
        )

        # Display results
        st.subheader(f"Users analysed ({len(final_df)})")

        # Add process more users button with user input
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            # User input for number of users to process
            users_to_fetch = st.number_input(
                "Users to process",
                min_value=1,
                max_value=30000,
                value=MAX_ADDRESSES_TO_FETCH,
                step=1,
                help="Number of users to process from leaderboard",
            )
        with col2:
            if st.button(f"➕ Process {users_to_fetch} More Users"):
                with st.spinner("Processing more users..."):
                    new_data = fetch_user_addresses(
                        max_addresses=users_to_fetch, show_progress=True
                    )
                    if new_data:
                        # Clear cache to force reprocessing
                        if os.path.exists(DATAFRAME_CACHE_FILE):
                            os.remove(DATAFRAME_CACHE_FILE)
                        st.rerun()

    # Save to cache
    final_df.to_pickle(DATAFRAME_CACHE_FILE)
    df_rets.to_pickle(DATAFRAME_RETS_CACHE_FILE)
    st.toast(f"{data_type} analysis data cached!", icon="✅")


####################################
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

is_algo = not is_disctation_trader

# Organize sliders into rows of 3
sliders = [
    # from "https://api-ui.hyperliquid.xyz/info"
    {
        "label": f"Total {identifier_name} Value",
        "column": f"Total {identifier_name} Value",
        "default_min": 100 if is_algo else 0,
        "default_max": float("inf") if is_algo else 1000_000,
        "step": 10,
        "log_scale": True,
    },
    {
        "label": f"Total {identifier_name} PnL",
        "column": f"Total {identifier_name} PnL",
        "default_min": 10000,
        "default_max": float("inf") if is_algo else 1000_000,
        "step": 1,
        "log_scale": True,
    },
    {
        "label": "Days from Return(Estimate)",
        "column": "Days from Return(Estimate)",
        "default_min": 90 if data_range == DataRange.ALL_TIME else 30,
        "default_max": 10000,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": f"{sharpe_prefix} Sharpe Ratio",
        "column": f"{sharpe_prefix} Sharpe Ratio",
        "default_min": 0.1 if is_algo else -0.1,
        "default_max": 10,
        "step": 0.05,
        "log_scale": False,
    },
    {
        "label": f"{sharpe_prefix} Sortino Ratio",
        "column": f"{sharpe_prefix} Sortino Ratio",
        "default_min": 0.1 if is_algo else -0.1,
        "default_max": 10,
        "step": 0.05,
        "log_scale": False,
    },
    {
        "label": "Annualized Gain(simple) %/yr",
        "column": "Annualized Gain(simple) %/yr",
        "default_min": 20 if is_algo else -20,
        "default_max": 1000_000,
        "step": 1,
        "log_scale": True,
    },
    {
        "label": "Annualized Median Gain(compound) %/yr",
        "column": "Annualized Median Gain(compound) %/yr",
        "default_min": 0,
        "default_max": 1000_000,
        "step": 1,
        "log_scale": True,
    },
    {
        "label": "Max DD %",
        "column": "Max DD %",
        "default_min": 0,
        "default_max": 50 if is_algo else 100,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": "Rekt",
        "column": "Rekt",
        "default_min": 0,
        "default_max": 0 if is_algo else 100,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": "Followers",
        "column": "Act. Followers",
        "default_min": 0,
        "default_max": 1000_000,
        "step": 1,
        "log_scale": True,
    },
    {
        "label": "APR(30D) %",
        "column": "APR(30D) %",
        "default_min": 0 if is_algo else -100,
        "default_max": 1000,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": "TVL Leader fraction %",
        "column": "TVL Leader fraction %",
        "default_min": 0,
        "default_max": 100,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": "Estimated Fill Counts (30D)",
        "column": "Estimated Fill Counts (30D)",
        "default_min": 1,
        "default_max": float("inf") if is_algo else 1000,
        "step": 1,
        "log_scale": True,
    },
    {
        "label": "Perp Taker Fee (bips)",
        "column": "Perp Taker Fee (bips)",
        "default_min": -100,
        "default_max": 100,
        "step": 0.0001,
        "log_scale": False,
    },
    {
        "label": "Spot Taker Fee (bips)",
        "column": "Spot Taker Fee (bips)",
        "default_min": -100,
        "default_max": 100,
        "step": 0.0001,
        "log_scale": False,
    },
    {
        "label": "Maker Volume (14D)",
        "column": "Maker Volume (14D)",
        "default_min": 0,
        "default_max": float("inf") if is_algo else 10_000_000,
        "step": 100,
        "log_scale": True,
    },
    {
        "label": "Taker Volume (14D)",
        "column": "Taker Volume (14D)",
        "default_min": 0,
        "default_max": float("inf") if is_algo else 10_000_000,
        "step": 100,
        "log_scale": True,
    },
    # from "https://stats-data.hyperliquid.xyz/Mainnet/vaults"
    {
        "label": "TVL",
        "column": "Total Value Locked",
        "default_min": 10000,
        "default_max": float("inf") if is_algo else 1_000_000,
        "step": 10,
        "log_scale": True,
    },
    {
        "label": "Days Since",
        "column": "Days Since",
        "default_min": 90,
        "default_max": 1000,
        "step": 1,
        "log_scale": False,
    },
    {
        "label": "APR(7D) %",
        "column": "APR(7D) %",
        "default_min": 0 if is_algo else -100,
        "default_max": 1000,
        "step": 1,
        "log_scale": False,
    },
]


def slider_with_label(
    label, col, min_value, max_value, default_value, step, key, log_scale=False
):
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

    if log_scale:
        # symlogスケールの場合
        # symlogは0付近の値も適切に扱える
        # symlog(x) = arcsinh(x) = log(x + sqrt(x^2 + 1))
        symlog_min = np.arcsinh(min_value)
        symlog_max = np.arcsinh(max_value)
        symlog_default = np.arcsinh(default_value)

        # デフォルト値の範囲調整
        if symlog_default < symlog_min:
            symlog_default = symlog_min
        if symlog_default > symlog_max:
            symlog_default = symlog_max

        # symlogスケール用のスライダー
        # symlogスケールの範囲を分割して、段階的な値を作成
        num_steps = 100
        symlog_steps = np.linspace(symlog_min, symlog_max, num_steps + 1)

        # symlogの逆変換（sinh）を使用して実際の値に戻す
        actual_steps = [np.sinh(symlog_step) for symlog_step in symlog_steps]

        # デフォルト値に最も近いインデックスを見つける
        default_idx = min(
            range(len(actual_steps)), key=lambda i: abs(actual_steps[i] - default_value)
        )

        # select_sliderを使用して実際の値を表示
        selected_value = col.select_slider(
            label,
            options=actual_steps,
            value=actual_steps[default_idx],
            format_func=lambda x: "∞" if x == float("inf") else f"{x:,.0f}",
            label_visibility="hidden",
            key=key,
        )

        return selected_value
    else:
        # 通常のリニアスケール
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


def range_slider_with_label(
    label,
    col,
    min_value,
    max_value,
    default_min,
    default_max,
    step,
    key,
    log_scale=False,
):
    """Create a range slider with a custom centered title."""
    col.markdown(
        f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True
    )
    if not min_value < max_value:
        col.markdown(
            f"<p style='text-align: center;'>No choice available ({min_value} for all)</p>",
            unsafe_allow_html=True,
        )
        return None, None

    if log_scale:
        # symlogスケールの場合
        # symlogは0付近の値も適切に扱える
        # symlog(x) = arcsinh(x) = log(x + sqrt(x^2 + 1))
        symlog_min = np.arcsinh(min_value)
        symlog_max = (
            np.arcsinh(max_value) if max_value != float("inf") else np.arcsinh(10**10)
        )
        symlog_default_min = np.arcsinh(default_min)
        symlog_default_max = (
            np.arcsinh(default_max) if default_max != float("inf") else symlog_max
        )

        # デフォルト値の範囲調整
        if symlog_default_min < symlog_min:
            symlog_default_min = symlog_min
        if symlog_default_max > symlog_max:
            symlog_default_max = symlog_max
        if symlog_default_min > symlog_default_max:
            symlog_default_min, symlog_default_max = (
                symlog_default_max,
                symlog_default_min,
            )

        # symlogスケール用のスライダー
        # symlogスケールの範囲を分割して、段階的な値を作成
        num_steps = 100
        symlog_steps = np.linspace(symlog_min, symlog_max, num_steps + 1)

        # symlogの逆変換（sinh）を使用して実際の値に戻す
        actual_steps = [np.sinh(symlog_step) for symlog_step in symlog_steps]

        # デフォルト値に最も近いインデックスを見つける
        default_min_idx = min(
            range(len(actual_steps)), key=lambda i: abs(actual_steps[i] - default_min)
        )
        default_max_idx = min(
            range(len(actual_steps)), key=lambda i: abs(actual_steps[i] - default_max)
        )

        # select_sliderを使用して実際の値を表示（範囲選択）
        selected_values = col.select_slider(
            label,
            options=actual_steps,
            value=(actual_steps[default_min_idx], actual_steps[default_max_idx]),
            format_func=lambda x: "∞" if x == float("inf") else f"{x:,.0f}",
            label_visibility="hidden",
            key=key,
        )

        return selected_values
    else:
        # 通常のリニアスケール
        if default_min < min_value:
            default_min = min_value
        if default_max > max_value:
            default_max = max_value
        if default_min > default_max:
            default_min, default_max = default_max, default_min

        return col.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=(default_min, default_max),
            step=step,
            label_visibility="hidden",
            key=key,
        )


for i in range(0, len(sliders), 3):
    cols = st.columns(3)
    for slider, col in zip(sliders[i : i + 3], cols):
        column = slider["column"]
        if column in filtered_df.columns:
            # Range slider (all sliders are now range sliders)
            values = range_slider_with_label(
                slider["label"],
                col,
                min_value=float(filtered_df[column].min()),
                max_value=float(filtered_df[column].max()),
                default_min=float(slider["default_min"]),
                default_max=float(slider["default_max"]),
                step=float(slider["step"]),
                key=f"slider_{column}",
                log_scale=slider.get("log_scale", False),
            )
            if values is not None and values != (None, None):
                min_val, max_val = values
                filtered_df = filtered_df[
                    (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
                ]


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

# validatorを除く
filtered_weight_df = filtered_weight_df[filtered_weight_df[sort_col_weight] < 2]

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
