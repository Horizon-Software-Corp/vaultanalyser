# source .venv/bin/activate
# streamlit run main.py

import json
import os
import time
from datetime import datetime, timedelta
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from pprint import pprint


from hyperliquid.vaults import fetch_vault_details, fetch_vaults_data
from metrics.drawdown import (
    calculate_max_drawdown_on_accountValue,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from metrics.sharpe_reliability import calculate_sharpe_reliability
from metrics.metric_styler import MetricsStyler

# 浮動小数点エラー（invalid, divide, over, under）も例外化
# warnings.filterwarnings("error", category=RuntimeWarning)
# np.seterr(all="raise")

# Page config
st.set_page_config(
    page_title="HyperLiquid Vault Analyser", page_icon="📊", layout="wide"
)

# Title and description
st.title("📊 HyperLiquid Vault Analyser")

# Update time display
try:
    with open("./cache/vaults_cache.json", "r") as f:
        cache = json.load(f)
        last_update = datetime.fromisoformat(cache["last_update"])
        st.caption(f"🔄 Last update: {last_update.strftime('%Y-%m-%d %H:%M')} UTC")
except (FileNotFoundError, KeyError, ValueError):
    st.warning("⚠️ Cache not found. Data will be fetched fresh.")
st.markdown("---")  # Add a separator line


def check_date_file_exists(directory="./cache"):
    """
    Checks if the `date.json` file exists in the specified directory.

    :param directory: Directory where the file is expected to be located (default: /cache).
    :return: True if the file exists, otherwise False.
    """
    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Check existence
    return os.path.exists(file_path)


def create_date_file(directory="./cache"):
    """
    Creates a `date.json` file in the specified directory with the current date.

    :param directory: Directory where the file will be created (default: /cache).
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Content to write
    current_date = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # Write to the file
    with open(file_path, "w") as file:
        json.dump(current_date, file, indent=4)
    print(f"`date.json` file created in {file_path}")


def read_date_file(directory="./cache"):
    """
    Reads and returns the date saved in the `date.json` file from the specified directory.

    :param directory: Directory where the file is located (default: /cache).
    :return: The date as a string or None if the file doesn't exist.
    """
    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Check if the file exists
    if not os.path.exists(file_path):
        print("`date.json` file not found.")
        return None

    # Read the file
    with open(file_path, "r") as file:
        data = json.load(file)
        return data.get("date")


# Layout for 3 columns


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


limit_vault = False


DATAFRAME_CACHE_FILE = "./cache/dataframe.pkl"

cache_used = False
try:
    final_df = pd.read_pickle(DATAFRAME_CACHE_FILE)
    cache_used = True
except (FileNotFoundError, KeyError, ValueError):
    pass

if not cache_used or True:

    # Get vaults data (will use cache if valid)
    vaults = fetch_vaults_data()

    # Limit to the first 3 vaults if needed
    if limit_vault:
        vaults = vaults[:3]

    # Process vault details from cache
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing vault details...")
    total_steps = len(vaults)
    indicators = []
    progress_i = 1

    check_vault_name = None
    # check_vault_name = "Stabilizer"

    s_return_list = []

    for vault in vaults:
        if check_vault_name and vault["Name"] != check_vault_name:
            continue
        progress_bar.progress(progress_i / total_steps)
        progress_i = progress_i + 1
        status_text.text(f"Processing vault details ({progress_i}/{total_steps})...")

        while True:
            try:
                # Fetch vault details
                details = fetch_vault_details(vault["Leader"], vault["Vault"])
                break  # Exit the loop if successful
            except Exception as e:
                print(f"\nError fetching details for {vault['Name']}: {e}")
                st.warning(f"Retrying to fetch details for {vault['Name']}...")
                time.sleep(5)

        nb_followers = 0
        if details and "followers" in details:
            nb_followers = sum(
                1 for f in details["followers"] if float(f["vaultEquity"]) >= 0.01
            )

        if details and "portfolio" in details:
            if details["portfolio"][3][0] == "allTime":
                sampling_days = 7

                leader_eq = next(
                    f for f in details["followers"] if f["user"] == "Leader"
                )["vaultEquity"]
                leader_eq = float(leader_eq)

                tvl = sum(float(f["vaultEquity"]) for f in details["followers"])

                # --- Leader シェア -------------------------------------------------
                if tvl == 0:
                    continue
                leader_fraction = leader_eq / tvl  # 0.0–1.0

                # """
                # 日次だと思ってたが、違う！！！！
                # 別のデータソースから日次データを取得しなくては・・・
                # """

                data_source_pnlHistory = details["portfolio"][3][1].get(
                    "pnlHistory", []
                )
                data_source_accountValueHistory = details["portfolio"][3][1].get(
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
                        ret = 0
                    else:
                        ret = max(-1, pnl / used_capital)
                    returns.append(ret)

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
                        balance = round(balance * (1 + ret), 2)
                    rebuilded_pnl.append(balance)
                    bankrupts.append(bankrupt)
                    timestamps.append(data_source_pnlHistory[idx][0])

                #
                # if max(returns) > 1 and pnl > 1000:
                if is_print_bunkrupt and False:
                    df = pd.DataFrame(data_source_pnlHistory, columns=["Time", "PnL"])
                    df2 = pd.DataFrame(
                        data_source_accountValueHistory,
                        columns=["Time", "Account Val"],
                    )
                    df = pd.merge(df, df2, on="Time", how="left").astype(float)
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(transferIns, columns=["TransfIn"]),
                            pd.DataFrame(used_capitals, columns=["UsedCapt"]),
                            pd.DataFrame(returns, columns=["Returns"]),
                            pd.DataFrame(rebuilded_pnl, columns=["RebuildPnL"]),
                            pd.DataFrame(bankrupts, columns=["Rekt"]),
                        ],
                        axis=1,
                    )
                    pd.set_option("display.max_columns", None)  # Show all cols
                    df = df.drop(axis=1, columns=["Time"])
                    pd.set_option("display.float_format", "{:.4g}".format)
                    print(f"Vault {vault['Name']} has beel left bunkrupt:\n{df}")

                ret = pd.Series(
                    returns,
                    index=pd.to_datetime(
                        timestamps, unit="ms", origin="unix", utc=True
                    ),
                    name=vault["Name"],
                    dtype=float,
                )
                is_resample = True
                if is_resample:
                    ret_raw = ret.copy()
                    # weekly resampling by close
                    ret = (1 + ret).resample(
                        "W-MON", label="left", closed="left"
                    ).prod() - 1
                    ret.fillna(0)  # スカスカなので

                # pd.set_option("display.max_rows", 1000)  # Show
                # if ret.isna().any():
                #     raise ValueError(
                #         f"Vault {vault['Name']} has NaN values in returns.:{ret}"
                #     )
                if ret.std() == 0:
                    continue

                bankrupt = ret <= -1
                valid = ~bankrupt
                # if bankrupt.any():
                #     print(
                #         f"Vault {vault['Name']} has bankrupt returns: \n{ret[bankrupt]}"
                #     )
                # warnings.filterwarnings(
                #     "error", message="divide by zero encountered in log1p"
                # )
                log_ret = pd.Series(-np.inf, index=ret.index, dtype=float)
                log_ret.loc[valid] = np.log1p(ret.loc[valid])

                cum_ret = np.exp(log_ret.cumsum())
                dd = cum_ret / np.maximum.accumulate(cum_ret) - 1

                if check_vault_name:
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
                        .resample("D", label="left", closed="left")
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
                    "TVL Leader fraction %": round(leader_fraction * 100, 2),
                    "Days from Return(Estimate)": len(ret) * sampling_days,
                    "Weekly Sharpe Ratio": ret.mean() / ret.std(),
                    "Weekly Sortino Ratio": (
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
                    "Act. Followers": nb_followers,
                    "APR(30D) %": float(details["apr"]),
                }

                # Calculate Sharpe reliability metrics

                reliability_metrics = calculate_sharpe_reliability(
                    ret.values, h=8, SR0=0.1, LT=16  # DailyではなくWeeklyなので。
                )
                reliability_metrics_keys = reliability_metrics.keys()

                for key in reliability_metrics_keys:
                    metrics[key] = reliability_metrics[key]
                # Unpacks the metrics dictionary
                indicator_row = {"Name": vault["Name"], **metrics}
                indicators.append(indicator_row)

    progress_bar.empty()
    status_text.empty()

    st.toast("Vault details OK!", icon="✅")

    # Step 4: Merge indicators with the main table
    indicators_df = pd.DataFrame(indicators)

    vaults_df = pd.DataFrame(vaults)
    vaults_df["APR(7D) %"] = vaults_df["APR(7D) %"].astype(float)
    del vaults_df["Leader"]

    final_df = vaults_df.merge(indicators_df, on="Name", how="right")

    final_df.to_pickle(DATAFRAME_CACHE_FILE)


# Filters
# Add a column with clickable links
final_df["Link"] = final_df["Vault"].apply(
    lambda vault: f"https://app.hyperliquid.xyz/vaults/{vault}"
)

st.subheader(f"Vaults available ({len(final_df)})")
filtered_df = final_df

# Filter by 'Name' (last filter, free text)
st.markdown(
    "<h3 style='text-align: center;'>Filter by Name</h3>", unsafe_allow_html=True
)
name_filter = st.text_input(
    "Name Filter",
    "",
    placeholder="Enter names separated by ',' to filter (e.g., toto,tata)...",
    key="name_filter",
)

# Apply the filter
if name_filter.strip():  # Check that the filter is not empty
    name_list = [
        name.strip() for name in name_filter.split(",")
    ]  # List of names to search for
    pattern = "|".join(name_list)  # Create a regex pattern with logical "or"
    filtered_df = filtered_df[
        filtered_df["Name"].str.contains(pattern, case=False, na=False, regex=True)
    ]


# Organize sliders into rows of 3
sliders = [
    # from "https://api-ui.hyperliquid.xyz/info"
    {
        "label": "Min Days from Return(Estimate)",
        "column": "Days from Return(Estimate)",
        "max": False,
        "default": 90,
        "step": 1,
    },
    {
        "label": "Min Weekly Sharpe Ratio",
        "column": "Weekly Sharpe Ratio",
        "max": False,
        "default": 0.2,
        "step": 0.05,
    },
    {
        "label": "Min Weekly Sortino Ratio",
        "column": "Weekly Sortino Ratio",
        "max": False,
        "default": 0.0,
        "step": 0.05,
    },
    {
        "label": "Min Annualized Gain(simple) %/yr",
        "column": "Annualized Gain(simple) %/yr",
        "max": False,
        "default": 10,
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

for i in range(0, len(sliders), 3):
    cols = st.columns(3)
    for slider, col in zip(sliders[i : i + 3], cols):
        column = slider["column"]
        value = slider_with_label(
            slider["label"],
            col,
            min_value=float(filtered_df[column].min()),
            max_value=float(filtered_df[column].max()),
            default_value=float(slider["default"]),
            step=float(slider["step"]),
            key=f"slider_{column}",
        )
        if not value == None:
            if slider["max"]:
                filtered_df = filtered_df[filtered_df[column] <= value]
            else:
                filtered_df = filtered_df[filtered_df[column] >= value]

# Display the table
st.title(f"Vaults filtered ({len(filtered_df)}) ")


# Reset index for continuous ranking
filtered_df = filtered_df.reset_index(drop=True).sort_values(
    by="Weekly Sharpe Ratio",
    ascending=False,
    # ignore_index=True,  # 連番に振り直すなら
)

"""p 値列と数値列を自動で条件付き書式にするユーティリティ"""


styler = MetricsStyler(p_th=0.05, zmax=3).generate_style(filtered_df, final_df)

st.dataframe(
    styler,
    use_container_width=True,
    height=(len(filtered_df) * 35) + 50,
    column_config={
        "Link": st.column_config.LinkColumn("Vault Link", display_text="Vault Link")
    },
)
