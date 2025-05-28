# source .venv/bin/activate
# streamlit run main.py

import json
import os
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

# 浮動小数点エラー（invalid, divide, over, under）も例外化
# warnings.filterwarnings("error", category=RuntimeWarning)
# np.seterr(all="raise")

# Page config
st.set_page_config(page_title="HyperLiquid Vault Analyser", page_icon="📊", layout="wide")

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
    col.markdown(f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True)
    if not min_value < max_value:
        col.markdown(
            f"<p style='text-align: center;'>No choice available ({min_value} for all)</p>", unsafe_allow_html=True
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


def calculate_average_daily_gain(rebuilded_pnl, days_since):
    """
    Calculates the average daily gain percentage.

    :param rebuilded_pnl: List of cumulative PnL values ($).
    :param days_since: Number of days (int).
    :return: Average daily gain percentage (float).
    """
    if len(rebuilded_pnl) < 2 or days_since <= 0:
        return 0  # Not enough data to calculate

    initial_value = rebuilded_pnl[0]
    final_value = rebuilded_pnl[-1]

    # Avoid division by zero
    if initial_value == 0:
        return 0  # Cannot calculate if the initial value is 0

    average_daily_gain_pct = ((final_value - initial_value) / (initial_value * days_since)) * 100
    return average_daily_gain_pct


def calculate_total_gain_percentage(rebuilded_pnl):
    """
    Calculates the total percentage change since the beginning.

    :param rebuilded_pnl: List of cumulative PnL values ($).
    :return: Total percentage change (float).
    """
    if len(rebuilded_pnl) < 2:
        return 0  # Not enough data to calculate

    initial_value = rebuilded_pnl[0]
    final_value = rebuilded_pnl[-1]

    # Avoid division by zero
    if initial_value == 0:
        return 0  # Cannot calculate if the initial value is 0

    total_gain_pct = ((final_value - initial_value) / initial_value) * 100
    return total_gain_pct


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

    for vault in vaults:
        progress_bar.progress(progress_i / total_steps)
        progress_i = progress_i + 1
        status_text.text(f"Processing vault details ({progress_i}/{total_steps})...")

        details = fetch_vault_details(vault["Leader"], vault["Vault"])

        nb_followers = 0
        if details and "followers" in details:
            nb_followers = sum(1 for f in details["followers"] if float(f["vaultEquity"]) >= 0.01)

        if details and "portfolio" in details:
            if details["portfolio"][3][0] == "allTime":
                # """
                # 日次だと思ってたが、違う！！！！
                # 別のデータソースから日次データを取得しなくては・・・
                # """

                data_source_pnlHistory = details["portfolio"][3][1].get("pnlHistory", [])
                data_source_accountValueHistory = details["portfolio"][3][1].get("accountValueHistory", [])
                rebuilded_pnl = []
                final_capital_virtuals = []
                used_capitals = []
                returns = []
                bankrupts = []
                transferIns = []

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
                        ret = pnl / used_capital
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
                    pd.set_option("display.max_columns", None)  # Show all rows
                    df = df.drop(axis=1, columns=["Time"])
                    pd.set_option("display.float_format", "{:.4g}".format)
                    print(f"Vault {vault['Name']} has beel left bunkrupt:\n{df}")

                ret = np.asarray(returns, dtype=float)
                ret = ret[np.isfinite(ret)]
                if len(ret) < 3 or ret.std() == 0:
                    # null strategy, skip it
                    continue

                bankrupt = ret <= -1
                log_ret = np.log1p(ret, where=~bankrupt, out=np.full_like(ret, -np.inf))
                cum_ret = np.exp(log_ret.cumsum())
                dd = cum_ret / np.maximum.accumulate(cum_ret) - 1

                metrics = {
                    "Days from Return(Estimate)": len(ret) * 7,
                    "Weekly Sharpe Ratio": ret.mean() / ret.std(),
                    "Weekly Sortino Ratio": (
                        ret.mean() / ret[ret > 0].std()
                        if len(ret[ret > 0]) >= 2 and ret[ret > 0].std() != 0
                        else 1000000
                    ),
                    # "Daily Gain %": ret.mean() * 100,
                    "Gain(simple) %": ret.sum() * 100,
                    "Annualized Gain(simple) %/yr": ret.mean() / 7 * 365 * 100,
                    # "Gain(compound) %": (
                    #     (cum_ret - 1) * 100 if ret.min() > -1 else -100
                    # ),
                    "Annualized Median Gain(compound) %/yr": (
                        ret.mean() - 1 / 2 * ret.var()
                    )
                    / 7
                    * 365
                    * 100,
                    "Max DD %": dd.min() * (-1) * 100,
                    "Rekt": nb_rekt,
                    "Act. Followers": nb_followers,
                    "APR(30D) %": float(details["apr"]),
                }

                # Calculate Sharpe reliability metrics

                reliability_metrics = calculate_sharpe_reliability(
                    ret, h=8, SR0=0.1, LT=16  # DailyではなくWeeklyなので。
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
    del vaults_df["Leader"]

    final_df = vaults_df.merge(indicators_df, on="Name", how="left")

    final_df.to_pickle(DATAFRAME_CACHE_FILE)


# Filters
st.subheader(f"Vaults available ({len(final_df)})")
filtered_df = final_df


# Filter by 'Name' (last filter, free text)
st.markdown("<h3 style='text-align: center;'>Filter by Name</h3>", unsafe_allow_html=True)
name_filter = st.text_input(
    "Name Filter", "", placeholder="Enter names separated by ',' to filter (e.g., toto,tata)...", key="name_filter"
)

# Apply the filter
if name_filter.strip():  # Check that the filter is not empty
    name_list = [name.strip() for name in name_filter.split(",")]  # List of names to search for
    pattern = "|".join(name_list)  # Create a regex pattern with logical "or"
    filtered_df = filtered_df[filtered_df["Name"].str.contains(pattern, case=False, na=False, regex=True)]


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
        "default": 0.1,
        "step": 0.05,
    },
    {
        "label": "Min Weekly Sortino Ratio",
        "column": "Weekly Sortino Ratio",
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
        "label": "Max DD % accepted",
        "column": "Max DD %",
        "max": True,
        "default": 50,
        "step": 1,
    },
    {
        "label": "Max Rekt accepted",
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
        "label": "Min APR(30D) accepted",
        "column": "APR(30D) %",
        "max": False,
        "default": 0,
        "step": 1,
    },
    # from "https://stats-data.hyperliquid.xyz/Mainnet/vaults"
    {
        "label": "Min Days Since accepted",
        "column": "Days Since",
        "max": False,
        "default": 90,
        "step": 1,
    },
    {
        "label": "Min TVL accepted",
        "column": "Total Value Locked",
        "max": False,
        "default": 10000,
        "step": 10,
    },
    {
        "label": "Min APR(7D) accepted",
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

# Add a column with clickable links
filtered_df["Link"] = filtered_df["Vault"].apply(lambda vault: f"https://app.hyperliquid.xyz/vaults/{vault}")

# Reset index for continuous ranking
filtered_df = filtered_df.reset_index(drop=True)


st.dataframe(
    filtered_df,
    use_container_width=True,
    # Adjust height based on the number of rows
    height=(len(filtered_df) * 35) + 50,
    column_config={
        "Link": st.column_config.LinkColumn(
            "Vault Link",
            display_text="Vault Link",
        )
    },
)
