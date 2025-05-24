# source .venv/bin/activate
# streamlit run user_analysis.py

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from hyperliquid.users import (
    process_user_addresses, 
    get_all_cached_user_data, 
    get_user_stats,
    MAX_ADDRESSES_TO_PROCESS
)
from metrics.drawdown import (
    calculate_max_drawdown_on_accountValue,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from metrics.sharpe_reliability import calculate_sharpe_reliability

# Page config
st.set_page_config(page_title="HyperLiquid User Analyser", page_icon="👤", layout="wide")

# Title and description
st.title("👤 HyperLiquid User Analyser")

# Get user statistics
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


def calculate_days_since_start(portfolio_data):
    """Calculate days since the user started trading."""
    try:
        # Look for allTime data
        for period_data in portfolio_data:
            if period_data[0] == "allTime":
                account_history = period_data[1].get("accountValueHistory", [])
                if account_history:
                    # Get first timestamp
                    first_timestamp = account_history[0][0]
                    first_date = datetime.fromtimestamp(first_timestamp / 1000)
                    days_since = (datetime.now() - first_date).days
                    return max(1, days_since)  # At least 1 day
        return 1  # Default to 1 day if no data
    except:
        return 1


def process_user_data_for_analysis(user_data_list):
    """Process user data to extract metrics similar to vault analysis."""
    indicators = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, user_data in enumerate(user_data_list):
        progress_bar.progress((i + 1) / len(user_data_list))
        status_text.text(f"Analyzing user {i+1}/{len(user_data_list)}: {user_data['address'][:10]}...")
        
        address = user_data["address"]
        portfolio_data = user_data["portfolio"]
        
        try:
            # Find allTime data
            rebuilded_pnl = []
            days_since = calculate_days_since_start(portfolio_data)
            
            for period_data in portfolio_data:
                if period_data[0] == "allTime":
                    data_source_pnlHistory = period_data[1].get("pnlHistory", [])
                    data_source_accountValueHistory = period_data[1].get("accountValueHistory", [])
                    
                    if not data_source_pnlHistory or not data_source_accountValueHistory:
                        continue
                    
                    balance = start_balance_amount = 1000000
                    nb_rekt = 0
                    last_rekt_idx = -10
                    
                    # Recalculate the balance without considering deposit movements
                    for idx, value in enumerate(data_source_pnlHistory):
                        if idx == 0:
                            continue

                        # Capital at time T
                        final_capital = float(data_source_accountValueHistory[idx][1])
                        # Cumulative PnL at time T
                        final_cumulated_pnl = float(data_source_pnlHistory[idx][1])
                        # Cumulative PnL at time T -1
                        previous_cumulated_pnl = float(data_source_pnlHistory[idx - 1][1]) if idx > 0 else 0
                        # Non-cumulative PnL at time T
                        final_pnl = final_cumulated_pnl - previous_cumulated_pnl
                        # Capital before the gain/loss
                        initial_capital = final_capital - final_pnl

                        if initial_capital <= 0:
                            if last_rekt_idx + 1 != idx:
                                rebuilded_pnl = []
                                balance = start_balance_amount
                                nb_rekt = nb_rekt + 1
                            last_rekt_idx = idx
                            continue
                        # Gain/loss ratio
                        ratio = final_capital / initial_capital

                        # Verify timestamp consistency
                        if data_source_pnlHistory[idx][0] != data_source_accountValueHistory[idx][0]:
                            print("Timestamp mismatch detected")
                            continue

                        # Update the simulated balance
                        balance = balance * ratio
                        rebuilded_pnl.append(balance)
                    
                    break
            
            if not rebuilded_pnl:
                continue
            
            # Calculate Sharpe reliability metrics
            reliability_metrics = calculate_sharpe_reliability(rebuilded_pnl, vault_name=address, debug_mode=False)
            
            # Calculate current account value (approximate)
            current_account_value = 0
            try:
                for period_data in portfolio_data:
                    if period_data[0] == "allTime":
                        account_history = period_data[1].get("accountValueHistory", [])
                        if account_history:
                            current_account_value = float(account_history[-1][1])
                        break
            except:
                current_account_value = 0
            
            metrics = {
                "Max DD %": calculate_max_drawdown_on_accountValue(rebuilded_pnl),
                "Rekt": nb_rekt,
                "Current Account Value": current_account_value,
                "Sharpe Ratio": calculate_sharpe_ratio(rebuilded_pnl),
                "Sortino Ratio": calculate_sortino_ratio(rebuilded_pnl),
                "Av. Daily Gain %": calculate_average_daily_gain(rebuilded_pnl, days_since),
                "Gain %": calculate_total_gain_percentage(rebuilded_pnl),
                "Days Since": days_since,
                "Sharpe Reliability": reliability_metrics["Sharpe Reliability"],
                "JKM Test P-value": reliability_metrics["JKM Test P-value"],
                "Lo Test P-value": reliability_metrics["Lo Test P-value"],
                "Fisher Score": reliability_metrics["Fisher Score"],
            }
            
            # Create indicator row
            indicator_row = {"Address": address, **metrics}
            indicators.append(indicator_row)
            
        except Exception as e:
            print(f"Error processing user {address}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return indicators


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


# Data processing section
USER_DATAFRAME_CACHE_FILE = "./user_cache/user_dataframe.pkl"

cache_used = False
try:
    final_df = pd.read_pickle(USER_DATAFRAME_CACHE_FILE)
    cache_used = True
    st.info(f"📊 Using cached analysis data ({len(final_df)} users)")
except (FileNotFoundError, KeyError, ValueError):
    pass

if not cache_used:
    # Check if we have any cached user data
    cached_user_data = get_all_cached_user_data()
    
    if not cached_user_data:
        st.warning("⚠️ No user data found. Please process some users first.")
        
        # Button to process users
        if st.button(f"🔄 Process {MAX_ADDRESSES_TO_PROCESS} Users from Leaderboard"):
            with st.spinner("Processing users..."):
                process_user_addresses(max_addresses=MAX_ADDRESSES_TO_PROCESS, show_progress=True)
            st.rerun()
        
        st.stop()
    
    # Process the cached data
    st.info(f"🔄 Processing {len(cached_user_data)} cached users...")
    indicators = process_user_data_for_analysis(cached_user_data)
    
    if not indicators:
        st.error("❌ No valid user data could be processed.")
        st.stop()
    
    # Create DataFrame
    final_df = pd.DataFrame(indicators)
    
    # Save to cache
    final_df.to_pickle(USER_DATAFRAME_CACHE_FILE)
    st.toast("User analysis data cached!", icon="✅")

# Display results
st.subheader(f"Users analyzed ({len(final_df)})")

# Add process more users button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button(f"➕ Process {MAX_ADDRESSES_TO_PROCESS} More Users"):
        with st.spinner("Processing more users..."):
            new_data = process_user_addresses(max_addresses=MAX_ADDRESSES_TO_PROCESS, show_progress=True)
            if new_data:
                # Clear cache to force reprocessing
                if os.path.exists(USER_DATAFRAME_CACHE_FILE):
                    os.remove(USER_DATAFRAME_CACHE_FILE)
                st.rerun()

# Filters
filtered_df = final_df.copy()

# Filter by 'Address' (last filter, free text)
st.markdown("<h3 style='text-align: center;'>Filter by Address</h3>", unsafe_allow_html=True)
address_filter = st.text_input(
    "Address Filter", "", placeholder="Enter addresses separated by ',' to filter (e.g., 0x123,0x456)...", key="address_filter"
)

# Apply the filter
if address_filter.strip():
    address_list = [addr.strip() for addr in address_filter.split(",")]
    pattern = "|".join(address_list)
    filtered_df = filtered_df[filtered_df["Address"].str.contains(pattern, case=False, na=False, regex=True)]

# Organize sliders into rows of 3
sliders = [
    {"label": "Min Sharpe Ratio", "column": "Sharpe Ratio", "max": False, "default": 0.4, "step": 0.1},
    {"label": "Min Sortino Ratio", "column": "Sortino Ratio", "max": False, "default": 0.5, "step": 0.1},
    {"label": "Max Rekt accepted", "column": "Rekt", "max": True, "default": 0, "step": 1},
    {"label": "Max DD % accepted", "column": "Max DD %", "max": True, "default": 15, "step": 1},
    {"label": "Min Days Since accepted", "column": "Days Since", "max": False, "default": 100, "step": 1},
    {"label": "Min Account Value", "column": "Current Account Value", "max": False, "default": 1000, "step": 100},
    {"label": "Min Av. Daily Gain %", "column": "Av. Daily Gain %", "max": False, "default": 0, "step": 0.1},
    {"label": "Min Gain %", "column": "Gain %", "max": False, "default": 0, "step": 1},
    {"label": "Max Sharpe Reliability", "column": "Sharpe Reliability", "max": True, "default": 0.05, "step": 0.01},
    {"label": "Min Fisher Score", "column": "Fisher Score", "max": False, "default": 0.0, "step": 0.01},
]

for i in range(0, len(sliders), 3):
    cols = st.columns(3)
    for slider, col in zip(sliders[i : i + 3], cols):
        column = slider["column"]
        if column in filtered_df.columns:
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
                if slider["max"]:
                    filtered_df = filtered_df[filtered_df[column] <= value]
                else:
                    filtered_df = filtered_df[filtered_df[column] >= value]

# Display the table
st.title(f"Users filtered ({len(filtered_df)}) ")

# Add a column with clickable links to HyperLiquid
filtered_df["Link"] = filtered_df["Address"].apply(lambda addr: f"https://app.hyperliquid.xyz/trade/{addr}")

# Reset index for continuous ranking
filtered_df = filtered_df.reset_index(drop=True)

st.dataframe(
    filtered_df,
    use_container_width=True,
    # Adjust height based on the number of rows
    height=(len(filtered_df) * 35) + 50,
    column_config={
        "Link": st.column_config.LinkColumn(
            "User Link",
            display_text="User Link",
        ),
        "Current Account Value": st.column_config.NumberColumn(
            "Current Account Value",
            format="$%.2f"
        )
    },
)
