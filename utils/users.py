import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st


from hyperliquid.info import Info
from pprint import pprint


# URLs for user data
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
USER_INFO_URL = "https://api.hyperliquid.xyz/info"
info = Info(base_url="https://api.hyperliquid.xyz")  # mainnet

USER_CACHE_DIR = "./cache/user/"
LEADERBOARD_CACHE_FILE = USER_CACHE_DIR + "leaderboard.json"
FETCHED_ADDRESSES_FILE = USER_CACHE_DIR + "fetched_addresses.txt"
FAILED_ADDRESSES_FILE = USER_CACHE_DIR + "failed_addresses.txt"
USER_DATA_DIR = USER_CACHE_DIR + "user_data/"
FILLS_FILE = USER_CACHE_DIR + f"fills.json"

# Configuration
MAX_ADDRESSES_TO_FETCH = 20000
API_SLEEP_SECONDS = 0.3


def ensure_user_cache_dirs():
    """Create user cache directories if they don't exist."""
    os.makedirs(USER_CACHE_DIR, exist_ok=True)
    os.makedirs(USER_DATA_DIR, exist_ok=True)


def fetch_leaderboard():
    """Fetch leaderboard data with caching."""
    ensure_user_cache_dirs()

    # Check if leaderboard cache exists
    if os.path.exists(LEADERBOARD_CACHE_FILE):
        print("Leaderboard cache found, skipping download")
        with open(LEADERBOARD_CACHE_FILE, "r") as f:
            return json.load(f)

    print("Downloading leaderboard...")

    # Fetch leaderboard data (GET request)
    response = requests.get(LEADERBOARD_URL)

    if response.status_code == 200:
        leaderboard_data = response.json()

        # Save to cache
        with open(LEADERBOARD_CACHE_FILE, "w") as f:
            json.dump(leaderboard_data, f, indent=2)

        print(f"Leaderboard saved with {len(leaderboard_data)} entries")
        return leaderboard_data
    else:
        raise Exception(f"Failed to fetch leaderboard: {response.status_code}")


def get_fetched_addresses():
    """Get list of already fetched addresses."""
    if not os.path.exists(FETCHED_ADDRESSES_FILE):
        return set()

    with open(FETCHED_ADDRESSES_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def add_fetched_address(address):
    """Add address to fetched list."""
    with open(FETCHED_ADDRESSES_FILE, "a") as f:
        f.write(f"{address}\n")


def add_failed_address(address, error):
    """Add address to failed list with error info."""
    with open(FAILED_ADDRESSES_FILE, "a") as f:
        f.write(f"{address}: {error}\n")


def fetch_user_portfolio(user_address):
    """Fetch user portfolio data with caching."""
    cache_file = os.path.join(USER_DATA_DIR, f"{user_address}.json")

    # Check if cache exists
    if os.path.exists(cache_file):
        # print(f"Cache found for {user_address}")
        with open(cache_file, "r") as f:
            return json.load(f)

    # Fetch user portfolio
    payload = {"type": "portfolio", "user": user_address}
    response = requests.post(USER_INFO_URL, json=payload)

    if response.status_code == 200:
        portfolio_data = response.json()

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(portfolio_data, f, indent=2)

        time.sleep(API_SLEEP_SECONDS)

        return portfolio_data
    else:
        raise Exception(f"Failed to fetch portfolio: {response.status_code}")


def extract_addresses_from_leaderboard(leaderboard_data):
    """Extract addresses from leaderboard data in ranking order."""
    addresses = []

    # Check if leaderboard_data has the correct structure
    if isinstance(leaderboard_data, dict) and "leaderboardRows" in leaderboard_data:
        leaderboard_rows = leaderboard_data["leaderboardRows"]
    elif isinstance(leaderboard_data, list):
        leaderboard_rows = leaderboard_data
    else:
        print(f"Unexpected leaderboard data structure: {type(leaderboard_data)}")
        return addresses

    for entry in leaderboard_rows:
        if "ethAddress" in entry:
            addresses.append(entry["ethAddress"])

    return addresses


def fetch_user_addresses(max_addresses=MAX_ADDRESSES_TO_FETCH, show_progress=True):
    """Fetch user addresses from leaderboard."""
    ensure_user_cache_dirs()

    # Get leaderboard
    leaderboard_data = fetch_leaderboard()
    all_addresses = extract_addresses_from_leaderboard(leaderboard_data)

    # Get already fetched addresses
    fetched_addresses = get_fetched_addresses()

    # Filter unfetched addresses
    unfetched_addresses = [
        addr for addr in all_addresses if addr not in fetched_addresses
    ]

    # Limit to max_addresses
    addresses_to_fetch = unfetched_addresses[:max_addresses]

    print(f"Total addresses in leaderboard: {len(all_addresses)}")
    print(f"Already fetched: {len(fetched_addresses)}")
    print(f"Will fetch: {len(addresses_to_fetch)}")

    if not addresses_to_fetch:
        print("No new addresses to fetch")
        return []

    # Progress tracking for Streamlit
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    fetched_data = []

    for i, address in enumerate(addresses_to_fetch):
        if show_progress:
            progress_bar.progress((i + 1) / len(addresses_to_fetch))
            status_text.text(
                f"Fetching address {i+1}/{len(addresses_to_fetch)}: {address[:10]}..."
            )

        try:
            # Fetch user portfolio
            portfolio_data = fetch_user_portfolio(address)

            # Add to fetched list
            add_fetched_address(address)

            fill_num = count_fills(address, days=30)

            # Store for return
            fetched_data.append(
                {
                    "address": address,
                    "portfolio": portfolio_data,
                    f"fills": fill_num,
                }
            )

        except Exception as e:
            error_msg = str(e)
            print(f"Failed to fetch {address}: {error_msg}")
            add_failed_address(address, error_msg)

        if show_progress and i >= 100 and i % 100 == 0:
            print(
                f"Fetching address {i+1}/{len(addresses_to_fetch)}: {address[:10]}..."
            )

    if show_progress:
        progress_bar.empty()
        status_text.empty()
        st.toast(f"Fetched {len(fetched_data)} user addresses!", icon="✅")

    return fetched_data


###### not necessary
def get_all_cached_user_data(st, is_debug=False):
    """Get all cached user data for analysis."""
    ensure_user_cache_dirs()

    user_data = []

    if not os.path.exists(USER_DATA_DIR):
        return user_data

    total_steps = len(os.listdir(USER_DATA_DIR))
    st.info(f"🔄 Reading {total_steps} cached users...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    with open(FILLS_FILE, "r") as f:
        fills = json.load(f)

    for i, filename in enumerate(os.listdir(USER_DATA_DIR)):
        if filename.endswith(".json"):
            address = filename[:-5]  # Remove .json extension
            filepath = os.path.join(USER_DATA_DIR, filename)

            if address not in fills:
                # print(f"Address {address} not found in fills data.")
                continue

            try:
                with open(filepath, "r") as f:
                    portfolio_data = json.load(f)
                    user_data.append(
                        {
                            "address": address,
                            "portfolio": portfolio_data,
                            "fills": fills[address],
                        }
                    )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
            if is_debug and i >= 100:
                print("Debug mode: stopping after 100 files")
                break

            progress_bar.progress((i + 1) / total_steps)
            status_text.text(
                f"Reading cached users {i+1}/{total_steps}: {address[:15]}..."
            )

    return user_data


def get_user_stats():
    """Get statistics about fetched users."""
    ensure_user_cache_dirs()

    # Count fetched addresses
    fetched_count = len(get_fetched_addresses())

    # Count failed addresses
    failed_count = 0
    if os.path.exists(FAILED_ADDRESSES_FILE):
        with open(FAILED_ADDRESSES_FILE, "r") as f:
            failed_count = len([line for line in f if line.strip()])

    # Count cached data files
    cached_count = 0
    if os.path.exists(USER_DATA_DIR):
        cached_count = len(
            [f for f in os.listdir(USER_DATA_DIR) if f.endswith(".json")]
        )

    # Get total leaderboard size
    total_count = 0
    if os.path.exists(LEADERBOARD_CACHE_FILE):
        with open(LEADERBOARD_CACHE_FILE, "r") as f:
            leaderboard_data = json.load(f)
            total_count = len(extract_addresses_from_leaderboard(leaderboard_data))

    return {
        "total_addresses": total_count,
        "fetched_addresses": fetched_count,
        "failed_addresses": failed_count,
        "cached_data_files": cached_count,
    }


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


def count_fills(
    address: str,
    days: int = 30,
    *,
    timeout: int = 10,
) -> int:
    """
    指定アドレスが直近 `days` 日間に行った Place Order アクション数を返す。

    Parameters
    ----------
    address : str
        0x で始まる 42 文字の Hyperliquid アカウントアドレス
    days : int, optional
        何日前までさかのぼるか（既定は 30 日）
    testnet : bool, optional
        True の場合はテストネット API を利用
    timeout : int, optional
        HTTP タイムアウト秒

    Returns
    -------
    int
        Place Order 件数
    """

    if os.path.exists(FILLS_FILE):
        # キャッシュがあれば読み込む

        with open(FILLS_FILE, "r") as f:
            fills = json.load(f)
            if address in fills:
                return fills[address]
    else:
        fills = {}

    # 期間の下限をエポックミリ秒で計算
    now_ms = int(time.time() * 1_000)
    since_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1_000
    )

    data = info.user_fills(address)
    fill_num = sum(since_ms <= fill["time"] <= now_ms for fill in data)
    fills[address] = fill_num

    with open(FILLS_FILE, "w") as f:
        json.dump(fills, f, indent=2)
    time.sleep(API_SLEEP_SECONDS)
    # フィルタしてカウント
    return fill_num
