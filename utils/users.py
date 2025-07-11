import json
import os
import time
from decimal import Decimal
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st


from hyperliquid.info import Info
from pprint import pprint
import traceback

# URLs for user data
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
USER_INFO_URL = "https://api.hyperliquid.xyz/info"
info = Info(base_url="https://api.hyperliquid.xyz")  # mainnet

USER_CACHE_DIR = "./cache/user/"
LEADERBOARD_CACHE_FILE = USER_CACHE_DIR + "leaderboard.json"
FETCHED_ADDRESSES_FILE = USER_CACHE_DIR + "fetched_addresses.txt"
FAILED_ADDRESSES_FILE = USER_CACHE_DIR + "failed_addresses.txt"
USER_DATA_DIR = USER_CACHE_DIR + "user_data/"


# Configuration
MAX_ADDRESSES_TO_FETCH = 20000
API_SLEEP_SECONDS = 0.4


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


def load_user_data(address):
    """Fetch user portfolio data with caching."""
    cache_file = os.path.join(USER_DATA_DIR, f"{address}.json")
    # Check if cache exists
    if os.path.exists(cache_file):
        # print(f"Cache found for {user_address}")
        with open(cache_file, "r") as f:
            info = json.load(f)
    else:
        info = {}
    return info


def save_user_data(address, data, data_type=None):
    """Save user portfolio data to cache."""
    cache_file = os.path.join(USER_DATA_DIR, f"{address}.json")
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"User data: {data_type} for {address} saved to cache.")


def add_address(address):
    data = load_user_data(address)
    if "address" in data:
        pass
    else:
        data["address"] = address
        save_user_data(address, data, "address")
    return


def fetch_user_portfolio(address):
    data = load_user_data(address)
    if type(data) is list:
        save_user_data(address, {"portfolio": data}, "portfolio")
        return
    elif "portfolio" in data:
        # if "portfolio" in data["portfolio"]:
        #     print("nested portfolio found, saving directly")
        #     # すでにポートフォリオがある場合は何もしない
        #     save_user_data(address, data["portfolio"], "portfolio")
        return
    # elif "portofolio" in data:
    #     print("typo error: 'portofolio' should be 'portfolio'")
    #     data["portfolio"] = data["portofolio"]
    #     del data["portofolio"]

    #     save_user_data(address, data, "portfolio")
    #     return

    # Fetch user portfolio
    payload = {"type": "portfolio", "user": address}
    response = requests.post(USER_INFO_URL, json=payload)

    if response.status_code == 200:
        portfolio_data = response.json()

        # Save to cache
        data["portfolio"] = portfolio_data
        save_user_data(address, data, "portfolio")
        time.sleep(API_SLEEP_SECONDS)
        return
    else:
        raise Exception(f"Failed to fetch portfolio: {response.status_code}")


def fetch_fills(address: str, days: int = 30) -> int:
    data = load_user_data(address)

    if "fills" in data:
        # すでにフィルがある場合は何もしない
        return

    # 期間の下限をエポックミリ秒で計算
    now_ms = int(time.time() * 1_000)
    since_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1_000
    )
    res = info.user_fills(address)
    fill_num = sum(since_ms <= fill["time"] <= now_ms for fill in res)

    data["fills"] = fill_num
    save_user_data(address, data, "fills")
    time.sleep(API_SLEEP_SECONDS)
    # フィルタしてカウント
    return


def fetch_fees(address: str) -> dict:
    data = load_user_data(address)
    if "fees" in data:
        # すでにフィーがある場合は何もしない
        return

    # 期間の下限をエポックミリ秒で計算
    res = info.user_fees(address)
    USER_KEYS = {
        "activeReferralDiscount",
        "activeStakingDiscount",
        "feeTrialReward",
        "nextTrialAvailableTimestamp",
        "stakingLink",
        "trial",
        "userAddRate",
        "userCrossRate",
        "userSpotAddRate",
        "userSpotCrossRate",
    }

    fees = {k: res[k] for k in USER_KEYS if k in res}

    # 返り値も他フィールドに合わせて文字列化
    # -------- dailyUserVlm を dict に畳み込む -----------------------
    total_exchange = Decimal("0")
    total_add = Decimal("0")
    total_cross = Decimal("0")

    for day in res.get("dailyUserVlm", []):
        total_exchange += Decimal(day["exchange"])
        total_add += Decimal(day["userAdd"])
        total_cross += Decimal(day["userCross"])

    fees["dailyUserVlm_14D"] = {
        "exchange": float(total_exchange),
        "userAdd": float(total_add),
        "userCross": float(total_cross),
    }

    data["fees"] = fees
    save_user_data(address, data, "fees")
    time.sleep(API_SLEEP_SECONDS)
    # フィルタしてカウント
    return


def fetch_state(address: str, state_type: str = "perp") -> None:
    """
    指定アドレスの 'perp' または 'spot' 状態を取得し、
    data["state"][state_type] に保存する。

    Parameters
    ----------
    address : str
        Hyperliquid L1 アドレス (0x...)
    state_type : {'perp', 'spot'}
        取得対象のステート種類
        - 'perp' : clearinghouse_state
        - 'spot' : spot_user_state
    """

    data = load_user_data(address)  # 既存ファイル読み込み

    # 最初の呼び出しで state 用 dict を用意
    if "state" not in data:
        data["state"] = {}

    # すでに同種 state が保存済みなら何もしない
    if state_type in data["state"]:
        return

    # ------------------- API 呼び分け -------------------
    if state_type == "perp":
        state = info.user_state(address)
    elif state_type == "spot":
        state = info.spot_user_state(address)  # SDK >=0.2.3
    else:
        raise ValueError(f"unsupported state_type: {state_type}")
    # ---------------------------------------------------

    # 保存してディスクへ
    data["state"][state_type] = state
    save_user_data(address, data, f"state_{state_type}")

    # レート制限を避けるためスリープ
    time.sleep(API_SLEEP_SECONDS)
    return


#########################################
#########################################


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


def fetch_user_addresses(
    max_addresses=MAX_ADDRESSES_TO_FETCH, addresses_force_fetch=[], show_progress=True
):
    """Fetch user addresses from leaderboard."""
    ensure_user_cache_dirs()

    # Get leaderboard
    leaderboard_data = fetch_leaderboard()
    all_addresses = extract_addresses_from_leaderboard(leaderboard_data)

    # Get already fetched addresses
    fetched_addresses = get_fetched_addresses()

    # Filter unfetched addresses
    unfetched_addresses = [
        addr
        for addr in all_addresses
        if (addr not in fetched_addresses) or (addr in addresses_force_fetch)
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

            # Fetch info for the user
            fetch_user_portfolio(address)
            fetch_fills(address)
            fetch_fees(address)
            fetch_state(address, "perp")
            fetch_state(address, "spot")
            add_address(address)

            data = load_user_data(address)

            add_fetched_address(address)
            # Store for return
            fetched_data.append(data)

        except Exception as e:
            error_msg = str(e)
            print(f"Failed to fetch {address}: {error_msg}")
            # traceback.print_exc()
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

    failed_data = []

    for i, filename in enumerate(os.listdir(USER_DATA_DIR)):
        if filename.endswith(".json"):
            address = filename[:-5]  # Remove .json extension

            data = load_user_data(address)  # Ensure data is loaded

            # if not data:
            #     print(f"No data found for {address}")

            if type(data) is dict:
                keys = ["address", "portfolio", "fees", "fills", "state"]
                is_all_key = all(key in data for key in keys)
                if not is_all_key:
                    print(f"Missing keys in {address}: {data.keys()}")
                    failed_data.append(data)
                    continue
            else:
                failed_data.append(data)
                print(f"Invalid data format for {address}: {type(data)}")
                continue

            user_data.append(data)

            if is_debug and i >= 100:
                print("Debug mode: stopping after 100 files")
                break

            progress_bar.progress((i + 1) / total_steps)
            status_text.text(
                f"Reading cached users {i+1}/{total_steps}: {address[:15]}..."
            )

    return user_data, failed_data


def get_user_stats():
    """Get statistics about fetched users."""
    ensure_user_cache_dirs()
    fetch_leaderboard()

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
