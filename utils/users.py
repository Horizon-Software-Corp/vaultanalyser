import json
import os
import time
from decimal import Decimal
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import pandas as pd
from hyperliquid.info import Info
from pprint import pprint
import traceback
from .proxies import PROXIES


class UserDataManager:
    """Manages user data fetching and caching for Hyperliquid users."""

    def __init__(self, base_url="https://api.hyperliquid.xyz", use_proxies=True):
        # URLs for user data
        self.LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
        self.USER_INFO_URL = "https://api.hyperliquid.xyz/info"

        # Load proxies and create Info instances
        self.info_instances = []
        self.current_info_index = 0

        if use_proxies:
            self._load_proxies_and_create_instances(base_url)
        else:
            # Fallback to single instance without proxy
            self.info_instances.append(Info(base_url=base_url))

        self.USER_CACHE_DIR = "./cache/user/"
        self.LEADERBOARD_CACHE_FILE = self.USER_CACHE_DIR + "leaderboard.json"
        self.FETCHED_ADDRESSES_FILE = self.USER_CACHE_DIR + "fetched_addresses.txt"
        self.FAILED_ADDRESSES_FILE = self.USER_CACHE_DIR + "failed_addresses.txt"
        self.USER_DATA_DIR = self.USER_CACHE_DIR + "user_data/"

        # Configuration
        self.MAX_ADDRESSES_TO_FETCH = 30000
        self.API_SLEEP_SECONDS = 0.3

    def _load_proxies_and_create_instances(self, base_url):
        """Load proxies from proxies.json and create Info instances for each."""
        try:
            proxies = PROXIES
            print(f"Loaded {len(proxies)} proxies\n")

            # Create Info instance without proxy first (default)
            self.info_instances.append(Info(base_url=base_url))

            # Create Info instances with proxies
            for i, proxy in enumerate(proxies):
                try:
                    # Create Info instance with proxy
                    info_with_proxy = Info(base_url=base_url)

                    # Configure proxy for the session
                    proxy_url = f"{proxy['protocol']}://{proxy['username']}:{proxy['password']}@{proxy['ip']}"
                    proxy_dict = {"http": proxy_url, "https": proxy_url}

                    # Set proxy for the session
                    info_with_proxy.session.proxies.update(proxy_dict)

                    self.info_instances.append(info_with_proxy)
                    print(f"Created Info instance {i+1} with proxy {proxy['ip']}")

                except Exception as e:
                    print(
                        f"Failed to create Info instance with proxy {proxy['ip']}: {e}"
                    )

        except FileNotFoundError:
            print("proxies.json not found, using single instance without proxy")
            self.info_instances.append(Info(base_url=base_url))

    def _get_next_info(self):
        """Get the next Info instance in rotation."""
        info = self.info_instances[self.current_info_index]
        proxy_info = (
            "direct"
            if self.current_info_index == 0
            else f"proxy-{self.current_info_index}"
        )
        # print(
        #     f"Using Info instance {self.current_info_index + 1}/{len(self.info_instances)} ({proxy_info})"
        # )
        self.current_info_index = (self.current_info_index + 1) % len(
            self.info_instances
        )
        return info

    @property
    def info(self):
        """Get the next Info instance for load balancing."""
        return self._get_next_info()

    def ensure_user_cache_dirs(self):
        """Create user cache directories if they don't exist."""
        os.makedirs(self.USER_CACHE_DIR, exist_ok=True)
        os.makedirs(self.USER_DATA_DIR, exist_ok=True)

    def fetch_leaderboard(self):
        """Fetch leaderboard data with caching."""
        self.ensure_user_cache_dirs()

        # Check if leaderboard cache exists
        if os.path.exists(self.LEADERBOARD_CACHE_FILE):
            print("Leaderboard cache found, skipping download")
            with open(self.LEADERBOARD_CACHE_FILE, "r") as f:
                return json.load(f)

        print("Downloading leaderboard...")

        # Try to use proxy if available
        proxies = None
        if len(self.info_instances) > 1 and hasattr(self.info_instances[1], "session"):
            proxies = self.info_instances[1].session.proxies

        # Fetch leaderboard data (GET request)
        response = requests.get(self.LEADERBOARD_URL, proxies=proxies)

        if response.status_code == 200:
            leaderboard_data = response.json()

            # Save to cache
            with open(self.LEADERBOARD_CACHE_FILE, "w") as f:
                json.dump(leaderboard_data, f, indent=2)

            print(f"Leaderboard saved with {len(leaderboard_data)} entries")
            return leaderboard_data
        else:
            raise Exception(f"Failed to fetch leaderboard: {response.status_code}")

    def get_fetched_addresses(self):
        """Get list of already fetched addresses."""
        if not os.path.exists(self.FETCHED_ADDRESSES_FILE):
            return set()

        with open(self.FETCHED_ADDRESSES_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())

    def add_fetched_address(self, address):
        """Add address to fetched list."""
        with open(self.FETCHED_ADDRESSES_FILE, "a") as f:
            f.write(f"{address}\n")

    def add_failed_address(self, address, error):
        """Add address to failed list with error info."""
        with open(self.FAILED_ADDRESSES_FILE, "a") as f:
            f.write(f"{address}: {error}\n")

    def load_user_data(self, address):
        """Fetch user portfolio data with caching."""
        cache_file = os.path.join(self.USER_DATA_DIR, f"{address}.json")
        # Check if cache exists
        if os.path.exists(cache_file):
            # print(f"Cache found for {user_address}")
            with open(cache_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        return data

    def save_user_data(self, address, data, data_type=None):
        """Save user portfolio data to cache."""
        cache_file = os.path.join(self.USER_DATA_DIR, f"{address}.json")
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"User data: {data_type} for {address} saved to cache.")

    def add_address(self, address):
        data = self.load_user_data(address)
        if "address" in data:
            pass
        else:
            data["address"] = address
            self.save_user_data(address, data, "address")
        return

    def fetch_user_portfolio(self, address):
        data = self.load_user_data(address)
        if type(data) is list:
            self.save_user_data(address, {"portfolio": data}, "portfolio")
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

        # Fetch user portfolio using info.user_state
        try:
            portfolio_data = self.info.user_state(address)

            # Save to cache
            data["portfolio"] = portfolio_data
            self.save_user_data(address, data, "portfolio")
            time.sleep(self.API_SLEEP_SECONDS)
            return
        except Exception as e:
            raise Exception(f"Failed to fetch portfolio: {e}")

    def fetch_fills(self, address: str, days: int = 30) -> int:
        data = self.load_user_data(address)

        if "fills" in data:
            if (
                isinstance(data["fills"], dict)
                and "count" in data["fills"]
                and "traded_symbols" in data["fills"]
                and "last_fill_seconds" in data["fills"]
            ):
                return

        # 期間の下限をエポックミリ秒で計算
        now_ms = int(time.time() * 1_000)
        since_ms = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1_000
        )
        res = self.info.user_fills(address)
        fill_num = sum(since_ms <= fill["time"] <= now_ms for fill in res)

        traded_symbols = set()

        for fill in res:
            if since_ms <= fill["time"] <= now_ms:
                traded_symbols.add(fill["coin"])

        if fill_num > 0:
            first_fill_time = min(
                fill["time"] for fill in res if since_ms <= fill["time"] <= now_ms
            )
            last_fill_seconds = (now_ms - first_fill_time) // 1000
        else:
            last_fill_seconds = pd.Timedelta(days=days).total_seconds()

        data["fills"] = {
            "count": fill_num,
            "traded_symbols": list(traded_symbols),
            "last_fill_seconds": last_fill_seconds,
        }

        self.save_user_data(address, data, "fills")
        time.sleep(self.API_SLEEP_SECONDS)
        # フィルタしてカウント
        return

    def fetch_fees(self, address: str) -> dict:
        data = self.load_user_data(address)
        if "fees" in data:
            # すでにフィーがある場合は何もしない
            return

        # 期間の下限をエポックミリ秒で計算
        res = self.info.user_fees(address)
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
        self.save_user_data(address, data, "fees")
        time.sleep(self.API_SLEEP_SECONDS)
        # フィルタしてカウント
        return

    def fetch_state(self, address: str, state_type: str = "perp") -> None:
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

        data = self.load_user_data(address)  # 既存ファイル読み込み

        # 最初の呼び出しで state 用 dict を用意
        if "state" not in data:
            data["state"] = {}

        # すでに同種 state が保存済みなら何もしない
        if state_type in data["state"]:
            return

        # ------------------- API 呼び分け -------------------
        if state_type == "perp":
            state = self.info.user_state(address)
        elif state_type == "spot":
            state = self.info.spot_user_state(address)  # SDK >=0.2.3
            state["time"] = int(time.time() * 1000)
        else:
            raise ValueError(f"unsupported state_type: {state_type}")
        # ---------------------------------------------------

        # 保存してディスクへ
        data["state"][state_type] = state
        self.save_user_data(address, data, f"state_{state_type}")

        # レート制限を避けるためスリープ
        time.sleep(self.API_SLEEP_SECONDS)
        return

    def extract_addresses_from_leaderboard(self, leaderboard_data):
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
        self, max_addresses=None, addresses_force_fetch=[], show_progress=True
    ):
        """Fetch user addresses from leaderboard."""
        if max_addresses is None:
            max_addresses = self.MAX_ADDRESSES_TO_FETCH

        self.ensure_user_cache_dirs()

        # Get leaderboard
        leaderboard_data = self.fetch_leaderboard()
        all_addresses = self.extract_addresses_from_leaderboard(leaderboard_data)

        # Get already fetched addresses
        fetched_addresses = self.get_fetched_addresses()

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
                self.fetch_user_portfolio(address)
                self.fetch_fills(address)
                self.fetch_fees(address)
                self.fetch_state(address, "perp")
                self.fetch_state(address, "spot")
                self.add_address(address)

                data = self.load_user_data(address)

                self.add_fetched_address(address)
                # Store for return
                fetched_data.append(data)

            except Exception as e:
                error_msg = str(e)
                print(f"Failed to fetch {address}: {error_msg}")
                # traceback.print_exc()
                self.add_failed_address(address, error_msg)

            if show_progress and i >= 100 and i % 100 == 0:
                print(
                    f"Fetching address {i+1}/{len(addresses_to_fetch)}: {address[:10]}..."
                )

        if show_progress:
            progress_bar.empty()
            status_text.empty()
            st.toast(f"Fetched {len(fetched_data)} user addresses!", icon="✅")

        return fetched_data

    def get_all_cached_user_data(self, st, is_debug=False):
        """Get all cached user data for analysis."""
        self.ensure_user_cache_dirs()

        user_data = []

        if not os.path.exists(self.USER_DATA_DIR):
            return user_data

        total_steps = len(os.listdir(self.USER_DATA_DIR))
        st.info(f"🔄 Reading {total_steps} cached users...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        failed_data = []

        for i, filename in enumerate(os.listdir(self.USER_DATA_DIR)):
            if filename.endswith(".json"):
                address = filename[:-5]  # Remove .json extension

                data = self.load_user_data(address)  # Ensure data is loaded

                # if not data:
                #     print(f"No data found for {address}")

                if type(data) is dict:
                    data["address"] = address  # Ensure address is set
                    keys = ["address", "portfolio", "fees", "fills", "state"]
                    missing_keys = [key for key in keys if key not in data]
                    if missing_keys:
                        print(
                            f"Missing keys {missing_keys} in address {address}, {i + 1} / {total_steps}"
                        )
                        failed_data.append(data)
                        continue
                    fills_key = ["count", "traded_symbols", "last_fill_seconds"]
                    if not isinstance(data["fills"], dict):
                        print(
                            f"Invalid fills format for {address}: {type(data['fills'])}, {i + 1} / {total_steps}"
                        )
                        failed_data.append(data)
                        continue
                    missing_subkeys = [
                        key for key in fills_key if key not in data["fills"]
                    ]
                    if missing_subkeys:
                        print(
                            f"Missing subkeys {missing_subkeys} in fills for address {address}, {i + 1} / {total_steps}"
                        )
                        failed_data.append(data)
                        continue
                else:
                    print(f"Invalid data format for {address}: {type(data)}")
                    failed_data.append({"address": address})
                    continue

                user_data.append(data)

                if is_debug and i >= 100:
                    print("Debug mode: stopping after 100 files")
                    break

                progress_bar.progress((i + 1) / total_steps)
                status_text.text(
                    f"Reading cached users {i+1}/{total_steps}: {address[:15]}..."
                )

        if os.path.exists(self.FAILED_ADDRESSES_FILE):
            with open(self.FAILED_ADDRESSES_FILE, "w") as f:
                f.writelines([f"{data['address']}\n" for data in failed_data])

        if os.path.exists(self.FETCHED_ADDRESSES_FILE):
            with open(self.FETCHED_ADDRESSES_FILE, "w") as f:
                f.writelines([f"{data['address']}\n" for data in user_data])

        return user_data, failed_data

    def get_user_stats(self):
        """Get statistics about fetched users."""
        self.ensure_user_cache_dirs()
        self.fetch_leaderboard()

        # Count fetched addresses
        fetched_count = len(self.get_fetched_addresses())

        # Count failed addresses
        failed_count = 0
        if os.path.exists(self.FAILED_ADDRESSES_FILE):
            with open(self.FAILED_ADDRESSES_FILE, "r") as f:
                failed_count = len([line for line in f if line.strip()])

        # Count cached data files
        cached_count = 0
        if os.path.exists(self.USER_DATA_DIR):
            cached_count = len(
                [f for f in os.listdir(self.USER_DATA_DIR) if f.endswith(".json")]
            )

        # Get total leaderboard size
        total_count = 0
        if os.path.exists(self.LEADERBOARD_CACHE_FILE):
            with open(self.LEADERBOARD_CACHE_FILE, "r") as f:
                leaderboard_data = json.load(f)
                total_count = len(
                    self.extract_addresses_from_leaderboard(leaderboard_data)
                )

        return {
            "total_addresses": total_count,
            "fetched_addresses": fetched_count,
            "failed_addresses": failed_count,
            "cached_data_files": cached_count,
        }


# Backward compatibility: Create a default instance


# Create default manager instance with proxy support
_default_manager = UserDataManager(use_proxies=True)
USER_CACHE_DIR = _default_manager.USER_CACHE_DIR
LEADERBOARD_CACHE_FILE = _default_manager.LEADERBOARD_CACHE_FILE
FETCHED_ADDRESSES_FILE = _default_manager.FETCHED_ADDRESSES_FILE
FAILED_ADDRESSES_FILE = _default_manager.FAILED_ADDRESSES_FILE
USER_DATA_DIR = _default_manager.USER_DATA_DIR
MAX_ADDRESSES_TO_FETCH = _default_manager.MAX_ADDRESSES_TO_FETCH
API_SLEEP_SECONDS = _default_manager.API_SLEEP_SECONDS


# Expose functions for backward compatibility
def ensure_user_cache_dirs():
    return _default_manager.ensure_user_cache_dirs()


def fetch_leaderboard():
    return _default_manager.fetch_leaderboard()


def get_fetched_addresses():
    return _default_manager.get_fetched_addresses()


def add_fetched_address(address):
    return _default_manager.add_fetched_address(address)


def add_failed_address(address, error):
    return _default_manager.add_failed_address(address, error)


def load_user_data(address):
    return _default_manager.load_user_data(address)


def save_user_data(address, data, data_type=None):
    return _default_manager.save_user_data(address, data, data_type)


def add_address(address):
    return _default_manager.add_address(address)


def fetch_user_portfolio(address):
    return _default_manager.fetch_user_portfolio(address)


def fetch_fills(address: str, days: int = 30):
    return _default_manager.fetch_fills(address, days)


def fetch_fees(address: str):
    return _default_manager.fetch_fees(address)


def fetch_state(address: str, state_type: str = "perp"):
    return _default_manager.fetch_state(address, state_type)


def extract_addresses_from_leaderboard(leaderboard_data):
    return _default_manager.extract_addresses_from_leaderboard(leaderboard_data)


def fetch_user_addresses(
    max_addresses=MAX_ADDRESSES_TO_FETCH, addresses_force_fetch=[], show_progress=True
):
    return _default_manager.fetch_user_addresses(
        max_addresses, addresses_force_fetch, show_progress
    )


def get_all_cached_user_data(st, is_debug=False):
    return _default_manager.get_all_cached_user_data(st, is_debug)


def get_user_stats():
    return _default_manager.get_user_stats()
