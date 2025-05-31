import json
import os
import time
from datetime import datetime, timedelta

import requests
import streamlit as st

# URLs for user data
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
USER_INFO_URL = "https://api.hyperliquid.xyz/info"

USER_CACHE_DIR = "./user_cache/"
LEADERBOARD_CACHE_FILE = USER_CACHE_DIR + "leaderboard.json"
PROCESSED_ADDRESSES_FILE = USER_CACHE_DIR + "processed_addresses.txt"
FAILED_ADDRESSES_FILE = USER_CACHE_DIR + "failed_addresses.txt"
USER_DATA_DIR = USER_CACHE_DIR + "user_data/"

# Configuration
MAX_ADDRESSES_TO_PROCESS = 60
API_SLEEP_SECONDS = 1


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


def get_processed_addresses():
    """Get list of already processed addresses."""
    if not os.path.exists(PROCESSED_ADDRESSES_FILE):
        return set()
    
    with open(PROCESSED_ADDRESSES_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def add_processed_address(address):
    """Add address to processed list."""
    with open(PROCESSED_ADDRESSES_FILE, "a") as f:
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
        print(f"Cache found for {user_address}")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    print(f"Downloading portfolio for {user_address}")
    
    # Fetch user portfolio
    payload = {"type": "portfolio", "user": user_address}
    response = requests.post(USER_INFO_URL, json=payload)
    
    if response.status_code == 200:
        portfolio_data = response.json()
        
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(portfolio_data, f, indent=2)
        
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


def process_user_addresses(max_addresses=MAX_ADDRESSES_TO_PROCESS, show_progress=True):
    """Process user addresses from leaderboard."""
    ensure_user_cache_dirs()
    
    # Get leaderboard
    leaderboard_data = fetch_leaderboard()
    all_addresses = extract_addresses_from_leaderboard(leaderboard_data)
    
    # Get already processed addresses
    processed_addresses = get_processed_addresses()
    
    # Filter unprocessed addresses
    unprocessed_addresses = [addr for addr in all_addresses if addr not in processed_addresses]
    
    # Limit to max_addresses
    addresses_to_process = unprocessed_addresses[:max_addresses]
    
    print(f"Total addresses in leaderboard: {len(all_addresses)}")
    print(f"Already processed: {len(processed_addresses)}")
    print(f"Will process: {len(addresses_to_process)}")
    
    if not addresses_to_process:
        print("No new addresses to process")
        return []
    
    # Progress tracking for Streamlit
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    processed_data = []
    
    for i, address in enumerate(addresses_to_process):
        if show_progress:
            progress_bar.progress((i + 1) / len(addresses_to_process))
            status_text.text(f"Processing address {i+1}/{len(addresses_to_process)}: {address[:10]}...")
        
        try:
            # Fetch user portfolio
            portfolio_data = fetch_user_portfolio(address)
            
            # Add to processed list
            add_processed_address(address)
            
            # Store for return
            processed_data.append({
                "address": address,
                "portfolio": portfolio_data
            })
            
            print(f"Successfully processed {address}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to process {address}: {error_msg}")
            add_failed_address(address, error_msg)
        
        # Sleep to avoid rate limiting
        if i < len(addresses_to_process) - 1:  # Don't sleep after the last request
            time.sleep(API_SLEEP_SECONDS)
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
        st.toast(f"Processed {len(processed_data)} user addresses!", icon="✅")
    
    return processed_data


def get_all_cached_user_data():
    """Get all cached user data for analysis."""
    ensure_user_cache_dirs()
    
    user_data = []
    
    if not os.path.exists(USER_DATA_DIR):
        return user_data
    
    for filename in os.listdir(USER_DATA_DIR):
        if filename.endswith('.json'):
            address = filename[:-5]  # Remove .json extension
            filepath = os.path.join(USER_DATA_DIR, filename)
            
            try:
                with open(filepath, 'r') as f:
                    portfolio_data = json.load(f)
                    user_data.append({
                        "address": address,
                        "portfolio": portfolio_data
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return user_data


def get_user_stats():
    """Get statistics about processed users."""
    ensure_user_cache_dirs()
    
    # Count processed addresses
    processed_count = len(get_processed_addresses())
    
    # Count failed addresses
    failed_count = 0
    if os.path.exists(FAILED_ADDRESSES_FILE):
        with open(FAILED_ADDRESSES_FILE, "r") as f:
            failed_count = len([line for line in f if line.strip()])
    
    # Count cached data files
    cached_count = 0
    if os.path.exists(USER_DATA_DIR):
        cached_count = len([f for f in os.listdir(USER_DATA_DIR) if f.endswith('.json')])
    
    # Get total leaderboard size
    total_count = 0
    if os.path.exists(LEADERBOARD_CACHE_FILE):
        with open(LEADERBOARD_CACHE_FILE, "r") as f:
            leaderboard_data = json.load(f)
            total_count = len(extract_addresses_from_leaderboard(leaderboard_data))
    
    return {
        "total_addresses": total_count,
        "processed_addresses": processed_count,
        "failed_addresses": failed_count,
        "cached_data_files": cached_count
    }
