import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Path management for cross-directory compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'beijing_aqi.csv')

def get_processed_data(cache_path=DEFAULT_CACHE_PATH):
    """
    Fetches, cleans, and caches the Beijing PM2.5 dataset.
    Uses absolute naming to avoid issues when imported from notebooks in different subdirectories.
    """
    # 1. Check if we already have the data locally
    if os.path.exists(cache_path):
        print(f"✅ Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col='datetime', parse_dates=True)

    # 2. If not, fetch it from UCI
    print("🚀 Downloading dataset from UCI Machine Learning Repository...")
    try:
        beijing_pm2_5 = fetch_ucirepo(id=381) 
        X = beijing_pm2_5.data.features 
        y = beijing_pm2_5.data.targets 
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

    # 3. Standardize and Clean
    df = pd.concat([X, y], axis=1)
    
    # Create the datetime index
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    
    # Interpolate missing values
    df['pm2.5'] = df['pm2.5'].interpolate(method='linear')

    # 4. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    print(f"📦 Data saved and cached to {cache_path}")

    return df

if __name__ == "__main__":
    # Test loading
    df = get_processed_data()
    if df is not None:
        print(df.head())