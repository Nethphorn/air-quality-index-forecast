import os
import logging
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Path Handling: Works in both Notebooks and Scripts
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# Point to the project root for the data folder
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, 'beijing_aqi.csv')

def get_standardized_data(cache_path=DEFAULT_CACHE_PATH, force_download=False):
    """
    Fetches, cleans, and standardizes the Beijing Multi-Site Air-Quality Data.
    Returns a cleaned DataFrame with a DatetimeIndex and interpolated PM2.5.
    """
    # 1. Check for Cache
    if not force_download and os.path.exists(cache_path):
        logger.info(f"Loading data from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_variable='datetime')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df

    # 2. Fetch from UCI
    logger.info("Fetching data from UCI repository...")
    try:
        beijing_multi_site_air_quality_data = fetch_ucirepo(id=501)
        # Combine all features into one DataFrame
        df = beijing_multi_site_air_quality_data.data.features
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return None

    # 3. Standardization & Cleaning
    logger.info("Standardizing data...")
    
    # Create Datetime Index
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    
    # Drop raw time columns as they are now in the index
    df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

    # Handle Missing Values in target (PM2.5) using Linear Interpolation
    missing_before = df['pm2.5'].isna().sum()
    df['pm2.5'] = df['pm2.5'].interpolate(method='linear')
    
    logger.info(f"Interpolated {missing_before} missing PM2.5 values.")

    # 4. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    logger.info(f"Data cached to {cache_path}")

    return df

if __name__ == "__main__":
    # Test run
    data = get_standardized_data()
    if data is not None:
        print("Data Loader successfully executed!")
        print(data.head())