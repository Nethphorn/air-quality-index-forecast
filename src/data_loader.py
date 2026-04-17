import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'beijing_aqi.csv')

def get_processed_data(cache_path=DEFAULT_CACHE_PATH):
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col='datetime', parse_dates=True)

    print("Downloading dataset from UCI Machine Learning Repository...")
    try:
        beijing_pm2_5 = fetch_ucirepo(id=381) 
        X = beijing_pm2_5.data.features 
        y = beijing_pm2_5.data.targets 
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    df = pd.concat([X, y], axis=1)
    
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    
    df['pm2.5'] = df['pm2.5'].interpolate(method='linear')
    df = df.dropna(subset=['pm2.5'])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    print(f"Data saved and cached to {cache_path}")

    return df

if __name__ == "__main__":
    df = get_processed_data()
    if df is not None:
        print(df.head())
        print(df['pm2.5'].isna().sum())