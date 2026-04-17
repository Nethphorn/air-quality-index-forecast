import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, CategoryEncoding
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_loader import get_processed_data

def transform_data(df, n_lags=3):
    """
    Transforms raw dataframe into features (X) and target (y).
    Includes temporal lagging and TensorFlow one-hot encoding.
    """
    # 1. Create Lag Features
    df_n = df.copy()
    for i in range(1, n_lags + 1):
        df_n[f'pm2.5_lag_{i}'] = df_n['pm2.5'].shift(i)
    df_lagged = df_n.dropna()

    # 2. TensorFlow Encoding for Wind Direction (cbwd)
    vocab = df_lagged['cbwd'].unique().tolist()
    lookup = StringLookup(vocabulary=vocab, output_mode='int')
    encoder = CategoryEncoding(num_tokens=len(vocab) + 1, output_mode="one_hot")

    wind_tensor = tf.constant(df_lagged['cbwd'].values)
    encoded_wind = encoder(lookup(wind_tensor)).numpy()

    # 3. Reconstruct DataFrame
    encoded_cols = [f"cbwd_{v}" for v in vocab] + ["cbwd_unknown"]
    wind_df = pd.DataFrame(encoded_wind, index=df_lagged.index, columns=encoded_cols)
    
    df_final = pd.concat([df_lagged.drop(columns=['cbwd']), wind_df], axis=1)
    
    X = df_final.drop(columns=['pm2.5'])
    y = df_final['pm2.5']
    
    return X, y

def train_baseline_model(X, y):
    """
    Splits data chronologically and trains a Random Forest Regressor.
    """
    # Chronological split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions)
    }
    
    return model, metrics

if __name__ == "__main__":
    print("--- Starting Modeling Pipeline ---")
    data = get_processed_data()
    X, y = transform_data(data)
    
    model, scores = train_baseline_model(X, y)
    
    print(f"Model Training Complete.")
    print(f"Mean Absolute Error: {scores['mae']:.2f}")
    print(f"R2 Score: {scores['r2']:.2f}")