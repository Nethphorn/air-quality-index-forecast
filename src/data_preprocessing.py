import pandas as pd
import numpy as np
import os
from data_loader import get_processed_data
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array


def get_datasets():
    # Load the data using our standardized loader
    df = get_processed_data()

    # One-hot encoding for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # Split the data chronologically BEFORE scaling to avoid data leakage
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    # Scaling (Fit ONLY on training data)
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # Transform all sets into numpy arrays of type float32
    train_series = scaler.transform(train_df).astype('float32')
    val_series = scaler.transform(val_df).astype('float32')
    test_series = scaler.transform(test_df).astype('float32')

    # 5. Parameters for windowing
    WINDOW_SIZE = 24
    BATCH_SIZE = 32   
    target_idx = df.columns.get_loc('pm2.5')

    # 6. Create Training, Validation, and Test datasets
    train_ds = timeseries_dataset_from_array(
        data=train_series,
        targets=train_series[WINDOW_SIZE:, target_idx],
        sequence_length=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = timeseries_dataset_from_array(
        data=val_series,
        targets=val_series[WINDOW_SIZE:, target_idx],
        sequence_length=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = timeseries_dataset_from_array(
        data=test_series,
        targets=test_series[WINDOW_SIZE:, target_idx],
        sequence_length=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Samples: Train={len(train_series)}, Val={len(val_series)}, Test={len(test_series)}")
    return train_ds, val_ds, test_ds, scaler