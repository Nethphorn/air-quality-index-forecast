# Air Quality Forecasting: Pipeline Part 2

This document covers the advanced preparation steps required for high-integrity Machine Learning models, specifically focusing on data splitting and data leakage prevention.

---

## 1. Chronological Data Splitting

Unlike traditional ML problems, time-series data **cannot be shuffled** randomly before splitting. We must split the data in order of time.

**Why:** We want to train on the "past" and test on the "future." Randomly shuffling would allow the model to see the future during training, which is impossible in a real-world application.

```python
n = len(df)
train_df = df[0:int(n*0.7)]      # First 70%
val_df = df[int(n*0.7):int(n*0.9)] # Next 20%
test_df = df[int(n*0.9):]        # Final 10%
```

---

## 2. Preventing Data Leakage

One of the most common mistakes in ML is "leakage." This happens when information from the future "leaks" into your training phase via the preprocessing steps.

**The Fix:** We **fit** the `MinMaxScaler` only on the training data. This ensures the model's environment is consistent with real-world forecasting where you don't know the future statistics yet.

```python
# 1. Learn the scale from the PAST only
scaler = MinMaxScaler()
scaler.fit(train_df)

# 2. Apply that scale to the FUTURE
train_series = scaler.transform(train_df)
val_series = scaler.transform(val_df)
test_series = scaler.transform(test_df)
```

---

## 3. Creating Multiple Datasets

For a complete training cycle, we need three distinct windowed datasets.

- **Train DS:** Used to update the model's weights during training.
- **Val DS:** Used as a benchmark to tune hyperparameters and detect overfitting.
- **Test DS:** The final "exam" that provides the cold, hard metrics for your model.

```python
# Using a 24-hour window for each split
train_ds = timeseries_dataset_from_array(
    data=train_series,
    targets=train_series[24:, target_idx],
    sequence_length=24,
    batch_size=32,
    shuffle=True
)

val_ds = timeseries_dataset_from_array(data=val_series, ...)
test_ds = timeseries_dataset_from_array(data=test_series, ...)
```

**References:**

- [Scikit-Learn: Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [TensorFlow Tutorial: Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
