# Air Quality Forecasting: ML Pipeline Documentation

This document summarizes the data engineering journey and the code used to prepare the Beijing PM2.5 dataset for Deep Learning.

---

## 1. Data Ingestion & Modularization

We moved our data logic out of the notebook and into `src/data_loader.py` to make it reusable and repeatable.

**Why:** To implement **Caching** (avoiding repeated downloads) and **Absolute Paths** (so the code works from any folder).

```python
from src.data_loader import get_processed_data

# Fetches from UCI, cleans, handles NaNs, and saves locally
df = get_processed_data()
```

---

## 2. Categorical Encoding (One-Hot)

Deep Learning models only understand numbers. We must convert text labels like wind direction ('NW', 'SE') into binary columns.

**Why:** This prevents the model from assuming a mathematical order (like thinking 'SE' is 4x better than 'NW') while still capturing the information.

```python
# Select text columns and create dummy variables (0 or 1)
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
df = pd.get_dummies(df, columns=categorical_cols)
```

---

## 3. Min-Max Scaling

We squish all numerical values into a standard range, typically between `0` and `1`.

**Why:** If one feature has a range of 1000 and another has 0.1, the model will struggle to learn. Scaling puts every feature on a level playing field.

```python
# Initialize scaler and transform all numeric columns
scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])
```

---

## 4. Time-Series Windowing

This is the "special sauce" for forecasting. We turn a flat list of hours into overlapping "windows" of history.

**Why:** An LSTM model needs to see a _sequence_ of past hours (e.g., the last 24 hours) to predict what happens in the next hour.

```python
# Using a 24-hour window to predict the 25th hour
WINDOW_SIZE = 24
BATCH_SIZE = 32
target_idx = df.columns.get_loc('pm2.5')

dataset = timeseries_dataset_from_array(
    data=series,
    targets=series[WINDOW_SIZE:, target_idx],
    sequence_length=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

**References:**

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-Learn Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)
- [TensorFlow Timeseries API](https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array)
