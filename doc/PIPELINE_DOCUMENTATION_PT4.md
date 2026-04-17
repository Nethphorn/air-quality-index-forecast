# Air Quality Forecasting: Pipeline Part 4

This document details the rigorous evaluation framework and inference logic used to translate raw neural network outputs into actionable air quality insights.

---

## 1. Multi-Dimensional Performance Metrics

We don't rely on a single number to judge the model. Instead, we use a trio of metrics that provide different perspectives on the forecast's accuracy.

### Mean Absolute Error (MAE)

- **Calculation**: $\frac{1}{n} \sum |y_{actual} - y_{predicted}|$
- **Purpose**: Provides a direct measure of the average error in **ug/m³**. For example, an MAE of 10.56 means that, on average, the forecast is off by about 10 units.

### Root Mean Squared Error (RMSE)

- **Calculation**: $\sqrt{\frac{1}{n} \sum (y_{actual} - y_{predicted})^2}$
- **Significance**: Because RMSE squares the errors before averaging, it is highly sensitive to **large errors**. If the RMSE is significantly higher than the MAE (e.g., 17.7 vs 10.5), it tells us that while the model is usually accurate, it occasionally misses a pollution peak by a large margin.

### R-Squared ($R^2$) Score

- **Interpretation**: Measures the "Goodness of Fit." An $R^2$ of 0.96 means the model captures **96% of the patterns** seen in the data. Only 4% of the fluctuations are treated as unexplained noise.

---

## 2. Inverse Transformation & Unit Recovery

During training, all features are squashed into the `[0, 1]` range (Normalization). This is essential for the LSTM's `tanh` and `sigmoid` activation functions to work effectively without gradients exploding.

However, a prediction of `0.15` is useless for a public health official. We must perform **Inverse Scaling**.

### The Math:

The `MinMaxScaler` stores the minimum and maximum values seen during the **Training Phase** (fitting on training data to prevent leakage). In `evaluate.py`, we apply the reverse formula:

$$Value_{Original} = Value_{Scaled} \times (Max - Min) + Min$$

By doing this, we recover the exact concentration levels in $ug/m³$, allowing for direct comparison against environmental standards.

---

## 3. Sliding Window Inference (The Predictor)

In `src/model1/predict.py`, we implement a "Live" sliding window technique. This is the process of using the most recent block of time to predict the immediate future.

### The Process:

1. **Buffer Retrieval**: We take the absolute last 24 entries from the cleaned dataset.
2. **Preprocessing Sync**: We apply the **exact same Scaler** instance used during training. This ensures that a PM2.5 of 200 is represented by the same decimal value the model learned.
3. **Ghost Prediction**: The model outputs a decimal.
4. **Rescaling**: We convert that decimal to the final AQI forecast.

This methodology allows the model to be used in a real-time production app where data arrives hourly.

---

## 4. Visual Analysis Rationale

The "Actual vs Predicted" plot is our most important qualitative tool.

- **Lag Analysis**: We look for "Horizontal Lag." If the predicted peaks appear an hour later than the actual peaks, the model is merely "copying" the previous value (Auto-regression failure). Our current model shows tight alignment, indicating true predictive capability.
- **Damping**: We check if the predicted peaks are much shorter than the actual peaks. This "Damping" is common in LSTMs and is a target for future hyperparameter tuning.

---

## 🔧 File Reference

- `src/model1/evaluate.py`: The validation engine.
- `src/model1/predict.py`: The production-ready inference engine.
- `doc/img/air_quality_index_forecast.png`: The visual proof of performance.
