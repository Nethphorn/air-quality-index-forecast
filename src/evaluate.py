import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import get_datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data and Model
print("Loading test data and saved model...")
train_ds, val_ds, test_ds, scaler = get_datasets()
model = tf.keras.models.load_model('final_model.keras')

# DYNAMICALLY find the index for pm2.5
# Important: We must use the exact same scaler index used during training
# Fetch the list of features we processed
import pandas as pd
from data_loader import get_processed_data
df_raw = get_processed_data()
categorical_cols = df_raw.select_dtypes(include=['object', 'string']).columns
df_final = pd.get_dummies(df_raw, columns=categorical_cols)

target_idx = df_final.columns.get_loc('pm2.5')
print(f"Target column 'pm2.5' found at index: {target_idx}")

pm25_min = scaler.data_min_[target_idx]
pm25_max = scaler.data_max_[target_idx]

# Generate Predictions (Scaled Units)
print("Making predictions on test set...")
predictions_scaled = model.predict(test_ds)

# Inverse Scaling to Real Units
predictions = predictions_scaled * (pm25_max - pm25_min) + pm25_min

# Extract Actual Values from the Test Dataset
actuals_scaled = np.concatenate([y for x, y in test_ds], axis=0)
actuals = actuals_scaled * (pm25_max - pm25_min) + pm25_min

# Calculate Metrics
predictions = predictions.flatten()
actuals = actuals.flatten()

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f"\n--- Performance Summary (REAL UNITS) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} ug/m³")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} ug/m³")

# Visualization
plt.figure(figsize=(15, 6))
plt.plot(actuals[:250], label='Actual PM2.5', color='navy', linewidth=2)
plt.plot(predictions[:250], label='Predicted PM2.5', color='crimson', linestyle='--', linewidth=2)
plt.title('Air Quality Index Forecast', fontsize=14)
plt.ylabel('PM2.5 Concentration (ug/m³)', fontsize=12)
plt.xlabel('Time (Hours)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print some comparison rows
print("\n--- Sample Comparison (Actual vs Predicted) ---")
for i in range(10):
    print(f"Hour {i}: Actual={actuals[i]:.1f} | Predicted={predictions[i]:.1f}")
