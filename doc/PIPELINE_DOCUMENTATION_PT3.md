# Air Quality Forecasting: Pipeline Part 3

This document provides a highly detailed technical breakdown of the model architecture and training configuration used for the Air Quality Index (AQI) forecast.

---

## 1. Sequence Modeling: Why LSTM?

Traditional neural networks (Dense layers) process inputs independently. However, air quality is inherently **autoregressive**—the pollution level at Hour 10 is strongly influenced by Hour 9, 8, and so on.

We utilize **Long Short-Term Memory (LSTM)** units because they solve the "vanishing gradient" problem found in standard Recurrent Neural Networks (RNNs). The LSTM's **Cell State** acts as a long-term memory track, allowing the model to carry information across the entire 24-hour sequence without losing the signal from the early hours.

### Input Specification

The model expects data in a 3D tensor format: `(Batch Size, 24, Number of Features)`.

- **24**: Represents the temporal lookback window (the previous 24 hours).
- **Number of Features**: Includes meteorological data, temporal markers, and historical PM2.5 levels.

---

## 2. Micro-Architecture Breakdown

The design follows a "Stacked LSTM" approach with aggressive regularization to ensure the model generalizes to unseen weather patterns.

### First Layer: Feature Extraction

- **LSTM (64 units)**: Acts as the primary feature extractor.
- **return_sequences=True**: Mandatory for stacking. This ensures that the output is still a 3D sequence, allowing the second LSTM layer to see the temporal evolution of the features extracted by the first.

### Second Layer: Temporal Aggregation

- **LSTM (32 units)**: Compresses the 64-dimensional feature space into a 32-dimensional temporal summary.
- **return_sequences=False**: Only the final hidden state of the sequence is passed forward. This transforms the 3D sequence into a 2D vector for the final regression.

### Regularization: Dropout (0.2)

By randomly dropping 20% of the neurons during each training step, we force the model to find **redundant patterns**. This is critical for air quality data, where sensor noise or missing values (interpolated) could otherwise lead to overfitting.

---

## 3. Mathematical Optimization (Compilation)

The compilation step defines how the model "learns" from its mistakes.

### Loss Function: Mean Squared Error (MSE)

While we report MAE for human readability, we optimize for **MSE**.

- **Formula**: $\frac{1}{n} \sum (y_{actual} - y_{predicted})^2$
- **Rationale**: Since errors are squared, large misses in pollution spikes are penalized much more heavily than small misses. This ensures the model prioritizes staying close to dangerous pollution peaks.

### Optimizer: Adam (Learning Rate = 0.001)

Adam is used for its **Adaptive Moment Estimation**. It maintains a separate learning rate for each weight, which is particularly helpful for sparse features or data with varying scales (like wind speed vs. pressure).

---

## 4. Training Governance (Callbacks)

We use an automated governance system to stop training the moment the model begins to "memorize" the training set rather than learning features.

### EarlyStopping

- **Monitor**: `val_loss`
- **Patience**: `5`
- **restore_best_weights=True**: This is a critical detail. If the model starts to overfit at Epoch 10 and stops at Epoch 15, this callback will automatically roll back the weights to the superior version from Epoch 10.

### ModelCheckpoint

- **save_best_only=True**: Only overwrites the file if the validation loss improves. This creates a "Golden Model" artifact (`best_model.keras`) that represents the absolute peak of the model's predictive power.

---

## 🔧 File Reference

- `src/model1/train.py`: Implementation of the stack and training loop.
- `src/model1/data_preprocessing.py`: Provides the windowed tensors for the input shape.
