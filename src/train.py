from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import get_datasets

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model():
    train_ds, val_ds, test_ds, scaler = get_datasets()
    
    # Get input shape from the first batch
    for x, y in train_ds.take(1):
        input_shape = x.shape[1:]
        break
    
    model = build_model(input_shape)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return model, history, scaler

if __name__ == "__main__":
    model, history, scaler = train_model()
    model.save('final_model.keras')
    print("Model saved to final_model.keras")