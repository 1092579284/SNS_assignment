import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def create_model(input_shape, forecast_days=3):
    """Create multi-feature LSTM model for multi-day forecasting"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(forecast_days)  # Output forecast_days predictions
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.Huber(),
                 metrics=['mae', 'mse'])
    
    return model


def train_model(symbol, output_folder, feature, epochs=100, batch_size=64, forecast_days=3):
    """Train model for a specific feature"""
    symbol_dir = os.path.join(output_folder, symbol)
    X_path = os.path.join(symbol_dir, f'X_{symbol}.npy')
    y_path = os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy')
    model_path = os.path.join(symbol_dir, f'model_{symbol}_{feature}.keras')

    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    
    # Data split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create model
    model = create_model((X.shape[1], X.shape[2]), forecast_days)
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(model_path)
    print(f"Model for {feature} saved to {model_path}")
    return history

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    for symbol in symbols:
        symbol_dir = os.path.join(output_folder, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        print(f"\nTraining models for {symbol}...")
        if not os.path.exists(os.path.join(symbol_dir, f'X_{symbol}.npy')):
            print(f"Data not found for {symbol}")
            continue
            
        for feature in features:
            if os.path.exists(os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy')):
                print(f"Training {symbol} - {feature} model...")
                train_model(symbol, output_folder, feature)
            else:
                print(f"Data for {feature} not found")

if __name__ == "__main__":
    main()