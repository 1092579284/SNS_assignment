import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
from sklearn.multioutput import MultiOutputRegressor

def create_rf_model():
    """Create random forest regression model with multi-output support"""
    base_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    # Wrap with MultiOutputRegressor for multiple day predictions
    model = MultiOutputRegressor(base_rf)
    return model

def train_rf_model(symbol, output_folder, feature, n_estimators=100, max_depth=10):
    """Train random forest model for multi-day prediction"""
    symbol_dir = os.path.join(output_folder, symbol)
    X_path = os.path.join(symbol_dir, f'X_{symbol}.npy')
    y_path = os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy')
    model_path = os.path.join(symbol_dir, f'model_rf_{symbol}_{feature}.joblib')

    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    
    # Reshape 3D data to 2D (samples, features)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42
    )

    # Create model
    model = create_rf_model()
    
    # Train model
    print(f"Training RF model for {symbol} - {feature}...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R2 score: {train_score:.4f}")
    print(f"Test R2 score: {test_score:.4f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, train_score, test_score

def predict_rf(model, X_input, norm_params, feature):
    """Use random forest model to predict multiple days"""
    mean, std = norm_params[feature]
    
    # Normalize input and reshape
    X_reshaped = X_input.reshape(1, -1)
    
    # Predict
    predictions = model.predict(X_reshaped)[0]
    
    # Denormalize
    denormalized_predictions = predictions * std + mean
    
    return denormalized_predictions

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    for symbol in symbols:
        symbol_dir = os.path.join(output_folder, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        print(f"\nTraining RF models for {symbol}...")
        if not os.path.exists(os.path.join(symbol_dir, f'X_{symbol}.npy')):
            print(f"Data not found for {symbol}")
            continue
            
        for feature in features:
            if os.path.exists(os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy')):
                print(f"Training RF model for {symbol} - {feature}...")
                train_rf_model(symbol, output_folder, feature)
            else:
                print(f"Data for {feature} not found")

if __name__ == "__main__":
    main() 