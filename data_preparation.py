import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_stock_data(symbol, period='3y'):
    """Download stock data"""
    # Check if downloaded NPY file exists
    npy_path = os.path.join("project_files", symbol, f'{symbol}_latest.npy')
    if os.path.exists(npy_path):
        print(f"Using downloaded data: {npy_path}")
        # Load NPY data
        close_values = np.load(npy_path)
        
        # Delete temporary NPY file
        os.remove(npy_path)
        
        # Create a simple DataFrame with only Close column
        dates = [datetime.now() - timedelta(days=len(close_values)-i) for i in range(len(close_values))]
        df = pd.DataFrame({
            'Close': close_values
        }, index=dates)
        
        return df
    else:
        # If NPY file not found, download directly
        print(f"Downloading {symbol} data directly...")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df

def prepare_data(df, sequence_length=60, forecast_days=3):
    """Generate multi-feature sequence data for multi-day forecasting"""
    # Use all 5 features: Close, High, Low, Open, Volume
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    data = {}
    norm_params = {}
    
    # Normalize each feature separately
    for feature in features:
        values = df[feature].values
        mean = np.mean(values)
        std = np.std(values)
        norm_params[feature] = (mean, std)
        data[feature] = (values - mean) / std
    
    X, y_close, y_high, y_low, y_open, y_volume = [], [], [], [], [], []
    
    # Create sequences
    for i in range(len(data['Close']) - sequence_length - forecast_days + 1):
        # Input sequence has all features
        features_sequence = []
        for feature in features:
            features_sequence.append(data[feature][i:i + sequence_length])
        
        X.append(np.column_stack(features_sequence))
        
        # Output has forecast for each feature for multiple days
        y_close.append(data['Close'][i + sequence_length:i + sequence_length + forecast_days])
        y_high.append(data['High'][i + sequence_length:i + sequence_length + forecast_days])
        y_low.append(data['Low'][i + sequence_length:i + sequence_length + forecast_days])
        y_open.append(data['Open'][i + sequence_length:i + sequence_length + forecast_days])
        y_volume.append(data['Volume'][i + sequence_length:i + sequence_length + forecast_days])
    
    X = np.array(X)
    y_close = np.array(y_close)
    y_high = np.array(y_high)
    y_low = np.array(y_low)
    y_open = np.array(y_open)
    y_volume = np.array(y_volume)
    
    y_dict = {
        'Close': y_close,
        'High': y_high,
        'Low': y_low,
        'Open': y_open,
        'Volume': y_volume
    }
    
    return X, y_dict, norm_params

def main():
    output_folder = "project_files"
    os.makedirs(output_folder, exist_ok=True)
    
    symbols = ['AAPL', 'MSFT']
    for symbol in symbols:
        print(f"Processing {symbol} data...")
        try:
            # Create symbol-specific directory
            symbol_dir = os.path.join(output_folder, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Download or use downloaded data
            df = download_stock_data(symbol)
            
            # Save raw data
            for feature in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if feature in df.columns:
                    raw_path = os.path.join(symbol_dir, f'full_history_{symbol}_{feature}.npy')
                    np.save(raw_path, df[feature].values)
                    print(f"Saved full history for {feature} to {raw_path}")
            
            # Prepare training data
            X, y_dict, norm_params = prepare_data(df)
            
            # Save processed data
            np.save(os.path.join(symbol_dir, f'X_{symbol}.npy'), X)
            
            for feature in y_dict.keys():
                np.save(os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy'), y_dict[feature])
            
            # Save normalization parameters for each feature
            np.save(os.path.join(symbol_dir, f'norm_params_{symbol}.npy'), norm_params)
            
            print(f"Successfully processed {symbol} data")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

if __name__ == "__main__":
    main()