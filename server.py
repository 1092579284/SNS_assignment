from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
import subprocess
import threading


app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

class OracleServer:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.models = {}  # LSTM models
        self.rf_models = {}  # Random Forest models
        self.norm_params = {}
        self.history_data = {}
        self.symbols = ['AAPL', 'MSFT']
        self.features = ['Close', 'High', 'Low', 'Open', 'Volume']
        self.output_folder = "project_files"
        self.updating = False
        self.update_status = {"step": "", "message": "", "progress": 0}
        self.load_resources()
        
    def load_resources(self):
        """Load all necessary resources for each symbol and feature"""
        for symbol in self.symbols:
            try:
                symbol_dir = os.path.join(self.output_folder, symbol)
                if not os.path.exists(symbol_dir):
                    os.makedirs(symbol_dir, exist_ok=True)
                    print(f"Created directory for {symbol}")
                    continue
                
                # Initialize data structures for this symbol
                self.models[symbol] = {}
                self.rf_models[symbol] = {}
                self.history_data[symbol] = {}
                
                # Load norm parameters
                norm_path = os.path.join(symbol_dir, f'norm_params_{symbol}.npy')
                if os.path.exists(norm_path):
                    self.norm_params[symbol] = np.load(norm_path, allow_pickle=True).item()
                    print(f"Loaded normalization parameters for {symbol}")
                
                # Load history data for each feature
                for feature in self.features:
                    history_path = os.path.join(symbol_dir, f'full_history_{symbol}_{feature}.npy')
                    if os.path.exists(history_path):
                        self.history_data[symbol][feature] = np.load(history_path)
                        print(f"Loaded {feature} history for {symbol}")
                    
                    # Load LSTM model for this feature
                    model_path = os.path.join(symbol_dir, f'model_{symbol}_{feature}.keras')
                    if os.path.exists(model_path):
                        self.models[symbol][feature] = load_model(model_path)
                        print(f"Loaded LSTM model for {symbol} - {feature}")
                    
                    # Load RF model for this feature
                    rf_model_path = os.path.join(symbol_dir, f'model_rf_{symbol}_{feature}.joblib')
                    if os.path.exists(rf_model_path):
                        self.rf_models[symbol][feature] = joblib.load(rf_model_path)
                        print(f"Loaded RF model for {symbol} - {feature}")
                
                print(f"Successfully loaded resources for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def download_latest_data(self, symbol=None):
        """Download latest stock data for one or all symbols"""
        self.update_status = {"step": "download", "message": "Downloading latest stock data...", "progress": 20}
        
        # Ensure output directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        if symbol is not None:
            symbols_to_process = [symbol]
        else:
            symbols_to_process = self.symbols
        
        success = True
        for sym in symbols_to_process:
            try:
                symbol_dir = os.path.join(self.output_folder, sym)
                os.makedirs(symbol_dir, exist_ok=True)
                
                # Download latest data using yfinance
                print(f"Downloading latest data for {sym}...")
                stock = yf.Ticker(sym)
                df = stock.history(period='3y')  # Download 3 years of data
                
                if df.empty:
                    print(f"No data found for {sym}")
                    continue
                
                # Save raw data for each feature
                for feature in self.features:
                    if feature in df.columns:
                        raw_path = os.path.join(symbol_dir, f'{sym}_latest_{feature}.npy')
                        np.save(raw_path, df[feature].values)
                        
                        # Also save full history
                        hist_path = os.path.join(symbol_dir, f'full_history_{sym}_{feature}.npy')
                        np.save(hist_path, df[feature].values)
                        
                        print(f"Saved {feature} data for {sym}")
                
                print(f"Successfully downloaded data for {sym}")
            except Exception as e:
                print(f"Error downloading {sym} data: {str(e)}")
                success = False
        
        return success
    
    def update_data(self, specific_symbol=None):
        """Update data and models for one or all symbols"""
        if self.updating:
            return False, "Data update already in progress, please try again later"
        
        self.updating = True
        self.update_status = {"step": "start", "message": "Starting data update...", "progress": 0}
        
        try:
            # Step 1: Download latest data
            print(f"Starting download for {specific_symbol if specific_symbol else 'all'} stocks...")
            if not self.download_latest_data(specific_symbol):
                error_msg = "Failed to download latest data"
                self.update_status = {"step": "error", "message": error_msg, "progress": 0}
                self.updating = False
                return False, error_msg
            
            # Step 2: Data preprocessing
            self.update_status = {"step": "prepare", "message": "Preprocessing data...", "progress": 40}
            print(f"Starting data preprocessing... Symbol: {specific_symbol if specific_symbol else 'all'}")
            
            # Check if data preparation script exists
            if not os.path.exists('data_preparation.py'):
                error_msg = "Data preparation script data_preparation.py not found"
                print(error_msg)
                self.update_status = {"step": "error", "message": error_msg, "progress": 40}
                self.updating = False
                return False, error_msg
            
            # If updating a specific symbol, pass it as an argument
            cmd_args = ['python', 'data_preparation.py']
            if specific_symbol:
                cmd_args.append(specific_symbol)
                
            print(f"Executing command: {' '.join(cmd_args)}")
            result1 = subprocess.run(cmd_args, capture_output=True, text=True, check=False)
            
            # Print standard output and error
            print("Data preprocessing standard output:")
            print(result1.stdout)
            
            if result1.stderr:
                print("Data preprocessing error output:")
                print(result1.stderr)
            
            if result1.returncode != 0:
                error_msg = f"Data preprocessing failed: {result1.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "Data preprocessing failed", "progress": 40}
                self.updating = False
                return False, error_msg
            
            # Step 3: Train LSTM models
            self.update_status = {"step": "lstm", "message": "Training LSTM models...", "progress": 60}
            print(f"Starting LSTM model training... Symbol: {specific_symbol if specific_symbol else 'all'}")
            
            # Check if LSTM model training script exists
            if not os.path.exists('train_model.py'):
                error_msg = "LSTM model training script train_model.py not found"
                print(error_msg)
                self.update_status = {"step": "error", "message": error_msg, "progress": 60}
                self.updating = False
                return False, error_msg
            
            # Build LSTM training command
            cmd_args = ['python', 'train_model.py']
            if specific_symbol:
                cmd_args.append(specific_symbol)
                
            print(f"Executing command: {' '.join(cmd_args)}")
            result2 = subprocess.run(cmd_args, capture_output=True, text=True, check=False)
            
            # Print standard output and error
            print("LSTM model training standard output:")
            print(result2.stdout)
            
            if result2.stderr:
                print("LSTM model training error output:")
                print(result2.stderr)
            
            if result2.returncode != 0:
                error_msg = f"LSTM model training failed: {result2.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "LSTM model training failed", "progress": 60}
                self.updating = False
                return False, error_msg
            
            # Step 4: Train Random Forest models
            self.update_status = {"step": "rf", "message": "Training Random Forest models...", "progress": 80}
            print(f"Starting Random Forest model training... Symbol: {specific_symbol if specific_symbol else 'all'}")
            
            # Check if Random Forest model training script exists
            if not os.path.exists('train_model_rf.py'):
                error_msg = "Random Forest model training script train_model_rf.py not found"
                print(error_msg)
                self.update_status = {"step": "error", "message": error_msg, "progress": 80}
                self.updating = False
                return False, error_msg
            
            # Build Random Forest training command
            cmd_args = ['python', 'train_model_rf.py']
            if specific_symbol:
                cmd_args.append(specific_symbol)
                
            print(f"Executing command: {' '.join(cmd_args)}")
            result3 = subprocess.run(cmd_args, capture_output=True, text=True, check=False)
            
            # Print standard output and error
            print("Random Forest model training standard output:")
            print(result3.stdout)
            
            if result3.stderr:
                print("Random Forest model training error output:")
                print(result3.stderr)
            
            if result3.returncode != 0:
                error_msg = f"Random Forest model training failed: {result3.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "Random Forest model training failed", "progress": 80}
                self.updating = False
                return False, error_msg
            
            # Step 5: Verify the model files exist
            if specific_symbol:
                symbol_dir = os.path.join(self.output_folder, specific_symbol)
                lstm_model_exists = False
                rf_model_exists = False
                
                for feature in self.features:
                    lstm_model_path = os.path.join(symbol_dir, f'model_{specific_symbol}_{feature}.keras')
                    rf_model_path = os.path.join(symbol_dir, f'model_rf_{specific_symbol}_{feature}.joblib')
                    
                    if os.path.exists(lstm_model_path):
                        lstm_model_exists = True
                        print(f"Found LSTM model file: {lstm_model_path}")
                    else:
                        print(f"LSTM model file not found: {lstm_model_path}")
                        
                    if os.path.exists(rf_model_path):
                        rf_model_exists = True
                        print(f"Found Random Forest model file: {rf_model_path}")
                    else:
                        print(f"Random Forest model file not found: {rf_model_path}")
                
                if not lstm_model_exists or not rf_model_exists:
                    print(f"Warning: Not all expected model files were found, but will continue to try loading available models")
            
            # Step 6: Reload resources
            self.update_status = {"step": "reload", "message": "Reloading models and data...", "progress": 95}
            print("Reloading models and data...")
            
            # Save existing symbols list
            existing_symbols = list(self.symbols)
            
            # If it's a new symbol, ensure it's added to the list
            if specific_symbol and specific_symbol not in existing_symbols:
                existing_symbols.append(specific_symbol)
                print(f"Adding new symbol {specific_symbol} to symbols list")
            
            self.load_resources()
            
            # Verify if models were successfully loaded
            if specific_symbol:
                if specific_symbol not in self.models:
                    print(f"Warning: Models for symbol {specific_symbol} not found after reload")
                else:
                    model_count = len(self.models.get(specific_symbol, {}))
                    rf_model_count = len(self.rf_models.get(specific_symbol, {}))
                    print(f"Loaded {model_count} LSTM models and {rf_model_count} Random Forest models for {specific_symbol}")
            
            # Update complete
            self.update_status = {"step": "complete", "message": "Data and models successfully updated", "progress": 100}
            self.updating = False
            return True, "Data and models successfully updated"
        
        except Exception as e:
            self.updating = False
            error_msg = f"Error during update process: {str(e)}"
            self.update_status = {"step": "error", "message": error_msg, "progress": 0}
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg
    
    def async_update_data(self, symbol=None):
        """Asynchronously execute data update"""
        threading.Thread(target=lambda: self._async_update_worker(symbol)).start()
        return True, "Data update started, please wait..."
    
    def _async_update_worker(self, symbol=None):
        """Asynchronous update worker thread"""
        self.update_data(symbol)
    
    def get_update_status(self):
        """Get current update status"""
        return self.update_status
    
    def get_available_symbols(self):
        """Get list of available symbols"""
        return self.symbols
    
    def predict_multiple_days(self, symbol, feature, days=3, model_type='lstm'):
        """Predict multiple days for a specific feature"""
        try:
            if symbol not in self.symbols:
                print(f"Symbol {symbol} not found in available symbols")
                return None
                
            if model_type == 'lstm':
                if symbol not in self.models or feature not in self.models[symbol]:
                    print(f"LSTM model for {symbol}-{feature} not found")
                    return None
                
                # Get the latest data
                if symbol not in self.norm_params or feature not in self.history_data[symbol]:
                    print(f"Normalization params or history data for {symbol}-{feature} not found")
                    return None
                    
                # Get normalization parameters and history data
                norm_params = self.norm_params[symbol]
                
                try:
                    # Check if the feature exists in norm_params
                    if feature not in norm_params:
                        print(f"Feature {feature} not found in normalization parameters for {symbol}")
                        return None
                        
                    last_sequence = self.history_data[symbol][feature][-60:]
                    
                    # Create input sequence with all features
                    input_sequence = []
                    feature_count = 0
                    
                    for feat in self.features:
                        if feat in self.history_data[symbol]:
                            # Verify we have enough data
                            if len(self.history_data[symbol][feat]) < 60:
                                print(f"Not enough history data for {symbol}-{feat}: only {len(self.history_data[symbol][feat])} points")
                                continue
                                
                            feat_data = self.history_data[symbol][feat][-60:]
                            # Normalize only if we have normalization parameters
                            if feat in norm_params:
                                mean, std = norm_params[feat]
                                # Avoid division by zero
                                if std == 0:
                                    std = 1
                                feat_data = (feat_data - mean) / std
                            input_sequence.append(feat_data)
                            feature_count += 1
                    
                    # Make sure we have at least one feature
                    if feature_count == 0:
                        print(f"No valid features found for {symbol}")
                        return None
                    
                    # Stack features to create multi-feature input
                    input_array = np.column_stack(input_sequence)
                    input_shape = (1, 60, feature_count)
                    print(f"Input shape for {symbol}: {input_shape}")
                    input_array = input_array.reshape(input_shape)
                    
                    # Use a try block specifically for prediction
                    try:
                        # Ensure input is float32 to avoid retracing issues
                        input_array = input_array.astype(np.float32)
                        
                        # Create a function once for prediction to minimize retracing
                        if not hasattr(self, '_predict_cache'):
                            self._predict_cache = {}
                            
                        # Create or reuse cached prediction function
                        cache_key = f"{symbol}_{feature}"
                        if cache_key not in self._predict_cache:
                            @tf.function(reduce_retracing=True)
                            def _predict_fn(inputs):
                                return self.models[symbol][feature](inputs, training=False)
                            
                            self._predict_cache[cache_key] = _predict_fn
                        
                        # Use cached prediction function
                        pred_fn = self._predict_cache[cache_key]
                        predictions = pred_fn(input_array).numpy()[0]
                    except Exception as pred_err:
                        print(f"Prediction error for {symbol}-{feature}: {str(pred_err)}")
                        print(f"Model input shape: {input_array.shape}")
                        print(f"Model expects input shape: {self.models[symbol][feature].input_shape}")
                        return None
                        
                    # Denormalize
                    mean, std = norm_params[feature]
                    # Avoid division by zero
                    if std == 0:
                        std = 1
                    denorm_predictions = predictions * std + mean
                    
                    # Error correction - compare with last known price
                    last_known_price = self.history_data[symbol][feature][-1]
                    if abs(denorm_predictions[0] - last_known_price) / last_known_price > 0.05:
                        correction = last_known_price / denorm_predictions[0]
                        denorm_predictions = denorm_predictions * correction
                    
                    return denorm_predictions
                except Exception as inner_e:
                    print(f"Error preparing data for prediction on {symbol}-{feature}: {str(inner_e)}")
                    return None
                
            elif model_type == 'rf':
                if symbol not in self.rf_models or feature not in self.rf_models[symbol]:
                    print(f"RF model for {symbol}-{feature} not found")
                    return None
                
                # Get normalization parameters and history data
                if symbol not in self.norm_params or feature not in self.history_data[symbol]:
                    print(f"Normalization params or history data for {symbol}-{feature} not found")
                    return None
                
                try:
                    norm_params = self.norm_params[symbol]
                    
                    # Check if the feature exists in norm_params
                    if feature not in norm_params:
                        print(f"Feature {feature} not found in normalization parameters for {symbol}")
                        return None
                    
                    # Prepare input for RF model
                    input_sequence = []
                    feature_count = 0
                    
                    for feat in self.features:
                        if feat in self.history_data[symbol]:
                            # Verify we have enough data
                            if len(self.history_data[symbol][feat]) < 60:
                                print(f"Not enough history data for {symbol}-{feat}: only {len(self.history_data[symbol][feat])} points")
                                continue
                                
                            feat_data = self.history_data[symbol][feat][-60:]
                            # Normalize
                            if feat in norm_params:
                                mean, std = norm_params[feat]
                                # Avoid division by zero
                                if std == 0:
                                    std = 1
                                feat_data = (feat_data - mean) / std
                            input_sequence.append(feat_data)
                            feature_count += 1
                    
                    # Make sure we have at least one feature
                    if feature_count == 0:
                        print(f"No valid features found for {symbol}")
                        return None
                    
                    # Stack features to create multi-feature input
                    input_array = np.column_stack(input_sequence)
                    input_array = input_array.reshape(1, -1)
                    
                    # Predict with RF model
                    rf_model = self.rf_models[symbol][feature]
                    if rf_model is None:
                        print(f"RF model is None for {symbol}-{feature}")
                        return None
                        
                    # Predict with try/except
                    try:
                        predictions = rf_model.predict(input_array)[0]
                    except Exception as rf_err:
                        print(f"RF prediction error for {symbol}-{feature}: {str(rf_err)}")
                        return None
                    
                    # Denormalize
                    mean, std = norm_params[feature]
                    # Avoid division by zero
                    if std == 0:
                        std = 1
                    denorm_predictions = predictions * std + mean
                    
                    # Error correction
                    last_known_price = self.history_data[symbol][feature][-1]
                    if abs(denorm_predictions[0] - last_known_price) / last_known_price > 0.05:
                        correction = last_known_price / denorm_predictions[0]
                        denorm_predictions = denorm_predictions * correction
                    
                    return denorm_predictions
                except Exception as inner_e:
                    print(f"Error preparing data for RF prediction on {symbol}-{feature}: {str(inner_e)}")
                    return None
                
            return None
            
        except Exception as e:
            print(f"Prediction error ({model_type}) for {symbol}-{feature}: {str(e)}")
            return None
    
    def get_plot_data(self, symbol, target_feature=None, target_date=None):
        """Generate plot data including predictions for all features"""
        if symbol not in self.symbols:
            return None
            
        features_to_plot = [target_feature] if target_feature else self.features
        result = {'features': {}}
        
        # Get dates for x-axis
        current_date = datetime.now()
        dates = [
            (current_date - timedelta(days=30-i)).strftime('%Y-%m-%d')
            for i in range(30)
        ]
        
        # Add future dates
        future_dates = [
            (current_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            for i in range(3)
        ]
        
        result['dates'] = dates + future_dates
        
        # Add history and predictions for each requested feature
        for feature in features_to_plot:
            if feature not in self.history_data[symbol]:
                continue
                
            # Get history data (last 30 days)
            history = self.history_data[symbol][feature][-30:].tolist()
            
            # Get predictions for both models
            lstm_predictions = self.predict_multiple_days(symbol, feature, days=3, model_type='lstm')
            rf_predictions = self.predict_multiple_days(symbol, feature, days=3, model_type='rf')
            
            # Determine which day to highlight (if any)
            highlight_index = None
            if target_date:
                try:
                    target_date_obj = self._parse_date(target_date)
                    for i, date_str in enumerate(future_dates):
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        if date_obj.date() == target_date_obj.date():
                            highlight_index = 30 + i  # 30 days of history + index in future
                            break
                except:
                    pass
            
            result['features'][feature] = {
                'history': history,
                'lstm_predictions': lstm_predictions.tolist() if lstm_predictions is not None else None,
                'rf_predictions': rf_predictions.tolist() if rf_predictions is not None else None,
                'highlight_index': highlight_index
            }
        
        return result
    
    def _parse_date(self, date_string):
        """Parse date string into datetime object"""
        try:
            # Try various formats
            today = datetime.now()
            
            # Handle relative dates
            if date_string.lower() == 'tomorrow':
                return today + timedelta(days=1)
            elif date_string.lower() == 'day after tomorrow':
                return today + timedelta(days=2)
            elif date_string.lower() == 'next day':
                return today + timedelta(days=1)
            elif date_string.lower() in ['in 2 days', 'in two days']:
                return today + timedelta(days=2)
            elif date_string.lower() in ['in 3 days', 'in three days']:
                return today + timedelta(days=3)
                
            # Try to parse as explicit date
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d-%m-%Y', '%d/%m/%Y'):
                try:
                    return datetime.strptime(date_string, fmt)
                except:
                    pass
                    
            # If all parsing attempts fail
            return None
        except:
            return None
    
    def process_request(self, request):
        """Process natural language request and extract symbol, feature, and date"""
        try:
            req = request.lower().strip()
            
            # Extract symbol
            symbol = None
            for sym in self.symbols:
                if sym.lower() in req:
                    symbol = sym
                    break
                    
            if not symbol:
                # Try to extract company names
                company_mapping = {
                    'apple': 'AAPL',
                    'microsoft': 'MSFT',
                    'google': 'GOOGL',
                    'alphabet': 'GOOGL',
                    'amazon': 'AMZN',
                    'netflix': 'NFLX',
                    'facebook': 'META',
                    'meta': 'META',
                    'tesla': 'TSLA',
                    'ibm': 'IBM',
                    'intel': 'INTC',
                    'amd': 'AMD',
                    'nvidia': 'NVDA',
                    'oracle': 'ORCL',
                    'cisco': 'CSCO',
                    'adobe': 'ADBE',
                    'salesforce': 'CRM'
                }
                
                for company, sym in company_mapping.items():
                    if company in req:
                        symbol = sym
                        break
            
            if not symbol:
                return json.dumps({'error': 'Please specify a valid company/stock symbol'})
            
            print(f"Processing request for symbol: {symbol}")
            
            # Check if we have trained models for this symbol
            if symbol not in self.models or not self.models[symbol]:
                error_msg = f"Models for {symbol} not ready yet. Please wait for the data processing and model training to complete."
                print(error_msg)
                return json.dumps({'error': error_msg})
            
            # Extract feature
            feature = None
            feature_keywords = {
                'close': 'Close',
                'closing': 'Close',
                'high': 'High',
                'highest': 'High',
                'low': 'Low',
                'lowest': 'Low',
                'open': 'Open',
                'opening': 'Open',
                'volume': 'Volume'
            }
            
            for keyword, feat in feature_keywords.items():
                if keyword in req:
                    feature = feat
                    break
            
            # Extract date
            date_keywords = [
                'tomorrow', 'day after tomorrow', 'next day',
                'in 2 days', 'in two days', 'in 3 days', 'in three days'
            ]
            
            target_date = None
            for date_kw in date_keywords:
                if date_kw in req:
                    target_date = date_kw
                    break
            
            print(f"Request details: symbol={symbol}, feature={feature}, target_date={target_date}")
            
            # Get predictions and plot data
            features_to_predict = [feature] if feature else self.features
            predictions = {}
            
            print(f"Making predictions for features: {features_to_predict}")
            for feat in features_to_predict:
                try:
                    if feat not in self.models.get(symbol, {}):
                        print(f"No model found for {symbol}-{feat}")
                        continue
                        
                    lstm_pred = self.predict_multiple_days(symbol, feat)
                    rf_pred = self.predict_multiple_days(symbol, feat, model_type='rf')
                    
                    if lstm_pred is not None or rf_pred is not None:
                        predictions[feat] = {
                            'lstm': lstm_pred.tolist() if lstm_pred is not None else None,
                            'rf': rf_pred.tolist() if rf_pred is not None else None
                        }
                except Exception as pred_err:
                    print(f"Error predicting {symbol}-{feat}: {str(pred_err)}")
                    continue
            
            if not predictions:
                error_msg = f"Prediction failed for {symbol}. Models may still be training or there might be issues with the data."
                print(error_msg)
                return json.dumps({'error': error_msg})
                
            try:
                plot_data = self.get_plot_data(symbol, feature, target_date)
                if not plot_data or not plot_data.get('features'):
                    print(f"No plot data available for {symbol}")
                    plot_data = {'features': {}, 'dates': []}
            except Exception as plot_err:
                print(f"Error getting plot data: {str(plot_err)}")
                plot_data = {'features': {}, 'dates': []}
            
            # Determine which day to report in message
            day_idx = 0
            if target_date:
                date_obj = self._parse_date(target_date)
                today = datetime.now()
                days_diff = (date_obj.date() - today.date()).days if date_obj else 0
                if 0 < days_diff <= 3:
                    day_idx = days_diff - 1
            
            # Build prediction message
            message = f"{symbol} prediction results:\n"
            
            if not predictions:
                message += "\nNo predictions available at this time. The models might still be training."
            else:
                for feat, pred in predictions.items():
                    lstm_val = pred['lstm'][day_idx] if pred['lstm'] else None
                    rf_val = pred['rf'][day_idx] if pred['rf'] else None
                    
                    if lstm_val is not None or rf_val is not None:
                        message += f"\n{feat}:\n"
                        if lstm_val is not None:
                            message += f"LSTM model: ${lstm_val:.2f}\n"
                        if rf_val is not None:
                            message += f"Random Forest model: ${rf_val:.2f}\n"
            
            print(f"Successfully generated prediction response for {symbol}")
            return json.dumps({
                'symbol': symbol,
                'feature': feature,
                'target_date': target_date,
                'predictions': predictions,
                'message': message,
                'plot_data': plot_data
            })
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return json.dumps({'error': error_msg})
    

oracle = OracleServer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        
        # Process the natural language query
        result_json = oracle.process_request(query)
        result = json.loads(result_json)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_data', methods=['POST'])
def update_data():
    """API endpoint to update data and models"""
    try:
        # Get specific symbol if provided
        data = request.json
        symbol = data.get('symbol') if data else None
        
        # Asynchronously execute update
        success, message = oracle.async_update_data(symbol)
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/update_status', methods=['GET'])
def update_status():
    """API endpoint to get update status"""
    status = oracle.get_update_status()
    return jsonify(status)

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """API endpoint to get available symbols"""
    symbols = oracle.get_available_symbols()
    return jsonify({
        'symbols': symbols
    })

if __name__ == "__main__":
    oracle.load_resources()
    app.run(host='localhost', port=5001, debug=True)