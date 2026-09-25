import os
import re
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Initialize Flask App
app = Flask(__name__)

# Constants
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(CACHE_DIR, exist_ok=True)

LOOKBACK_WINDOW = 60
CACHE_EXPIRY_DAYS = 3
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

CURRENCY_SYMBOLS = {
    'USD': '$',
    'INR': '₹',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'CA$',
    'AUD': 'A$',
    'HKD': 'HK$',
    'SGD': 'S$'
}

def sanitize_ticker(ticker):
    """Sanitize and validate input ticker string."""
    if not ticker or not isinstance(ticker, str):
        return ""
    clean = ticker.strip().upper()
    # Allow alphanumeric, dots, hyphens, carets
    clean = re.sub(r'[^A-Z0-9\.\=\-]', '', clean)
    return clean

def get_currency_symbol(currency_code):
    """Return currency symbol or code if unknown."""
    if not currency_code:
        return 'USD'
    return CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code.upper())

def fetch_usd_inr_exchange_rate():
    """Fetch current USD to INR exchange rate with fallback."""
    try:
        fx = yf.Ticker("USDINR=X")
        rate = getattr(fx.fast_info, 'last_price', None)
        if rate is None or np.isnan(rate) or rate <= 0:
            hist = fx.history(period="5d")
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1])
        if rate and not np.isnan(rate) and rate > 0:
            return float(rate)
    except Exception as e:
        print(f"Warning: Failed to fetch live USD/INR rate: {e}")
    return 83.50  # Reliable fallback rate

def fetch_fx_rate(from_curr, to_curr="INR"):
    """Fetch exchange rate from native currency to target currency."""
    if not from_curr or from_curr.upper() == to_curr.upper():
        return 1.0
    
    from_curr = from_curr.upper()
    if from_curr == "USD" and to_curr == "INR":
        return fetch_usd_inr_exchange_rate()
    
    try:
        pair = f"{from_curr}{to_curr}=X"
        fx = yf.Ticker(pair)
        rate = getattr(fx.fast_info, 'last_price', None)
        if rate is None or np.isnan(rate) or rate <= 0:
            hist = fx.history(period="5d")
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1])
        if rate and not np.isnan(rate) and rate > 0:
            return float(rate)
    except Exception as e:
        print(f"Warning: FX rate fetch failed for {pair}: {e}")
    
    # Static fallbacks for major currencies to INR
    fallbacks_to_inr = {
        'USD': 83.50,
        'EUR': 91.20,
        'GBP': 108.50,
        'JPY': 0.56,
        'CAD': 61.50,
        'AUD': 56.20
    }
    if to_curr == "INR" and from_curr in fallbacks_to_inr:
        return fallbacks_to_inr[from_curr]
    
    return 1.0

def calculate_next_business_day(last_date_str):
    """Calculate the estimated next business day string (YYYY-MM-DD)."""
    try:
        last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception:
        last_date = datetime.date.today()
    
    next_date = last_date + datetime.timedelta(days=1)
    # Skip weekends
    while next_date.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        next_date += datetime.timedelta(days=1)
    
    return next_date.strftime("%Y-%m-%d")

def fetch_stock_dataframe(ticker):
    """Download 5 years of daily market data and sanitize columns."""
    df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'. Please check the symbol.")
    
    # Handle multi-index columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize required columns
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Downloaded market data is missing columns: {missing_cols}")
    
    df = df[FEATURES].copy()
    df = df.dropna()
    
    if len(df) < (LOOKBACK_WINDOW + 30):
        raise ValueError(f"Insufficient historical data ({len(df)} days). At least {LOOKBACK_WINDOW + 30} trading days are required.")
    
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators: SMA20, SMA50, EMA20, RSI14, MACD."""
    df_ind = df.copy()
    
    # Simple Moving Averages
    df_ind['SMA_20'] = df_ind['Close'].rolling(window=20).mean()
    df_ind['SMA_50'] = df_ind['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df_ind['EMA_20'] = df_ind['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI 14)
    delta = df_ind['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_ind['RSI_14'] = 100 - (100 / (1 + rs))
    df_ind['RSI_14'] = df_ind['RSI_14'].fillna(50.0) # neutral default for initial days
    
    # MACD (12, 26, 9)
    ema_12 = df_ind['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_ind['Close'].ewm(span=26, adjust=False).mean()
    df_ind['MACD'] = ema_12 - ema_26
    df_ind['MACD_Signal'] = df_ind['MACD'].ewm(span=9, adjust=False).mean()
    df_ind['MACD_Hist'] = df_ind['MACD'] - df_ind['MACD_Signal']
    
    return df_ind

def build_lstm_architecture(input_shape=(60, 5)):
    """Construct 2-layer LSTM model matching project specification."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_sequences(data_features, data_target, lookback=60):
    """Build chronological lookback sequences (X) and target values (y)."""
    X, y = [], []
    for i in range(lookback, len(data_features)):
        X.append(data_features[i-lookback:i])
        y.append(data_target[i])
    return np.array(X), np.array(y)

def get_cache_paths(clean_ticker):
    """Return model, scaler, and metadata file paths."""
    model_path = os.path.join(CACHE_DIR, f"{clean_ticker}_model.keras")
    scaler_path = os.path.join(CACHE_DIR, f"{clean_ticker}_scalers.pkl")
    meta_path = os.path.join(CACHE_DIR, f"{clean_ticker}_metadata.json")
    return model_path, scaler_path, meta_path

def is_cache_valid(clean_ticker):
    """Check if valid cached model exists and is within expiry limit."""
    model_path, scaler_path, meta_path = get_cache_paths(clean_ticker)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        return False, None
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        train_date = datetime.datetime.strptime(meta['training_date'], "%Y-%m-%d").date()
        age_days = (datetime.date.today() - train_date).days
        
        if age_days <= CACHE_EXPIRY_DAYS:
            return True, meta
    except Exception as e:
        print(f"Cache check error for {clean_ticker}: {e}")
    
    return False, None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Invalid request. Please provide a stock ticker.'}), 400
        
        raw_ticker = data['ticker']
        ticker = sanitize_ticker(raw_ticker)
        if not ticker:
            return jsonify({'error': 'Please enter a valid stock ticker (e.g. AAPL, RELIANCE.NS).'}), 400
        
        clean_ticker = re.sub(r'[^A-Za-z0-9_]', '_', ticker)
        model_path, scaler_path, meta_path = get_cache_paths(clean_ticker)
        
        # Download historical data
        try:
            df = fetch_stock_dataframe(ticker)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 404
        except Exception as e:
            return jsonify({'error': f'Failed to fetch market data for {ticker}: {str(e)}'}), 500
        
        # Calculate technical indicators
        df_indicators = calculate_technical_indicators(df)
        
        # Get metadata from yfinance
        yf_ticker = yf.Ticker(ticker)
        fast_info = getattr(yf_ticker, 'fast_info', None)
        info = {}
        try:
            info = yf_ticker.info or {}
        except Exception:
            pass
        
        company_name = info.get('longName') or info.get('shortName') or ticker
        currency_code = getattr(fast_info, 'currency', None) or info.get('currency')
        if not currency_code:
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                currency_code = 'INR'
            else:
                currency_code = 'USD'
        
        exchange = getattr(fast_info, 'exchange', None) or info.get('exchange') or 'Market'
        
        # 52-week High/Low fallbacks
        fifty_two_high = getattr(fast_info, 'year_high', None) or info.get('fiftyTwoWeekHigh')
        fifty_two_low = getattr(fast_info, 'year_low', None) or info.get('fiftyTwoWeekLow')
        if fifty_two_high is None or np.isnan(fifty_two_high):
            fifty_two_high = float(df['High'].tail(252).max())
        if fifty_two_low is None or np.isnan(fifty_two_low):
            fifty_two_low = float(df['Low'].tail(252).min())
            
        latest_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
        latest_date_str = df.index[-1].strftime("%Y-%m-%d")
        latest_volume = int(df['Volume'].iloc[-1])
        next_business_day = calculate_next_business_day(latest_date_str)
        
        # Check Model Cache
        cached_valid, cached_meta = is_cache_valid(clean_ticker)
        model_source = "cached" if cached_valid else "newly_trained"
        
        if cached_valid:
            # Load cached model and scalers
            model = load_model(model_path)
            scalers = joblib.load(scaler_path)
            scaler_features = scalers['features']
            scaler_target = scalers['target']
            metrics = cached_meta['metrics']
            epochs_completed = cached_meta.get('epochs_completed', 20)
            last_trained_date = cached_meta.get('training_date', latest_date_str)
            train_samples = cached_meta.get('training_samples', 0)
            test_samples = cached_meta.get('testing_samples', 0)
        else:
            # Chronological splitting
            raw_feature_data = df[FEATURES].values
            raw_target_data = df[['Close']].values
            
            total_samples = len(df)
            train_end = int(total_samples * 0.70)
            val_end = train_end + int(total_samples * 0.15)
            
            train_features = raw_feature_data[:train_end]
            train_target = raw_target_data[:train_end]
            
            # Fit Scalers ONLY on training data to avoid data leakage
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            
            scaler_features.fit(train_features)
            scaler_target.fit(train_target)
            
            # Scale entire dataset using train-fitted scalers
            scaled_features = scaler_features.transform(raw_feature_data)
            scaled_target = scaler_target.transform(raw_target_data).flatten()
            
            # Create sequences
            X_all, y_all = create_sequences(scaled_features, scaled_target, lookback=LOOKBACK_WINDOW)
            
            # Sequence splits
            seq_total = len(X_all)
            seq_train_end = int(seq_total * 0.70)
            seq_val_end = seq_train_end + int(seq_total * 0.15)
            
            X_train, y_train = X_all[:seq_train_end], y_all[:seq_train_end]
            X_val, y_val = X_all[seq_train_end:seq_val_end], y_all[seq_train_end:seq_val_end]
            X_test, y_test = X_all[seq_val_end:], y_all[seq_val_end:]
            
            train_samples = len(X_train)
            test_samples = len(X_test)
            
            # Build and train LSTM model
            model = build_lstm_architecture(input_shape=(LOOKBACK_WINDOW, len(FEATURES)))
            early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            epochs_completed = len(history.history['loss'])
            
            # Evaluate Model on Unseen Test Data
            y_test_pred_scaled = model.predict(X_test, verbose=0)
            y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler_target.inverse_transform(y_test_pred_scaled).flatten()
            
            mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
            mse = float(mean_squared_error(y_test_inv, y_pred_inv))
            rmse = float(np.sqrt(mse))
            mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / np.where(y_test_inv == 0, 1.0, y_test_inv))) * 100)
            
            metrics = {
                'mae': round(mae, 4),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 2)
            }
            
            last_trained_date = datetime.date.today().strftime("%Y-%m-%d")
            
            # Save Model & Preprocessing Artifacts
            model.save(model_path)
            joblib.dump({'features': scaler_features, 'target': scaler_target}, scaler_path)
            
            metadata = {
                'ticker': ticker,
                'training_date': last_trained_date,
                'latest_training_data_date': latest_date_str,
                'features': FEATURES,
                'lookback': LOOKBACK_WINDOW,
                'metrics': metrics,
                'epochs_completed': epochs_completed,
                'training_samples': train_samples,
                'testing_samples': test_samples
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Predict Next Business Day Closing Price
        # Prepare latest 60 trading days sequence
        last_60_df = df[FEATURES].tail(LOOKBACK_WINDOW).values
        last_60_scaled = scaler_features.transform(last_60_df)
        X_predict = np.expand_dims(last_60_scaled, axis=0) # shape (1, 60, 5)
        
        predicted_scaled = model.predict(X_predict, verbose=0)
        predicted_close_native = float(scaler_target.inverse_transform(predicted_scaled)[0][0])
        
        # Residual error range derived from test RMSE
        rmse_val = metrics.get('rmse', 2.0)
        range_low_native = max(0.0, predicted_close_native - rmse_val)
        range_high_native = predicted_close_native + rmse_val
        
        pred_abs_change = predicted_close_native - latest_close
        pred_pct_change = (pred_abs_change / latest_close) * 100.0 if latest_close > 0 else 0.0
        
        # Currency Rates for INR Presentation Toggle
        fx_to_inr = fetch_fx_rate(currency_code, "INR")
        
        # Chart Data formatting (last 252 trading days ~ 1 year for interactive view)
        chart_df = df_indicators.tail(252).copy()
        chart_dates = [d.strftime("%Y-%m-%d") for d in chart_df.index]
        chart_close = chart_df['Close'].round(2).tolist()
        chart_sma20 = [round(v, 2) if not np.isnan(v) else None for v in chart_df['SMA_20']]
        chart_sma50 = [round(v, 2) if not np.isnan(v) else None for v in chart_df['SMA_50']]
        
        # Extended 5-year data points for 5Y view options
        full_chart_df = df_indicators.copy()
        full_dates = [d.strftime("%Y-%m-%d") for d in full_chart_df.index]
        full_close = full_chart_df['Close'].round(2).tolist()
        full_sma20 = [round(v, 2) if not np.isnan(v) else None for v in full_chart_df['SMA_20']]
        full_sma50 = [round(v, 2) if not np.isnan(v) else None for v in full_chart_df['SMA_50']]

        # Extract latest indicator values
        latest_ind = df_indicators.iloc[-1]
        technical_indicators = {
            'sma_20': round(float(latest_ind['SMA_20']), 2) if not np.isnan(latest_ind['SMA_20']) else None,
            'sma_50': round(float(latest_ind['SMA_50']), 2) if not np.isnan(latest_ind['SMA_50']) else None,
            'ema_20': round(float(latest_ind['EMA_20']), 2) if not np.isnan(latest_ind['EMA_20']) else None,
            'rsi_14': round(float(latest_ind['RSI_14']), 2) if not np.isnan(latest_ind['RSI_14']) else None,
            'macd_line': round(float(latest_ind['MACD']), 2) if not np.isnan(latest_ind['MACD']) else None,
            'macd_signal': round(float(latest_ind['MACD_Signal']), 2) if not np.isnan(latest_ind['MACD_Signal']) else None,
            'macd_hist': round(float(latest_ind['MACD_Hist']), 2) if not np.isnan(latest_ind['MACD_Hist']) else None,
        }
        
        response_payload = {
            'stock_info': {
                'ticker': ticker,
                'company_name': company_name,
                'exchange': exchange,
                'native_currency': currency_code,
                'currency_symbol': get_currency_symbol(currency_code),
                'latest_close': round(latest_close, 2),
                'prev_close': round(prev_close, 2),
                'fifty_two_high': round(fifty_two_high, 2),
                'fifty_two_low': round(fifty_two_low, 2),
                'latest_date': latest_date_str,
                'latest_volume': latest_volume,
                'fx_to_inr': round(fx_to_inr, 4)
            },
            'prediction': {
                'next_business_day': next_business_day,
                'predicted_price': round(predicted_close_native, 2),
                'predicted_change': round(pred_abs_change, 2),
                'predicted_pct_change': round(pred_pct_change, 2),
                'range_low': round(range_low_native, 2),
                'range_high': round(range_high_native, 2),
                'residual_rmse': round(rmse_val, 2),
                'confidence_label': 'Test Residual Margin (±1 RMSE)'
            },
            'model_info': {
                'model_source': model_source,
                'last_trained_date': last_trained_date,
                'lookback_window': LOOKBACK_WINDOW,
                'epochs_completed': epochs_completed,
                'training_samples': train_samples,
                'testing_samples': test_samples,
                'metrics': metrics
            },
            'technical_indicators': technical_indicators,
            'chart_data': {
                'dates': full_dates,
                'close': full_close,
                'sma20': full_sma20,
                'sma50': full_sma50
            }
        }
        
        return jsonify(response_payload)

    except Exception as e:
        print(f"Error processing prediction request: {e}")
        return jsonify({'error': f"An unexpected server error occurred: {str(e)}"}), 500

@app.route('/compare', methods=['POST'])
def compare_stocks():
    """Endpoint for normalized 1-year stock percentage performance comparison."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'tickers' not in data or not isinstance(data['tickers'], list):
            return jsonify({'error': 'Invalid payload. Expecting a list of tickers.'}), 400
        
        tickers = [sanitize_ticker(t) for t in data['tickers'] if t and isinstance(t, str)]
        tickers = [t for t in tickers if t][:3] # Max 3 tickers
        
        if not tickers:
            return jsonify({'error': 'Please provide at least one valid ticker symbol to compare.'}), 400
        
        comparison_results = {}
        shared_dates = None
        
        for t in tickers:
            try:
                df = fetch_stock_dataframe(t)
                # Take last 252 trading days (~1 year)
                df_1y = df.tail(252).copy()
                if df_1y.empty:
                    continue
                
                start_close = df_1y['Close'].iloc[0]
                pct_series = ((df_1y['Close'] / start_close) - 1.0) * 100.0
                
                dates_str = [d.strftime("%Y-%m-%d") for d in df_1y.index]
                if shared_dates is None or len(dates_str) < len(shared_dates):
                    shared_dates = dates_str
                
                comparison_results[t] = {
                    'dates': dates_str,
                    'pct_change': [round(v, 2) for v in pct_series],
                    'latest_pct': round(float(pct_series.iloc[-1]), 2)
                }
            except Exception as e:
                print(f"Comparison error for {t}: {e}")
                continue
        
        if not comparison_results:
            return jsonify({'error': 'Failed to fetch historical comparison data for the requested tickers.'}), 404
            
        return jsonify({
            'dates': shared_dates,
            'tickers_data': comparison_results
        })
        
    except Exception as e:
        return jsonify({'error': f"Failed to compute stock comparison: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting StockVision AI Server at http://127.0.0.1:5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
