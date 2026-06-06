"""
update.py — Atualização diária ITUB4
Roda às 8h (Brasília) via GitHub Actions
"""

import json, numpy as np, pandas as pd, yfinance as yf
import joblib, tensorflow as tf
from datetime import datetime, timedelta
import os, warnings
warnings.filterwarnings('ignore')

TICKER      = 'ITUB4.SA'
LOOKBACK    = 252
DATA_FILE   = 'data.json'
MAX_HISTORY = 90

FEATURES = [
    'Retorno Ibovespa', 'VIX', 'Retorno Dólar',
    'Bollinger %', 'Bollinger Largura',
    'MACD', 'MACD Histograma', 'RSI 14', 'ATR 14',
    'Volatilidade 5', 'Volatilidade 10', 'Volatilidade 20',
    'Volume', 'Volume Relativo', 'Volume MMS 20',
    'Retorno Lag 1', 'Retorno Lag 2', 'Retorno Lag 3',
    'Retorno Lag 5', 'Retorno Lag 10', 'Retorno Lag 20',
    'MMS 200', 'MME 5', 'MME 20', 'Hurst 252'
]

def hurst_exponent(series, min_lag=2, max_lag=20):
    lags = range(min_lag, max_lag)
    tau  = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def next_business_day(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')

def compute_features(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change()
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f'Retorno Lag {lag}'] = d['Retorno'].shift(lag)
    d['MMS 200'] = d['Close'].rolling(200).mean()
    d['MME 5']   = d['Close'].ewm(span=5).mean()
    d['MME 20']  = d['Close'].ewm(span=20).mean()
    for w in [5, 10, 20]:
        d[f'Volatilidade {w}'] = d['Retorno'].rolling(w).std()
    sma20 = d['Close'].rolling(20).mean()
    std20 = d['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    d['Bollinger %']       = (d['Close'] - lower) / (upper - lower)
    d['Bollinger Largura'] = (upper - lower) / sma20
    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['MACD']            = ema12 - ema26
    d['MACD Histograma'] = d['MACD'] - d['MACD'].ewm(span=9).mean()
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI 14'] = 100 - (100 / (1 + gain / loss))
    hl  = d['High'] - d['Low']
    hcp = (d['High'] - d['Close'].shift()).abs()
    lcp = (d['Low']  - d['Close'].shift()).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d['ATR 14']          = tr.rolling(14).mean()
    d['Volume MMS 20']   = d['Volume'].rolling(20).mean()
    d['Volume Relativo'] = d['Volume'] / d['Volume MMS 20']
    d['Hurst 252'] = d['Retorno'].rolling(LOOKBACK).apply(
        lambda x: hurst_exponent(x.dropna()) if len(x.dropna()) >= 20 else np.nan, raw=False)
    return d

def run():
    print('Carregando modelos...')
    model_xgb  = joblib.load('model_xgb.pkl')
    model_lstm = tf.keras.models.load_model('model_lstm.keras')
    scaler_X   = joblib.load('scaler_X.pkl')
    scaler_y   = joblib.load('scaler_y.pkl')

    print('Baixando dados...')
    end   = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=700)

    itub = yf.download(TICKER, start=start, end=end, progress=False)
    itub.columns = itub.columns.get_level_values(0)

    ibov  = yf.download('^BVSP', start=start, end=end, progress=False)['Close'].pct_change().rename('Retorno Ibovespa')
    vix   = yf.download('^VIX',  start=start, end=end, progress=False)['Close'].rename('VIX')
    dolar = yf.download('BRL=X', start=start, end=end, progress=False)['Close'].pct_change().rename('Retorno Dólar')

    df = compute_features(itub).join(ibov).join(vix).join(dolar)
    df = df.dropna(subset=FEATURES + ['Retorno'])

    latest   = df.iloc[-1]
    date_str = latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)[:10]
    price    = float(latest['Close'])
    real_ret = float(latest['Retorno'])
    H        = float(latest['Hurst 252'])

    X_raw        = latest[FEATURES].values.reshape(1, -1)
    X_sc         = scaler_X.transform(X_raw)
    pred_xgb_sc  = model_xgb.predict(X_sc)
    pred_xgb     = float(scaler_y.inverse_transform(pred_xgb_sc.reshape(-1,1)).ravel()[0])
    X_lstm       = X_sc.reshape(1, 1, X_sc.shape[1])
    pred_lstm_sc = model_lstm.predict(X_lstm, verbose=0).ravel()
    pred_lstm    = float(scaler_y.inverse_transform(pred_lstm_sc.reshape(-1,1)).ravel()[0])
    pred_hybrid  = (1 - H) * pred_xgb + H * pred_lstm

    next_date  = next_business_day(date_str)
    pred_price = round(price * (1 + pred_hybrid), 2)

    print(f'Data referência  : {date_str}')
    print(f'Fechamento       : R$ {price:.2f}')
    print(f'Retorno real     : {real_ret:.4%}')
    print(f'Previsão D+1     : {pred_hybrid:.4%}')
    print(f'Fechamento prev. : R$ {pred_price:.2f}')
    print(f'Próximo dia útil : {next_date}')
    print(f'Hurst            : {H:.3f}')

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []

    if not any(d['date'] == date_str for d in history):
        history.append({
            'date'      : date_str,
            'price'     : round(price, 2),
            'real'      : round(real_ret, 6),
            'pred'      : round(pred_hybrid, 6),
            'pred_price': pred_price,
            'next_date' : next_date,
            'hurst'     : round(H, 4)
        })

    history = sorted(history, key=lambda x: x['date'])[-MAX_HISTORY:]

    with open(DATA_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'data.json atualizado com {len(history)} registros.')

if __name__ == '__main__':
    run()
