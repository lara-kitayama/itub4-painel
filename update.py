"""
update.py — Script de atualização diária do painel ITUB4
Roda automaticamente via GitHub Actions todo dia útil às 19h (Brasília)
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import os

# =========================
# CONFIGURAÇÕES
# =========================

TICKER    = 'ITUB4.SA'
LOOKBACK  = 252
DATA_FILE = 'data.json'
MAX_HISTORY = 90

# =========================
# FEATURES — mesma ordem do treino
# =========================

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

# =========================
# CARREGA MODELOS
# =========================

print('Carregando modelos...')
model_xgb  = joblib.load('model_xgb.pkl')
model_lstm = tf.keras.models.load_model('model_lstm.h5')
scaler_X   = joblib.load('scaler_X.pkl')
scaler_y   = joblib.load('scaler_y.pkl')

# =========================
# FUNÇÕES AUXILIARES
# =========================

def hurst_exponent(series, min_lag=2, max_lag=20):
    lags = range(min_lag, max_lag)
    tau  = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

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

    sma20  = d['Close'].rolling(20).mean()
    std20  = d['Close'].rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    d['Bollinger %']       = (d['Close'] - lower) / (upper - lower)
    d['Bollinger Largura'] = (upper - lower) / sma20

    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['MACD']            = ema12 - ema26
    d['MACD Histograma'] = d['MACD'] - d['MACD'].ewm(span=9).mean()

    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    d['RSI 14'] = 100 - (100 / (1 + rs))

    hl  = d['High'] - d['Low']
    hcp = (d['High'] - d['Close'].shift()).abs()
    lcp = (d['Low']  - d['Close'].shift()).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d['ATR 14'] = tr.rolling(14).mean()

    d['Volume']          = d['Volume']
    d['Volume MMS 20']   = d['Volume'].rolling(20).mean()
    d['Volume Relativo'] = d['Volume'] / d['Volume MMS 20']

    d['Hurst 252'] = d['Retorno'].rolling(LOOKBACK).apply(
        lambda x: hurst_exponent(x.dropna()) if len(x.dropna()) >= 20 else np.nan,
        raw=False
    )
    return d

def fetch_external(start, end):
    ibov  = yf.download('^BVSP', start=start, end=end, progress=False)['Close'].pct_change()
    vix   = yf.download('^VIX',  start=start, end=end, progress=False)['Close']
    dolar = yf.download('BRL=X', start=start, end=end, progress=False)['Close'].pct_change()
    return ibov.rename('Retorno Ibovespa'), vix.rename('VIX'), dolar.rename('Retorno Dólar')

def next_business_day(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')

def predict_hybrid(X_raw, H):
    X_sc = scaler_X.transform(X_raw)

    pred_xgb_sc  = model_xgb.predict(X_sc)
    pred_xgb     = scaler_y.inverse_transform(pred_xgb_sc.reshape(-1, 1)).ravel()[0]

    X_lstm       = X_sc.reshape(1, 1, X_sc.shape[1])
    pred_lstm_sc = model_lstm.predict(X_lstm, verbose=0).ravel()
    pred_lstm    = scaler_y.inverse_transform(pred_lstm_sc.reshape(-1, 1)).ravel()[0]

    pred_hybrid  = (1 - H) * pred_xgb + H * pred_lstm
    return round(float(pred_hybrid), 6)

# =========================
# PIPELINE PRINCIPAL
# =========================

def run():
    print('Baixando dados do ITUB4...')
    end   = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=600)

    itub = yf.download(TICKER, start=start, end=end, progress=False)
    itub.columns = itub.columns.get_level_values(0)

    print('Calculando features...')
    df = compute_features(itub)

    print('Buscando dados externos...')
    ibov, vix, dolar = fetch_external(start, end)
    df = df.join(ibov).join(vix).join(dolar)
    df = df.dropna(subset=FEATURES + ['Retorno'])

    latest   = df.iloc[-1]
    date_str = latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)[:10]

    X_raw        = latest[FEATURES].values.reshape(1, -1)
    H            = float(latest['Hurst 252'])
    pred_d1      = predict_hybrid(X_raw, H)
    price_today  = float(latest['Close'])
    pred_price_d1 = round(price_today * (1 + pred_d1), 2)
    next_date    = next_business_day(date_str)
    real_return  = round(float(latest['Retorno']), 6)

    print(f'Data atual    : {date_str}')
    print(f'Preço atual   : R$ {price_today:.2f}')
    print(f'Retorno real  : {real_return:.4%}')
    print(f'Previsão D+1  : {pred_d1:.4%}')
    print(f'Fechamento previsto D+1: R$ {pred_price_d1:.2f}')
    print(f'Próximo pregão: {next_date}')
    print(f'Hurst         : {H:.3f}')

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []

    # Atualiza real do dia anterior agora que temos o fechamento de hoje
    if len(history) > 0:
        prev = history[-1]
        if prev.get('next_date') == date_str:
            prev['real']       = real_return
            prev['real_price'] = round(price_today, 2)

    if not any(d['date'] == date_str for d in history):
        history.append({
            'date'       : date_str,
            'price'      : round(price_today, 2),
            'real'       : real_return,
            'pred'       : pred_d1,
            'pred_price' : pred_price_d1,
            'next_date'  : next_date,
            'hurst'      : round(H, 4)
        })

    history = sorted(history, key=lambda x: x['date'])[-MAX_HISTORY:]

    with open(DATA_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'data.json atualizado com {len(history)} registros.')

if __name__ == '__main__':
    run()
