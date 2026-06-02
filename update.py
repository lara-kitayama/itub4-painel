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

TICKER      = 'ITUB4.SA'
LOOKBACK    = 252        # janela para cálculo do Hurst
DATA_FILE   = 'data.json'
MAX_HISTORY = 90         # máximo de dias no histórico exibido

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
    """Calcula o expoente de Hurst via R/S."""
    lags   = range(min_lag, max_lag)
    tau    = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    poly   = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def compute_features(df):
    """Calcula todas as features a partir do OHLCV do yfinance."""
    d = df.copy()

    # Retorno
    d['Retorno'] = d['Close'].pct_change()

    # Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f'Retorno Lag {lag}'] = d['Retorno'].shift(lag)

    # Médias móveis
    d['MMS 200'] = d['Close'].rolling(200).mean()
    d['MME 5']   = d['Close'].ewm(span=5).mean()
    d['MME 20']  = d['Close'].ewm(span=20).mean()

    # Volatilidade rolling
    for w in [5, 10, 20]:
        d[f'Volatilidade {w}'] = d['Retorno'].rolling(w).std()

    # Bollinger Bands
    sma20  = d['Close'].rolling(20).mean()
    std20  = d['Close'].rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    d['Bollinger %']      = (d['Close'] - lower) / (upper - lower)
    d['Bollinger Largura'] = (upper - lower) / sma20

    # MACD
    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['MACD']           = ema12 - ema26
    d['MACD Histograma'] = d['MACD'] - d['MACD'].ewm(span=9).mean()

    # RSI 14
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    d['RSI 14'] = 100 - (100 / (1 + rs))

    # ATR 14
    hl  = d['High'] - d['Low']
    hcp = (d['High'] - d['Close'].shift()).abs()
    lcp = (d['Low']  - d['Close'].shift()).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d['ATR 14'] = tr.rolling(14).mean()

    # Volume
    d['Volume']          = d['Volume']
    d['Volume MMS 20']   = d['Volume'].rolling(20).mean()
    d['Volume Relativo'] = d['Volume'] / d['Volume MMS 20']

    # Hurst 252
    d['Hurst 252'] = d['Retorno'].rolling(LOOKBACK).apply(
        lambda x: hurst_exponent(x.dropna()) if len(x.dropna()) >= 20 else np.nan,
        raw=False
    )

    return d

def fetch_external(start, end):
    """Busca VIX, Ibovespa e Dólar."""
    ibov  = yf.download('^BVSP',  start=start, end=end, progress=False)['Close'].pct_change()
    vix   = yf.download('^VIX',   start=start, end=end, progress=False)['Close']
    dolar = yf.download('BRL=X',  start=start, end=end, progress=False)['Close'].pct_change()
    return ibov.rename('Retorno Ibovespa'), vix.rename('VIX'), dolar.rename('Retorno Dólar')

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
    latest = df.iloc[-1]

    # =========================
    # FEATURES DA ÚLTIMA OBSERVAÇÃO
    # =========================

    X_raw  = latest[FEATURES].values.reshape(1, -1)
    X_sc   = scaler_X.transform(X_raw)

    # XGBoost
    pred_xgb_sc  = model_xgb.predict(X_sc)
    pred_xgb     = scaler_y.inverse_transform(pred_xgb_sc.reshape(-1, 1)).ravel()[0]

    # LSTM
    X_lstm       = X_sc.reshape(1, 1, X_sc.shape[1])
    pred_lstm_sc = model_lstm.predict(X_lstm, verbose=0).ravel()
    pred_lstm    = scaler_y.inverse_transform(pred_lstm_sc.reshape(-1, 1)).ravel()[0]

    # Híbrido
    H            = latest['Hurst 252']
    pred_hybrid  = (1 - H) * pred_xgb + H * pred_lstm

    real_return  = latest['Retorno']
    price        = latest['Close']
    date_str     = latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)[:10]

    print(f'Data: {date_str}')
    print(f'Preço: R$ {price:.2f}')
    print(f'Retorno real: {real_return:.4%}')
    print(f'Previsão híbrida D+1: {pred_hybrid:.4%}')
    print(f'Hurst: {H:.3f}')

    # =========================
    # ATUALIZA data.json
    # =========================

    # Lê histórico existente
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []

    # Evita duplicata
    if not any(d['date'] == date_str for d in history):
        history.append({
            'date' : date_str,
            'price': round(float(price), 2),
            'real' : round(float(real_return), 6),
            'pred' : round(float(pred_hybrid), 6),
            'hurst': round(float(H), 4)
        })

    # Mantém só os últimos MAX_HISTORY dias
    history = sorted(history, key=lambda x: x['date'])[-MAX_HISTORY:]

    with open(DATA_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'data.json atualizado com {len(history)} registros.')

if __name__ == '__main__':
    run()
