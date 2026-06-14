"""
update.py — Atualização diária ITUB4
Carrega modelos pré-treinados e gera previsão D+1.
23 features — fiel ao TCC.
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ──────────────────────────────────────────
# 1. FEATURES — igual ao TCC
# ──────────────────────────────────────────
FEATURES = [
    'ibov_return', 'dolar_return', 'vix',
    'bb_pct', 'bb_width', 'macd', 'macd_hist',
    'rsi_14', 'atr_14',
    'volatility_5', 'volatility_10', 'volatility_20',
    'volume', 'volume_relative', 'volume_sma_20',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    'return_lag_5', 'return_lag_10', 'return_lag_20',
    'sma_200', 'hurst_252',
]

# ──────────────────────────────────────────
# 2. DOWNLOAD DE DADOS
# ──────────────────────────────────────────
DATA_INICIO = "2010-01-01"
DATA_FIM    = datetime.today().strftime("%Y-%m-%d")

print("Baixando dados...")

def baixar(ticker):
    df = yf.download(ticker, start=DATA_INICIO, end=DATA_FIM,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

itub4_raw = baixar("ITUB4.SA")
ibov_raw  = baixar("^BVSP")
dolar_raw = baixar("USDBRL=X")
vix_raw   = baixar("^VIX")

ibov_close  = ibov_raw["Close"].squeeze()
dolar_close = dolar_raw["Close"].squeeze()
vix_close   = vix_raw["Close"].squeeze()

itub4 = itub4_raw[["Close", "Volume"]].copy().dropna()
itub4["Retorno"] = itub4["Close"].pct_change()

print(f"ITUB4: {itub4.shape[0]} pregões — {itub4.index[0].date()} → {itub4.index[-1].date()}")

# ──────────────────────────────────────────
# 3. FEATURE ENGINEERING — nomes iguais ao TCC
# ──────────────────────────────────────────
df = itub4.copy()

# Lags de retorno
for lag in [1, 2, 3, 5, 10, 20]:
    df[f"return_lag_{lag}"] = df["Retorno"].shift(lag)

# Médias móveis
df["sma_200"] = df["Close"].rolling(200).mean().shift(1)

# Volatilidade rolling
for w in [5, 10, 20]:
    df[f"volatility_{w}"] = df["Retorno"].rolling(w).std().shift(1)

# RSI 14
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
df["rsi_14"] = (100 - (100 / (1 + rs))).shift(1)

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
macd_raw        = ema12 - ema26
macd_signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
df["macd"]      = macd_raw.shift(1)
df["macd_hist"] = (macd_raw - macd_signal_raw).shift(1)

# Bollinger Bands
sma20        = df["Close"].rolling(20).mean()
std20        = df["Close"].rolling(20).std()
bb_upper_raw = sma20 + 2 * std20
bb_lower_raw = sma20 - 2 * std20
bb_width_raw = bb_upper_raw - bb_lower_raw
df["bb_width"] = bb_width_raw.shift(1)
df["bb_pct"]   = (df["Close"].shift(1) - bb_lower_raw.shift(1)) / bb_width_raw.shift(1)

# ATR 14
high_low     = df["Close"].rolling(2).max() - df["Close"].rolling(2).min()
df["atr_14"] = high_low.rolling(14).mean().shift(1)

# Volume
df["volume"]          = df["Volume"].shift(1)
df["volume_relative"] = (df["Volume"] / df["Volume"].rolling(20).mean()).shift(1)
df["volume_sma_20"]   = df["Volume"].rolling(20).mean().shift(1)

# Contexto de mercado
df["ibov_return"]  = ibov_close.pct_change().reindex(df.index).shift(1)
df["dolar_return"] = dolar_close.pct_change().reindex(df.index).shift(1)
df["vix"]          = vix_close.reindex(df.index).shift(1)

# Hurst rolling 252 dias
def hurst_exp(series, min_lag=2, max_lag=100):
    ts = np.array(series)
    lags = range(min_lag, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if all(t == 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(np.clip(poly[0], 0.0, 1.0))

print("Calculando Hurst (pode demorar ~60s)...")
hurst_values = []
win = 252
for i in range(len(df)):
    if i < win:
        hurst_values.append(np.nan)
    else:
        h = hurst_exp(df["Close"].iloc[i - win : i].values)
        hurst_values.append(h)
df["hurst_252"] = pd.Series(hurst_values, index=df.index).shift(1)

# ──────────────────────────────────────────
# 4. DATASET FINAL
# ──────────────────────────────────────────
df_feat = df[FEATURES + ["Retorno", "Close"]].dropna().copy()
print(f"Dataset: {df_feat.shape[0]} linhas × {len(FEATURES)} features")

X_all = df_feat[FEATURES].values
y_all = df_feat["Retorno"].values
h_all = df_feat["hurst_252"].values

# ──────────────────────────────────────────
# 5. CARREGA MODELOS E SCALERS
# ──────────────────────────────────────────
print("Carregando modelos...")

scaler_X    = joblib.load("scaler_X.pkl")
scaler_y    = joblib.load("scaler_y.pkl")
best_thresh = joblib.load("xgb_threshold.pkl")
escala_xgb  = joblib.load("escala_xgb.pkl")

models_xgb = []
for i in range(10):
    m = xgb.Booster()
    m.load_model(f"model_xgb_{i}.json")
    models_xgb.append(m)

import tensorflow as tf
model_lstm = tf.keras.models.load_model("model_lstm.keras")

print(f"Modelos carregados — scaler espera {scaler_X.n_features_in_} features")

# ──────────────────────────────────────────
# 6. TRANSFORMA FEATURES
# ──────────────────────────────────────────
X_all_sc = scaler_X.transform(X_all)

WINDOW = 10

def make_seq(X, window):
    return np.array([X[i-window:i] for i in range(window, len(X))])

X_all_lstm = make_seq(X_all_sc, WINDOW)

# ──────────────────────────────────────────
# 7. PREVISÕES HISTÓRICAS
# ──────────────────────────────────────────
dall         = xgb.DMatrix(X_all_sc)
prob_all_xgb = np.mean([m.predict(dall) for m in models_xgb], axis=0)
pred_xgb_all = (prob_all_xgb - 0.5) * escala_xgb

pred_lstm_all = scaler_y.inverse_transform(
    model_lstm.predict(X_all_lstm, verbose=0)
).ravel()

# Alinha dimensões
n_lstm   = len(pred_lstm_all)
xgb_al   = pred_xgb_all[-n_lstm:]
y_al     = y_all[-n_lstm:]
h_al     = h_all[-n_lstm:]
dates_al = df_feat.index[-n_lstm:]
price_al = df_feat["Close"].values[-n_lstm:]

def peso_hibrido(h):
    return h if h >= 0.5 else 0.1 * h

pesos           = np.array([peso_hibrido(h) for h in h_al])
pred_hybrid_all = (1 - pesos) * xgb_al + pesos * pred_lstm_all

dir_acc = np.mean(np.sign(pred_hybrid_all) == np.sign(y_al))
print(f"Dir. Accuracy histórica: {dir_acc:.2%}")

# ──────────────────────────────────────────
# 8. PREVISÃO D+1
# ──────────────────────────────────────────
x_last  = X_all_sc[[-1]]
h_last  = float(h_all[-1])
p_last  = peso_hibrido(h_last)

prob_last      = float(np.mean([m.predict(xgb.DMatrix(x_last))[0] for m in models_xgb]))
pred_xgb_last  = (prob_last - 0.5) * escala_xgb

x_last_seq     = X_all_sc[-WINDOW:].reshape(1, WINDOW, X_all_sc.shape[1])
pred_lstm_last = float(scaler_y.inverse_transform(
    model_lstm.predict(x_last_seq, verbose=0)
).ravel()[0])

pred_last = (1 - p_last) * pred_xgb_last + p_last * pred_lstm_last

print(f"\n📊 Previsão D+1:")
print(f"   XGBoost prob Alta : {prob_last:.3f} (thr={best_thresh:.2f})")
print(f"   LSTM pred         : {pred_lstm_last*100:+.4f}%")
print(f"   Híbrido pred      : {pred_last*100:+.4f}%")
print(f"   Hurst atual       : {h_last:.4f}")
print(f"   Direção prevista  : {'↑ Alta' if pred_last >= 0 else '↓ Queda'}")

# ──────────────────────────────────────────
# 9. MONTA data.json
# ──────────────────────────────────────────
n_hist    = 252
idx_start = max(0, n_lstm - n_hist)

records = []
for k in range(idx_start, n_lstm):
    records.append({
        "date" : dates_al[k].strftime("%Y-%m-%d"),
        "price": round(float(price_al[k]), 2),
        "real" : round(float(y_al[k]), 6),
        "pred" : round(float(pred_hybrid_all[k]), 6),
        "hurst": round(float(h_al[k]), 4),
    })

from pandas.tseries.offsets import BDay
proximo_dia = (df_feat.index[-1] + BDay(1)).strftime("%Y-%m-%d")

records.append({
    "date" : proximo_dia,
    "price": round(float(df_feat["Close"].iloc[-1]), 2),
    "real" : None,
    "pred" : round(float(pred_last), 6),
    "hurst": round(h_last, 4),
})

with open("data.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)

anterior = records[-2]
latest   = records[-1]
real_str = f"{anterior['real']*100:+.2f}%" if anterior["real"] is not None else "—"

print(f"\n✓ data.json salvo — {len(records)} registros")
print(f"  Último pregão   : {anterior['date']} | Preço: R$ {anterior['price']:.2f} | Real: {real_str}")
print(f"  Previsão D+1    : {latest['date']} | Pred: {latest['pred']*100:+.4f}%")
print(f"  Hurst atual     : {latest['hurst']:.4f}")
print(f"  Direção prevista: {'↑ Alta' if latest['pred'] >= 0 else '↓ Queda'}")
