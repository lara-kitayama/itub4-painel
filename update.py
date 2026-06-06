"""
update.py — Atualização diária ITUB4
Baixa dados, calcula features, roda modelo híbrido XGBoost + LSTM,
salva data.json para o painel.
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ──────────────────────────────────────────
# 1. DOWNLOAD DE DADOS
# ──────────────────────────────────────────
DATA_INICIO = "2010-01-01"
DATA_FIM    = datetime.today().strftime("%Y-%m-%d")

print("Baixando dados...")

def baixar(ticker):
    df = yf.download(ticker, start=DATA_INICIO, end=DATA_FIM,
                     auto_adjust=True, progress=False)
    # Achata MultiIndex (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

itub4_raw = baixar("ITUB4.SA")
ibov_raw  = baixar("^BVSP")
dolar_raw = baixar("USDBRL=X")
vix_raw   = baixar("^VIX")

# Extrai Series de Close
ibov_close  = ibov_raw["Close"].squeeze()
dolar_close = dolar_raw["Close"].squeeze()
vix_close   = vix_raw["Close"].squeeze()

itub4 = itub4_raw[["Close", "Volume"]].copy()
itub4 = itub4.dropna()
itub4["Retorno"] = itub4["Close"].pct_change()

print(f"ITUB4: {itub4.shape[0]} pregões — {itub4.index[0].date()} → {itub4.index[-1].date()}")

# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────
df = itub4.copy()

# Lags de retorno
for lag in [1, 2, 3, 5, 10, 20]:
    df[f"return_lag_{lag}"] = df["Retorno"].shift(lag)

# Médias móveis simples
for w in [5, 10, 20, 50, 200]:
    df[f"sma_{w}"] = df["Close"].rolling(w).mean()

# Médias móveis exponenciais
for w in [5, 10, 20]:
    df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

# Volatilidade rolling
for w in [5, 10, 20]:
    df[f"volatility_{w}"] = df["Retorno"].rolling(w).std()

# RSI 14
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["macd"]        = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]   = df["macd"] - df["macd_signal"]

# Bollinger Bands 20
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["bb_upper"] = sma20 + 2 * std20
df["bb_lower"] = sma20 - 2 * std20
df["bb_width"] = df["bb_upper"] - df["bb_lower"]
df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / df["bb_width"]

# ATR 14
high_low    = df["Close"].rolling(2).max() - df["Close"].rolling(2).min()
df["atr_14"] = high_low.rolling(14).mean()

# Volume relativo e MMS 20 de volume
df["vol_rel"]    = df["Volume"] / df["Volume"].rolling(20).mean()
df["vol_sma_20"] = df["Volume"].rolling(20).mean()

# Calendário
df["day_of_week"]    = df.index.dayofweek
df["month"]          = df.index.month
df["quarter"]        = df.index.quarter
df["is_month_start"] = df.index.is_month_start.astype(int)
df["is_month_end"]   = df.index.is_month_end.astype(int)

# Dados externos — join por índice (evita rename em Series)
df["ibov_ret"]  = ibov_close.pct_change().reindex(df.index)
df["dolar_ret"] = dolar_close.pct_change().reindex(df.index)
df["vix"]       = vix_close.reindex(df.index)

# Expoente de Hurst (janela 252)
def hurst_exp(series, min_lag=2, max_lag=20):
    ts = np.array(series.dropna())
    if len(ts) < max_lag:
        return 0.5
    lags = range(min_lag, max_lag)
    tau  = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    if all(t == 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)

print("Calculando Hurst (pode demorar ~30s)...")
hurst_values = []
win = 252
for i in range(len(df)):
    if i < win:
        hurst_values.append(np.nan)
    else:
        h = hurst_exp(df["Retorno"].iloc[i - win : i])
        hurst_values.append(h)
df["hurst_252"] = hurst_values

print("Features calculadas.")

# ──────────────────────────────────────────
# 3. RENOMEIA PARA NOMES DO NOTEBOOK (pt-BR)
# ──────────────────────────────────────────
rename_map = {
    "ibov_ret"     : "Retorno Ibovespa",
    "vix"          : "VIX",
    "dolar_ret"    : "Retorno Dólar",
    "bb_pct"       : "Bollinger %",
    "bb_width"     : "Bollinger Largura",
    "macd"         : "MACD",
    "macd_hist"    : "MACD Histograma",
    "rsi_14"       : "RSI 14",
    "atr_14"       : "ATR 14",
    "volatility_5" : "Volatilidade 5",
    "volatility_10": "Volatilidade 10",
    "volatility_20": "Volatilidade 20",
    "Volume"       : "Volume",
    "vol_rel"      : "Volume Relativo",
    "vol_sma_20"   : "Volume MMS 20",
    "return_lag_1" : "Retorno Lag 1",
    "return_lag_2" : "Retorno Lag 2",
    "return_lag_3" : "Retorno Lag 3",
    "return_lag_5" : "Retorno Lag 5",
    "return_lag_10": "Retorno Lag 10",
    "return_lag_20": "Retorno Lag 20",
    "sma_200"      : "MMS 200",
    "ema_5"        : "MME 5",
    "ema_20"       : "MME 20",
    "hurst_252"    : "Hurst 252",
}

df = df.rename(columns=rename_map)

FEATURES_25 = [
    "Retorno Ibovespa", "VIX", "Retorno Dólar",
    "Bollinger %", "Bollinger Largura", "MACD", "MACD Histograma",
    "RSI 14", "ATR 14",
    "Volatilidade 5", "Volatilidade 10", "Volatilidade 20",
    "Volume", "Volume Relativo", "Volume MMS 20",
    "Retorno Lag 1", "Retorno Lag 2", "Retorno Lag 3",
    "Retorno Lag 5", "Retorno Lag 10", "Retorno Lag 20",
    "MMS 200", "MME 5", "MME 20",
    "Hurst 252",
]

df_feat = df[FEATURES_25 + ["Retorno", "Close"]].dropna().copy()
print(f"Dataset final: {df_feat.shape[0]} linhas × {len(FEATURES_25)} features")

# ──────────────────────────────────────────
# 4. CARREGA MODELOS E GERA PREVISÕES
# ──────────────────────────────────────────
print("Carregando scalers e modelos...")

scaler_X  = joblib.load("scaler_X.pkl")
scaler_y  = joblib.load("scaler_y.pkl")
model_xgb = joblib.load("model_xgb.pkl")

import tensorflow as tf

if os.path.exists("model_lstm.h5"):
    model_lstm = tf.keras.models.load_model("model_lstm.h5", compile=False)
elif os.path.exists("model_lstm"):
    model_lstm = tf.keras.models.load_model("model_lstm", compile=False)
else:
    raise FileNotFoundError("Modelo LSTM não encontrado (model_lstm.h5 ou model_lstm/)")

X_all  = df_feat[FEATURES_25].values
X_sc   = scaler_X.transform(X_all)
X_lstm = X_sc.reshape((X_sc.shape[0], 1, X_sc.shape[1]))

pred_xgb_sc  = model_xgb.predict(X_sc)
pred_lstm_sc = model_lstm.predict(X_lstm, verbose=0).ravel()

pred_xgb  = scaler_y.inverse_transform(pred_xgb_sc.reshape(-1, 1)).ravel()
pred_lstm = scaler_y.inverse_transform(pred_lstm_sc.reshape(-1, 1)).ravel()

# Ponderação dinâmica pelo Hurst
hurst       = np.clip(df_feat["Hurst 252"].values, 0, 1)
pred_hybrid = (1 - hurst) * pred_xgb + hurst * pred_lstm

print("Previsões calculadas.")

# ──────────────────────────────────────────
# 5. SALVA data.json
# ──────────────────────────────────────────
n_hist  = 252
df_out  = df_feat.iloc[-n_hist:].copy()
pred_h  = pred_hybrid[-n_hist:]
hurst_h = hurst[-n_hist:]
real_r  = df_out["Retorno"].values

records = []
for i, (idx, row) in enumerate(df_out.iterrows()):
    records.append({
        "date" : idx.strftime("%Y-%m-%d"),
        "price": round(float(row["Close"]), 2),
        "real" : round(float(real_r[i]), 6),
        "pred" : round(float(pred_h[i]), 6),
        "hurst": round(float(hurst_h[i]), 4),
    })

with open("data.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)

latest = records[-1]
print(f"\n✓ data.json salvo — {len(records)} registros")
print(f"  Última data  : {latest['date']}")
print(f"  Preço        : R$ {latest['price']:.2f}")
print(f"  Retorno real : {latest['real']*100:+.2f}%")
print(f"  Previsão D+1 : {latest['pred']*100:+.2f}%")
print(f"  Hurst        : {latest['hurst']:.3f}")
print(f"  Direção      : {'↑ Alta' if latest['pred'] >= 0 else '↓ Queda'}")
