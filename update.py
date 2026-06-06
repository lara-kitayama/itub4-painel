"""
update.py — Atualização diária ITUB4
Modelo fiel ao TCC: X[i] → y[i] (mesmo dia), 25 features com Hurst,
ponderação híbrida XGBoost + LSTM pelo Expoente de Hurst.

CORREÇÕES EM RELAÇÃO AO ORIGINAL:
  1. Hurst: removida raiz quadrada errada (era sqrt(std), correto é std)
  2. real=None tratado no print final
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

itub4 = itub4_raw[["Close", "Volume"]].copy()
itub4 = itub4.dropna()
itub4["Retorno"] = itub4["Close"].pct_change()

print(f"ITUB4: {itub4.shape[0]} pregões — {itub4.index[0].date()} → {itub4.index[-1].date()}")

# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING (igual ao TCC)
# ──────────────────────────────────────────
df = itub4.copy()

for lag in [1, 2, 3, 5, 10, 20]:
    df[f"return_lag_{lag}"] = df["Retorno"].shift(lag)

for w in [5, 10, 20, 50, 200]:
    df[f"sma_{w}"] = df["Close"].rolling(w).mean()

for w in [5, 10, 20]:
    df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

for w in [5, 10, 20]:
    df[f"volatility_{w}"] = df["Retorno"].rolling(w).std()

delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["macd"]        = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]   = df["macd"] - df["macd_signal"]

sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["bb_upper"] = sma20 + 2 * std20
df["bb_lower"] = sma20 - 2 * std20
df["bb_width"] = df["bb_upper"] - df["bb_lower"]
df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / df["bb_width"]

high_low     = df["Close"].rolling(2).max() - df["Close"].rolling(2).min()
df["atr_14"] = high_low.rolling(14).mean()

df["vol_rel"]    = df["Volume"] / df["Volume"].rolling(20).mean()
df["vol_sma_20"] = df["Volume"].rolling(20).mean()

df["day_of_week"]    = df.index.dayofweek
df["month"]          = df.index.month
df["quarter"]        = df.index.quarter
df["is_month_start"] = df.index.is_month_start.astype(int)
df["is_month_end"]   = df.index.is_month_end.astype(int)

df["ibov_ret"]  = ibov_close.pct_change().reindex(df.index)
df["dolar_ret"] = dolar_close.pct_change().reindex(df.index)
df["vix"]       = vix_close.reindex(df.index)

# ──────────────────────────────────────────
# CORREÇÃO 1: Hurst correto (sem sqrt)
# ──────────────────────────────────────────
def hurst_exp(series, min_lag=2, max_lag=20):
    ts = np.array(series.dropna())
    if len(ts) < max_lag:
        return 0.5
    lags = range(min_lag, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if all(t == 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(np.clip(poly[0] * 2.0, 0.0, 1.0))

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
# 3. RENOMEIA COLUNAS (igual ao TCC)
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

# 25 features igual ao TCC
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
# 4. X[i] → y[i] (fiel ao TCC)
# ──────────────────────────────────────────
X_raw     = df_feat[FEATURES_25].values
y_raw     = df_feat["Retorno"].values
hurst_raw = df_feat["Hurst 252"].values

# ──────────────────────────────────────────
# 5. TREINO / TESTE
# ──────────────────────────────────────────
split = int(len(X_raw) * 0.8)
X_train, X_test = X_raw[:split], X_raw[split:]
y_train, y_test = y_raw[:split], y_raw[split:]

scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

n_pos = np.sum(y_train > 0)
n_neg = np.sum(y_train <= 0)
print(f"XGBoost — Alta: {n_pos} dias | Queda: {n_neg} dias no treino")

# ──────────────────────────────────────────
# 6. XGBOOST
# ──────────────────────────────────────────
model_xgb = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model_xgb.fit(
    X_train_sc, y_train_sc,
    eval_set=[(X_test_sc, y_test_sc)],
    verbose=False,
)
print("XGBoost treinado.")

# ──────────────────────────────────────────
# 7. LSTM
# ──────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

X_train_lstm = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
X_test_lstm  = X_test_sc.reshape((X_test_sc.shape[0],  1, X_test_sc.shape[1]))

model_lstm = Sequential([
    LSTM(64, input_shape=(1, X_train_sc.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
model_lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5)

model_lstm.fit(
    X_train_lstm, y_train_sc,
    validation_data=(X_test_lstm, y_test_sc),
    epochs=100,
    batch_size=32,
    callbacks=[es, lr],
    verbose=1,
)
print("LSTM treinado.")

# ──────────────────────────────────────────
# 8. MÉTRICAS
# ──────────────────────────────────────────
pred_xgb_test  = scaler_y.inverse_transform(
    model_xgb.predict(X_test_sc).reshape(-1, 1)).ravel()
pred_lstm_test = scaler_y.inverse_transform(
    model_lstm.predict(X_test_lstm, verbose=0).reshape(-1, 1)).ravel()

hurst_test       = hurst_raw[split:]
pred_hybrid_test = (1 - hurst_test) * pred_xgb_test + hurst_test * pred_lstm_test

rmse    = np.sqrt(np.mean((pred_hybrid_test - y_test) ** 2))
mae     = np.mean(np.abs(pred_hybrid_test - y_test))
dir_acc = np.mean(np.sign(pred_hybrid_test) == np.sign(y_test))

print(f"\n📊 Métricas no conjunto de teste:")
print(f"   RMSE            : {rmse:.4f}")
print(f"   MAE             : {mae:.4f}")
print(f"   Dir. Accuracy   : {dir_acc*100:.2f}%")
print(f"   % previsões Alta      : {np.mean(pred_hybrid_test > 0)*100:.1f}%")
print(f"   % retornos reais Alta : {np.mean(y_test > 0)*100:.1f}%")

# ──────────────────────────────────────────
# 9. PREVISÕES EM TODO O DATASET
# ──────────────────────────────────────────
X_all_sc   = scaler_X.transform(X_raw)
X_all_lstm = X_all_sc.reshape((X_all_sc.shape[0], 1, X_all_sc.shape[1]))

pred_xgb_all  = scaler_y.inverse_transform(
    model_xgb.predict(X_all_sc).reshape(-1, 1)).ravel()
pred_lstm_all = scaler_y.inverse_transform(
    model_lstm.predict(X_all_lstm, verbose=0).reshape(-1, 1)).ravel()

pred_hybrid_all = (1 - hurst_raw) * pred_xgb_all + hurst_raw * pred_lstm_all

# ──────────────────────────────────────────
# 10. SALVA MODELOS
# ──────────────────────────────────────────
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(model_xgb, "model_xgb.pkl")
model_lstm.save("model_lstm.h5")
print("Modelos e scalers salvos.")

# ──────────────────────────────────────────
# 11. MONTA data.json
#
# pred[i] = previsão do modelo para o dia i (com features do dia i)
# real[i] = retorno que ocorreu no dia i
# O painel usa pred[i] como "o que o modelo previu ontem para hoje"
# e pred[último] como previsão para amanhã (D+1)
# ──────────────────────────────────────────
n_hist = 252
idx_start = max(0, len(df_feat) - n_hist)

records = []
for k in range(idx_start, len(df_feat)):
    records.append({
        "date" : df_feat.index[k].strftime("%Y-%m-%d"),
        "price": round(float(df_feat["Close"].iloc[k]), 2),
        "real" : round(float(y_raw[k]), 6),
        "pred" : round(float(pred_hybrid_all[k]), 6),
        "hurst": round(float(hurst_raw[k]), 4),
    })

# Registro D+1: previsão para o próximo pregão
from pandas.tseries.offsets import BDay
proximo_dia = (df_feat.index[-1] + BDay(1)).strftime("%Y-%m-%d")

records.append({
    "date" : proximo_dia,
    "price": round(float(df_feat["Close"].iloc[-1]), 2),
    "real" : None,
    "pred" : round(float(pred_hybrid_all[-1]), 6),
    "hurst": round(float(hurst_raw[-1]), 4),
})

with open("data.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)

latest   = records[-1]
anterior = records[-2]
real_str = f"{anterior['real']*100:+.2f}%" if anterior["real"] is not None else "—"

print(f"\n✓ data.json salvo — {len(records)} registros")
print(f"  Último pregão   : {anterior['date']} | Preço: R$ {anterior['price']:.2f} | Real: {real_str}")
print(f"  Previsão D+1    : {latest['date']} | Pred: {latest['pred']*100:+.2f}%")
print(f"  Hurst atual     : {latest['hurst']:.4f}")
print(f"  Direção prevista: {'↑ Alta' if latest['pred'] >= 0 else '↓ Queda'}")
