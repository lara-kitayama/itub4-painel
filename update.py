"""
update.py — Atualização diária ITUB4
Baixa dados, calcula features, roda modelo híbrido XGBoost + LSTM,
salva data.json para o painel.

CORREÇÕES APLICADAS:
  1. Hurst: removida raiz quadrada errada em tau (era sqrt(std), correto é std)
  2. XGBoost: adicionado scale_pos_weight para balancear classes alta/queda
  3. Alinhamento D+1: pred[i] é a previsão feita com dados de [i], para o retorno de [i+1]
     → no data.json, pred salvo em [i] representa o que o modelo previu PARA [i+1]
     → a coluna "real" em [i+1] é o que de fato aconteceu no dia [i+1]
     → a comparação de acerto é: sign(pred[i]) == sign(real[i+1]) ✓
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
from sklearn.model_selection import train_test_split
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
# 2. FEATURE ENGINEERING
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

high_low    = df["Close"].rolling(2).max() - df["Close"].rolling(2).min()
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
# CORREÇÃO 1: Hurst sem raiz quadrada errada
# ──────────────────────────────────────────
def hurst_exp(series, min_lag=2, max_lag=20):
    """
    Calcula o Expoente de Hurst pela análise R/S simplificada.
    CORREÇÃO: tau usa std puro — não sqrt(std).
    Resultado esperado: entre 0 e 1.
      H > 0.5 → série persistente (tendência)
      H < 0.5 → série anti-persistente (reversão)
      H ≈ 0.5 → random walk
    """
    ts = np.array(series.dropna())
    if len(ts) < max_lag:
        return 0.5
    lags = range(min_lag, max_lag)
    # CORRETO: std puro, sem sqrt
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if all(t == 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    h = float(poly[0] * 2.0)
    # Garante range válido [0, 1]
    return float(np.clip(h, 0.0, 1.0))

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
# 3. RENOMEIA COLUNAS
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
# CORREÇÃO 2: Alinhamento correto D+1
# X[i] → features do dia i
# y[i] → retorno do dia i+1 (o que o modelo deve prever)
# ──────────────────────────────────────────
X_raw = df_feat[FEATURES_25].values[:-1]       # todos menos o último
y_raw = df_feat["Retorno"].values[1:]           # retorno do dia seguinte
close_raw = df_feat["Close"].values[:-1]        # preço do dia i
hurst_raw = df_feat["Hurst 252"].values[:-1]    # hurst do dia i

# ──────────────────────────────────────────
# 4. TREINO / TESTE (sem shuffle — série temporal)
# ──────────────────────────────────────────
split = int(len(X_raw) * 0.8)
X_train, X_test = X_raw[:split], X_raw[split:]
y_train, y_test = y_raw[:split], y_raw[split:]

# Scalers
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ──────────────────────────────────────────
# CORREÇÃO 3: XGBoost balanceado
# scale_pos_weight = n_negativos / n_positivos
# (usado via sample_weight para regressão, que não tem scale_pos_weight nativo)
# ──────────────────────────────────────────
n_pos = np.sum(y_train > 0)
n_neg = np.sum(y_train <= 0)
ratio = n_neg / n_pos if n_pos > 0 else 1.0

# Para regressão, balanceamos via sample_weight:
# dias de queda recebem peso maior para compensar sub-representação
sample_weight = np.where(y_train <= 0, ratio, 1.0)

print(f"Balanceamento XGBoost — Alta: {n_pos} dias | Queda: {n_neg} dias | ratio: {ratio:.2f}")

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
    sample_weight=sample_weight,
    eval_set=[(X_test_sc, y_test_sc)],
    verbose=False,
)
print("XGBoost treinado.")

# ──────────────────────────────────────────
# 5. LSTM
# ──────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

X_train_lstm = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
X_test_lstm  = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))

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
# 6. MÉTRICAS DE AVALIAÇÃO
# ──────────────────────────────────────────
pred_xgb_test_sc  = model_xgb.predict(X_test_sc)
pred_lstm_test_sc = model_lstm.predict(X_test_lstm, verbose=0).ravel()

pred_xgb_test  = scaler_y.inverse_transform(pred_xgb_test_sc.reshape(-1, 1)).ravel()
pred_lstm_test = scaler_y.inverse_transform(pred_lstm_test_sc.reshape(-1, 1)).ravel()

hurst_test = hurst_raw[split:]
pred_hybrid_test = (1 - hurst_test) * pred_xgb_test + hurst_test * pred_lstm_test

rmse = np.sqrt(np.mean((pred_hybrid_test - y_test) ** 2))
mae  = np.mean(np.abs(pred_hybrid_test - y_test))
dir_acc = np.mean(np.sign(pred_hybrid_test) == np.sign(y_test))

print(f"\n📊 Métricas no conjunto de teste:")
print(f"   RMSE         : {rmse:.4f}")
print(f"   MAE          : {mae:.4f}")
print(f"   Dir. Accuracy: {dir_acc*100:.2f}%")
print(f"   % previsões Alta  : {np.mean(pred_hybrid_test > 0)*100:.1f}%")
print(f"   % retornos reais Alta: {np.mean(y_test > 0)*100:.1f}%")

# ──────────────────────────────────────────
# 7. PREVISÕES EM TODO O DATASET
# ──────────────────────────────────────────
X_all_sc   = scaler_X.transform(X_raw)
X_all_lstm = X_all_sc.reshape((X_all_sc.shape[0], 1, X_all_sc.shape[1]))

pred_xgb_sc_all  = model_xgb.predict(X_all_sc)
pred_lstm_sc_all = model_lstm.predict(X_all_lstm, verbose=0).ravel()

pred_xgb_all  = scaler_y.inverse_transform(pred_xgb_sc_all.reshape(-1, 1)).ravel()
pred_lstm_all = scaler_y.inverse_transform(pred_lstm_sc_all.reshape(-1, 1)).ravel()

pred_hybrid_all = (1 - hurst_raw) * pred_xgb_all + hurst_raw * pred_lstm_all

# ──────────────────────────────────────────
# 8. SALVA MODELOS E SCALERS
# ──────────────────────────────────────────
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(model_xgb, "model_xgb.pkl")
model_lstm.save("model_lstm.h5")
print("Modelos e scalers salvos.")

# ──────────────────────────────────────────
# 9. MONTA data.json
#
# LÓGICA CORRETA:
#   records[i].pred  = previsão feita com dados do dia i, PARA o dia i+1
#   records[i].real  = retorno que de fato ocorreu NO DIA i+1
#   records[i].price = preço de fechamento do dia i
#   records[i].date  = data do dia i+1 (o dia ao qual a previsão se refere)
#
# Assim, no painel:
#   D0 → o último registro mostra o retorno real e o preço de hoje
#   D+1 → a previsão do último registro é para amanhã
# ──────────────────────────────────────────
n_hist = 252

# df_feat tem N linhas. X_raw / pred têm N-1 entradas (alinhadas i → i+1).
# Para o data.json, exportamos os últimos n_hist pares alinhados.
df_base   = df_feat.iloc[1:]          # dias i+1 (onde o retorno real ocorreu)
df_source = df_feat.iloc[:-1]         # dias i (onde a feature/previsão foi gerada)

idx_start = max(0, len(pred_hybrid_all) - n_hist)

records = []
for k in range(idx_start, len(pred_hybrid_all)):
    date_alvo  = df_base.index[k]          # data do retorno real (i+1)
    real_ret   = df_base["Retorno"].iloc[k]
    price_hoje = df_base["Close"].iloc[k]   # preço no dia do retorno real
    pred_ret   = pred_hybrid_all[k]
    h_val      = hurst_raw[k]

    records.append({
        "date" : date_alvo.strftime("%Y-%m-%d"),
        "price": round(float(price_hoje), 2),
        "real" : round(float(real_ret), 6),
        "pred" : round(float(pred_ret), 6),
        "hurst": round(float(h_val), 4),
    })

# Adiciona o registro de D+1 (previsão para amanhã, sem retorno real ainda)
ultimo_idx   = df_feat.index[-1]
ultimo_price = df_feat["Close"].iloc[-1]
ultimo_pred  = pred_hybrid_all[-1]   # previsão gerada com os dados de hoje
ultimo_hurst = hurst_raw[-1]

from pandas.tseries.offsets import BDay
proximo_dia = (ultimo_idx + BDay(1)).strftime("%Y-%m-%d")

records.append({
    "date" : proximo_dia,
    "price": round(float(ultimo_price), 2),
    "real" : None,          # ainda não ocorreu
    "pred" : round(float(ultimo_pred), 6),
    "hurst": round(float(ultimo_hurst), 4),
})

with open("data.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)

latest = records[-1]
anterior = records[-2]
print(f"\n✓ data.json salvo — {len(records)} registros")
print(f"  Último pregão   : {anterior['date']} | Preço: R$ {anterior['price']:.2f} | Real: {anterior['real']*100:+.2f}%")
print(f"  Previsão D+1    : {latest['date']} | Pred: {latest['pred']*100:+.2f}%")
print(f"  Hurst atual     : {latest['hurst']:.4f}")
print(f"  Direção prevista: {'↑ Alta' if latest['pred'] >= 0 else '↓ Queda'}")
