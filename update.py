"""
update.py — Atualização diária ITUB4
Modelo fiel ao TCC final:
  - XGBoost: binary:logistic (classificação direcional)
  - LSTM: regressão com janela temporal de 10 dias
  - Híbrido: ponderação assimétrica pelo Expoente de Hurst
      p(H) = H        se H >= 0.5
      p(H) = 0.1 * H  se H < 0.5
  - 26 features (25 originais + OBV)
  - Ensemble de 10 runs para XGBoost e LSTM
"""

import json
import warnings
import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

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
# 2. FEATURE ENGINEERING (fiel ao TCC)
# Todas as features são shiftadas em 1 dia para evitar leakage
# ──────────────────────────────────────────
df = itub4.copy()

# Lags de retorno
for lag in [1, 2, 3, 5, 10, 20]:
    df[f"return_lag_{lag}"] = df["Retorno"].shift(lag)

# Médias móveis
for w in [5, 10, 20, 50, 200]:
    df[f"sma_{w}"] = df["Close"].rolling(w).mean().shift(1)
for w in [5, 10, 20]:
    df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean().shift(1)

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
df["macd"]        = macd_raw.shift(1)
df["macd_signal"] = macd_signal_raw.shift(1)
df["macd_hist"]   = (macd_raw - macd_signal_raw).shift(1)

# Bollinger Bands
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
bb_upper_raw = sma20 + 2 * std20
bb_lower_raw = sma20 - 2 * std20
bb_width_raw = bb_upper_raw - bb_lower_raw
df["bb_upper"] = bb_upper_raw.shift(1)
df["bb_lower"] = bb_lower_raw.shift(1)
df["bb_width"] = bb_width_raw.shift(1)
df["bb_pct"]   = (df["Close"].shift(1) - bb_lower_raw.shift(1)) / bb_width_raw.shift(1)

# ATR 14
high_low     = df["Close"].rolling(2).max() - df["Close"].rolling(2).min()
df["atr_14"] = high_low.rolling(14).mean().shift(1)

# OBV
df["obv"] = (np.sign(df["Retorno"]) * df["Volume"]).cumsum().shift(1)

# Volume
df["vol_rel"]    = (df["Volume"] / df["Volume"].rolling(20).mean()).shift(1)
df["vol_sma_20"] = df["Volume"].rolling(20).mean().shift(1)
df["volume"]     = df["Volume"].shift(1)

# Calendário
df["day_of_week"]    = df.index.dayofweek
df["month"]          = df.index.month
df["quarter"]        = df.index.quarter
df["is_month_start"] = df.index.is_month_start.astype(int)
df["is_month_end"]   = df.index.is_month_end.astype(int)

# Contexto de mercado
df["ibov_ret"]  = ibov_close.pct_change().reindex(df.index).shift(1)
df["dolar_ret"] = dolar_close.pct_change().reindex(df.index).shift(1)
df["vix"]       = vix_close.reindex(df.index).shift(1)

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

print("Features calculadas.")

# ──────────────────────────────────────────
# 3. RENOMEIA E SELECIONA 26 FEATURES DO TCC
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
    "volume"       : "Volume",
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
    "obv"          : "OBV",
}
df = df.rename(columns=rename_map)

FEATURES = [
    "Retorno Ibovespa", "VIX", "Retorno Dólar",
    "Bollinger %", "Bollinger Largura", "MACD", "MACD Histograma",
    "RSI 14", "ATR 14",
    "Volatilidade 5", "Volatilidade 10", "Volatilidade 20",
    "Volume", "Volume Relativo", "Volume MMS 20",
    "Retorno Lag 1", "Retorno Lag 2", "Retorno Lag 3",
    "Retorno Lag 5", "Retorno Lag 10", "Retorno Lag 20",
    "MMS 200", "MME 5", "MME 20",
    "Hurst 252", "OBV",
]

df_feat = df[FEATURES + ["Retorno", "Close"]].dropna().copy()
print(f"Dataset final: {df_feat.shape[0]} linhas × {len(FEATURES)} features")

# ──────────────────────────────────────────
# 4. DIVISÃO TEMPORAL 70/15/15
# ──────────────────────────────────────────
n          = len(df_feat)
train_end  = int(n * 0.70)
val_end    = int(n * 0.85)

X_all  = df_feat[FEATURES].values
y_all  = df_feat["Retorno"].values
h_all  = df_feat["Hurst 252"].values

X_train = X_all[:train_end];  y_train = y_all[:train_end]
X_val   = X_all[train_end:val_end]; y_val = y_all[train_end:val_end]
X_test  = X_all[val_end:];    y_test  = y_all[val_end:]
h_test  = h_all[val_end:]

# Scaler X
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_val_sc   = scaler_X.transform(X_val)
X_test_sc  = scaler_X.transform(X_test)
X_all_sc   = scaler_X.transform(X_all)

# Target binário para XGBoost
y_train_clf = (y_train > 0).astype(int)
y_val_clf   = (y_val   > 0).astype(int)
y_test_clf  = (y_test  > 0).astype(int)

# Scaler y para LSTM
scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_sc   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

# ──────────────────────────────────────────
# 5. XGBOOST — CLASSIFICAÇÃO BINÁRIA
#    Ensemble de 10 seeds
# ──────────────────────────────────────────
print("\nTreinando XGBoost (10 runs)...")

scale_pos = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()

params_xgb = {
    "objective"        : "binary:logistic",
    "eval_metric"      : "logloss",
    "max_depth"        : 3,
    "learning_rate"    : 0.05,
    "subsample"        : 0.7,
    "colsample_bytree" : 0.7,
    "reg_alpha"        : 0.5,
    "reg_lambda"       : 2.0,
    "min_child_weight" : 5,
    "scale_pos_weight" : scale_pos,
    "nthread"          : -1,
}

# Threshold calibrado na validação
def calibrar_threshold(prob_val, y_val_clf):
    best_t, best_acc = 0.5, 0.0
    for p in np.arange(0.40, 0.61, 0.01):
        acc = accuracy_score(y_val_clf, (prob_val > p).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, p
    return best_t

prob_val_runs  = []
prob_test_runs = []
prob_all_runs  = []
models_xgb     = []

for seed in range(10):
    params_xgb["seed"] = seed
    dtrain = xgb.DMatrix(X_train_sc, label=y_train_clf)
    dval   = xgb.DMatrix(X_val_sc,   label=y_val_clf)
    dtest  = xgb.DMatrix(X_test_sc)
    dall   = xgb.DMatrix(X_all_sc)

    m = xgb.train(
        params_xgb, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    models_xgb.append(m)
    prob_val_runs.append(m.predict(dval))
    prob_test_runs.append(m.predict(dtest))
    prob_all_runs.append(m.predict(dall))
    print(f"  XGB run {seed+1}/10 — melhor iteração: {m.best_iteration}")

prob_val_xgb  = np.mean(prob_val_runs,  axis=0)
prob_test_xgb = np.mean(prob_test_runs, axis=0)
prob_all_xgb  = np.mean(prob_all_runs,  axis=0)

best_thresh = calibrar_threshold(prob_val_xgb, y_val_clf)
print(f"XGBoost threshold calibrado: {best_thresh:.2f}")

# Previsão contínua centrada em zero para o híbrido
# Escala ajustada para compatibilidade com o LSTM (mesma ordem de grandeza do retorno)
escala_xgb    = float(np.std(y_train)) * 0.1
pred_xgb_test = (prob_test_xgb - 0.5) * escala_xgb
pred_xgb_all  = (prob_all_xgb  - 0.5) * escala_xgb

acc_xgb = accuracy_score(y_test_clf, (prob_test_xgb > best_thresh).astype(int))
print(f"XGBoost acurácia teste: {acc_xgb:.2%}")

# ──────────────────────────────────────────
# 6. LSTM — REGRESSÃO COM JANELA DE 10 DIAS
#    Ensemble de 10 seeds
# ──────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

WINDOW = 10

def make_seq(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Sequências com overlap nas bordas
y_train_sc_full = scaler_y.transform(y_train.reshape(-1,1)).ravel()
y_val_sc_full   = scaler_y.transform(y_val.reshape(-1,1)).ravel()

X_train_lstm, y_train_seq = make_seq(X_train_sc, y_train_sc_full, WINDOW)
X_val_lstm,   y_val_seq   = make_seq(
    np.vstack([X_train_sc[-WINDOW:], X_val_sc]),
    np.concatenate([y_train_sc_full[-WINDOW:], y_val_sc_full]), WINDOW)
X_test_lstm,  _           = make_seq(
    np.vstack([X_val_sc[-WINDOW:], X_test_sc]),
    np.zeros(WINDOW + len(X_test_sc)), WINDOW)

# Sequência completa para previsão histórica
X_all_lstm_full, _ = make_seq(X_all_sc, np.zeros(len(X_all_sc)), WINDOW)

y_train_orig = y_train[WINDOW:]
y_val_orig   = y_val
y_test_orig  = y_test

# Peso para equilibrar quedas no LSTM
n_pos = (y_train_orig > 0).sum()
n_neg = (y_train_orig <= 0).sum()
peso_queda = (n_pos / n_neg) * 1.1
sample_weights = np.where(y_train_orig <= 0, peso_queda, 1.0)

print("\nTreinando LSTM (10 runs)...")

pred_test_runs_lstm = []
pred_all_runs_lstm  = []

for seed in range(10):
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    model_lstm = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(WINDOW, X_train_sc.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model_lstm.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="huber"
    )
    model_lstm.fit(
        X_train_lstm, y_train_seq,
        sample_weight=sample_weights,
        validation_data=(X_val_lstm, y_val_seq),
        epochs=200, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=8, min_lr=1e-6),
        ],
    )
    pred_test_runs_lstm.append(
        scaler_y.inverse_transform(
            model_lstm.predict(X_test_lstm, verbose=0)
        ).ravel()
    )
    pred_all_runs_lstm.append(
        scaler_y.inverse_transform(
            model_lstm.predict(X_all_lstm_full, verbose=0)
        ).ravel()
    )
    print(f"  LSTM run {seed+1}/10 concluído")

pred_lstm_test = np.mean(pred_test_runs_lstm, axis=0)
pred_lstm_all  = np.mean(pred_all_runs_lstm,  axis=0)

dir_acc_lstm = np.mean(np.sign(pred_lstm_test) == np.sign(y_test_orig))
print(f"LSTM acurácia direcional teste: {dir_acc_lstm:.2%}")

# ──────────────────────────────────────────
# 7. HÍBRIDO — PONDERAÇÃO ASSIMÉTRICA
#    p(H) = H        se H >= 0.5
#    p(H) = 0.1 * H  se H < 0.5
# ──────────────────────────────────────────
def peso_hibrido(h):
    return h if h >= 0.5 else 0.1 * h

# Alinha dimensões (LSTM perde WINDOW observações no início)
n_lstm = len(pred_lstm_test)
pred_xgb_al = pred_xgb_test[-n_lstm:]
y_test_al   = y_test_orig[-n_lstm:]
h_test_al   = h_test[-n_lstm:]

pesos = np.array([peso_hibrido(h) for h in h_test_al])
pred_hybrid_test = (1 - pesos) * pred_xgb_al + pesos * pred_lstm_test

dir_acc_hybrid = np.mean(np.sign(pred_hybrid_test) == np.sign(y_test_al))
rmse_hybrid    = np.sqrt(np.mean((pred_hybrid_test - y_test_al) ** 2))
mae_hybrid     = np.mean(np.abs(pred_hybrid_test - y_test_al))

print(f"\n📊 Métricas — conjunto de teste:")
print(f"   XGBoost Dir.Acc  : {acc_xgb:.2%}")
print(f"   LSTM    Dir.Acc  : {dir_acc_lstm:.2%}")
print(f"   Híbrido Dir.Acc  : {dir_acc_hybrid:.2%}")
print(f"   Híbrido RMSE     : {rmse_hybrid:.4f}")
print(f"   Híbrido MAE      : {mae_hybrid:.4f}")

# ──────────────────────────────────────────
# 8. PREVISÕES HISTÓRICAS COMPLETAS
# ──────────────────────────────────────────
n_lstm_all = len(pred_lstm_all)
pred_xgb_all_al  = pred_xgb_all[-n_lstm_all:]
h_all_al         = h_all[-n_lstm_all:]
y_all_al         = y_all[-n_lstm_all:]
dates_all        = df_feat.index[-n_lstm_all:]
prices_all       = df_feat["Close"].values[-n_lstm_all:]

pesos_all       = np.array([peso_hibrido(h) for h in h_all_al])
pred_hybrid_all = (1 - pesos_all) * pred_xgb_all_al + pesos_all * pred_lstm_all

# ──────────────────────────────────────────
# 9. SALVA MODELOS E SCALERS
# ──────────────────────────────────────────
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(best_thresh, "xgb_threshold.pkl")

# Salva os 10 modelos XGBoost
for i, m in enumerate(models_xgb):
    m.save_model(f"model_xgb_{i}.json")

# Salva último modelo LSTM (representativo)
model_lstm.save("model_lstm.keras")
print("Modelos e scalers salvos.")

# ──────────────────────────────────────────
# 10. MONTA data.json
# ──────────────────────────────────────────
n_hist    = 252
idx_start = max(0, n_lstm_all - n_hist)

records = []
for k in range(idx_start, n_lstm_all):
    records.append({
        "date" : dates_all[k].strftime("%Y-%m-%d"),
        "price": round(float(prices_all[k]), 2),
        "real" : round(float(y_all_al[k]), 6),
        "pred" : round(float(pred_hybrid_all[k]), 6),
        "hurst": round(float(h_all_al[k]), 4),
    })

# D+1: previsão para o próximo pregão
from pandas.tseries.offsets import BDay
proximo_dia = (df_feat.index[-1] + BDay(1)).strftime("%Y-%m-%d")

# Usa as features do último dia disponível
x_last    = X_all_sc[[-1]]
h_last    = float(h_all[-1])
p_last    = peso_hibrido(h_last)

prob_last_xgb = np.mean([m.predict(xgb.DMatrix(x_last))[0] for m in models_xgb])
# LSTM: usa a última janela de 10 dias
x_last_seq = X_all_sc[-WINDOW:].reshape(1, WINDOW, X_all_sc.shape[1])
pred_last_lstm = float(scaler_y.inverse_transform(
    model_lstm.predict(x_last_seq, verbose=0)
).ravel()[0])

pred_last_xgb    = (float(prob_last_xgb) - 0.5) * escala_xgb
pred_last_hybrid = (1 - p_last) * pred_last_xgb + p_last * pred_last_lstm

records.append({
    "date" : proximo_dia,
    "price": round(float(df_feat["Close"].iloc[-1]), 2),
    "real" : None,
    "pred" : round(float(pred_last_hybrid), 6),
    "hurst": round(h_last, 4),
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
