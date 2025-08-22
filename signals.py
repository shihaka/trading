# signals.py
import os
import sys
import json
import math
import pathlib
import logging
import requests
from typing import Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SAMARA_TZ = ZoneInfo("Europe/Samara")

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
TELEGRAM_API = os.getenv("TELEGRAM_API", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@shihaleevka").strip()

HORIZONS = {
    "15m": ("1m", 15, "15 м"),
    "30m": ("1m", 30, "30 м"),
    "1h":  ("1h", 1,  "1 ч"),
    "1d":  ("1h", 24, "1 д"),
    "3d":  ("1h", 72, "3 д"),
    "7d":  ("1h", 168,"7 д"),
}

DECISION_THRESHOLDS = {
    "15m": 0.002,
    "30m": 0.004,
    "1h":  0.006,
    "1d":  0.015,
    "3d":  0.025,
    "7d":  0.040,
}

ENTRY_TRIGGER_PCT = 0.001

MAPE_CONF_THRESH = {
    "15m": 0.02,  "30m": 0.03, "1h": 0.04, "1d": 0.06, "3d": 0.08, "7d": 0.10,
}

# -------- утилиты общие --------
def read_coins_from_file(path: pathlib.Path) -> list[str]:
    if not path.exists():
        print(f"Файл coins.txt не найден: {path}")
        return []
    coins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        coin = line.strip().upper()
        if not coin or coin.startswith("#"):
            continue
        coins.append(coin)
    # убрать дубли, сохранив порядок
    seen, out = set(), []
    for c in coins:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def send_telegram(text: str):
    if not TELEGRAM_API:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception:
        return False

def setup_logger(coin: str):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{coin}.log"
    logger = logging.getLogger(f"signals_{coin}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def find_latest_parquet_for_coin(coin: str) -> pathlib.Path | None:
    candidates = sorted(
        DATA_DIR.glob(f"{coin}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def load_parquet(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["dt_local"] = df["dt_utc"].dt.tz_convert(SAMARA_TZ)
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = ema(up, length)
    loss = ema(down, length)
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower) / ma
    return ma, upper, lower, width

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["max"]
    low = df["min"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def ema_np(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def detect_trend(close: pd.Series):
    ema50 = ema_np(close, 50)
    ema200 = ema_np(close, 200)
    if ema200.isna().all():
        return "none", ema50, ema200
    if len(ema200.dropna()) == 0:
        return "none", ema50, ema200
    if ema50.iloc[-1] > ema200.iloc[-1] * 1.001:
        return "up", ema50, ema200
    elif ema50.iloc[-1] < ema200.iloc[-1] * 0.999:
        return "down", ema50, ema200
    else:
        return "flat", ema50, ema200

def find_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[float|None,float|None]:
    part = df.tail(lookback)
    if part.empty:
        return None, None
    sup = float(part["min"].min())
    res = float(part["max"].max())
    return sup, res

def near_level(price: float, level: float|None, atr_val: float|None, mult: float=0.2) -> bool:
    if level is None or (atr_val is None or np.isnan(atr_val) or atr_val<=0):
        return False
    return abs(price - level) <= mult * atr_val

def rsi_zone(rsi_val: float, long: bool) -> bool:
    return (35 <= rsi_val <= 50) if long else (55 <= rsi_val <= 65)

def candle_pattern_long(row_prev, row_cur) -> bool:
    bull_engulf = (row_cur['close'] > row_cur['open']) and (row_prev['close'] < row_prev['open']) and \
                  (row_cur['close'] >= max(row_prev['open'], row_prev['close'])) and \
                  (row_cur['open']  <= min(row_prev['open'], row_prev['close']))
    high = row_cur['max']; low = row_cur['min']; o = row_cur['open']; c = row_cur['close']
    body = abs(c - o); lower_wick = min(o,c) - low; upper_wick = high - max(o,c)
    hammer = (c > o) and (lower_wick >= 2*body) and (upper_wick <= body)
    return bool(bull_engulf or hammer)

def candle_pattern_short(row_prev, row_cur) -> bool:
    bear_engulf = (row_cur['close'] < row_cur['open']) and (row_prev['close'] > row_prev['open']) and \
                  (row_cur['close'] <= min(row_prev['open'], row_prev['close'])) and \
                  (row_cur['open']  >= max(row_prev['open'], row_prev['close']))
    high = row_cur['max']; low = row_cur['min']; o = row_cur['open']; c = row_cur['close']
    body = abs(c - o); lower_wick = min(o,c) - low; upper_wick = high - max(o,c)
    shooting_star = (c < o) and (upper_wick >= 2*body) and (lower_wick <= body)
    return bool(bear_engulf or shooting_star)

def volume_rising(vol_series: pd.Series, mult: float=1.2) -> bool:
    if len(vol_series) < 20:
        return False
    return vol_series.iloc[-1] >= mult * vol_series.iloc[-20:-1].mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    vol = df["volume"].replace(0, np.nan)
    out["ret1"] = close.pct_change(1)
    out["ret5"] = close.pct_change(5)
    out["ret15"] = close.pct_change(15)
    for w in [5, 12, 26, 50, 100, 200]:
        ma = close.rolling(w).mean()
        out[f"ma{w}"] = ma
        out[f"slope_ma{w}"] = ma.diff()
    for e in [5, 12, 26, 50]:
        out[f"ema{e}"] = ema(close, e)
        out[f"slope_ema{e}"] = out[f"ema{e}"].diff()
    out["rsi14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line; out["macd_sig"] = signal_line; out["macd_hist"] = hist
    bb_ma, bb_u, bb_l, bb_w = bollinger(close, 20, 2.0)
    out["bb_ma20"] = bb_ma; out["bb_width"] = bb_w
    out["atr14"] = atr(df, 14); out["atr_pct"] = (out["atr14"] / close).replace([np.inf, -np.inf], np.nan)
    out["log_vol"] = np.log(vol); out["dlog_vol1"] = out["log_vol"].diff()
    out["lag_close1"] = close.shift(1); out["lag_close5"] = close.shift(5)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def fit_linear_regression(X: np.ndarray, y: np.ndarray):
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    def predict(X_new: np.ndarray):
        Xn = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        return Xn @ beta
    return beta, predict

def horizon_config(tag: str):
    base, steps, human = HORIZONS[tag]
    freq = "1min" if base == "1m" else "1H"
    return base, steps, human, freq

def load_history_and_targets(df: pd.DataFrame, base: str, steps: int) -> tuple[pd.DataFrame, pd.Series]:
    part = df[df["interval"] == base].copy()
    if part.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    part = part.sort_values("dt_utc").set_index("dt_utc")
    feats = build_features(part)
    target = part["open"].shift(-steps)
    aligned = feats.join(target.rename("target"), how="inner").dropna()
    y = aligned["target"].copy()
    X = aligned.drop(columns=["target"]).copy()
    return X, y

def now_local():
    return datetime.now(SAMARA_TZ)

def ensure_model_dirs(coin: str) -> pathlib.Path:
    path = MODELS_DIR / coin
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_metrics(coin: str) -> dict:
    mdir = ensure_model_dirs(coin)
    mfile = mdir / "metrics.json"
    if mfile.exists():
        try:
            return json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_metrics(coin: str, metrics: dict):
    mdir = ensure_model_dirs(coin)
    (mdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

def predictions_history_path(coin: str) -> pathlib.Path:
    mdir = ensure_model_dirs(coin)
    return mdir / "predictions_history.csv"

def append_predictions_history(coin: str, rows: list[dict]):
    path = predictions_history_path(coin)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.drop_duplicates(subset=["horizon", "target_time_utc"], keep="last", inplace=True)
    df.to_csv(path, index=False)

def summarize_errors_by_horizon(hist: pd.DataFrame, tail_n: int = 200) -> dict:
    out: dict[str, dict] = {}
    for hz in HORIZONS.keys():
        sub = hist[(hist["horizon"] == hz) & pd.notna(hist["actual"]) & pd.notna(hist["predicted"])]
        if sub.empty:
            continue
        sub = sub.tail(tail_n).copy()
        ape = (sub["predicted"] - sub["actual"]).abs() / sub["actual"].replace(0, np.nan)
        mape = float(ape.mean())
        bias = float(((sub["predicted"] - sub["actual"]) / sub["actual"].replace(0, np.nan)).mean())
        out[hz] = {"mape": mape, "bias": bias}
    return out

def update_history_with_actuals(coin: str, df_all: pd.DataFrame) -> Tuple[dict, dict]:
    path = predictions_history_path(coin)
    mapes = {}
    if not path.exists():
        return mapes
    hist = pd.read_csv(path, parse_dates=["made_at_local", "target_time_local", "target_time_utc"])
    one_m = df_all[df_all["interval"] == "1m"].set_index("dt_utc").sort_index()
    one_h = df_all[df_all["interval"] == "1h"].set_index("dt_utc").sort_index()
    changed = False
    for i, row in hist.iterrows():
        if pd.notna(row.get("actual")):
            continue
        t_utc = row["target_time_utc"].tz_localize("UTC") if row["target_time_utc"].tzinfo is None else row["target_time_utc"]
        base = HORIZONS[row["horizon"]][0]
        source = one_m if base == "1m" else one_h
        if t_utc in source.index:
            actual_open = float(source.loc[t_utc, "open"])
            hist.at[i, "actual"] = actual_open
            pred = float(row["predicted"])
            ape = abs(actual_open - pred) / max(1e-9, actual_open)
            hist.at[i, "ape"] = ape
            changed = True
    if changed:
        hist.to_csv(path, index=False)
    summaries = summarize_errors_by_horizon(hist, tail_n=200)
    for hz, agg in summaries.items():
        mapes[hz] = agg["mape"]
    return mapes, summaries

def compute_confidence(hz: str, mape: float | None) -> float:
    if mape is None:
        return 50.0
    thr = MAPE_CONF_THRESH.get(hz, 0.05)
    conf = 100.0 * max(0.0, 1.0 - min(1.0, mape / (thr * 2)))
    return float(round(conf, 2))

def strength_score(expected_ret: float, hz: str, conf: float) -> float:
    thr = DECISION_THRESHOLDS.get(hz, 0.01)
    r_comp = min(1.5, abs(expected_ret) / (thr + 1e-9))
    return float(round(max(0.0, min(99.9, 100.0 * (0.5 * r_comp + 0.5 * (conf / 100.0)))), 2))

def decide(expected_ret: float, hz: str) -> str:
    thr = DECISION_THRESHOLDS.get(hz, 0.01)
    if expected_ret > thr: return "BUY"
    if expected_ret < -thr: return "SELL"
    return "HOLD"

def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:,.2f}".replace(",", " ")
    elif x >= 1:
        return f"{x:,.4f}".replace(",", " ")
    else:
        return f"{x:,.6f}".replace(",", " ")

def process_coin(coin: str):
    coin = coin.strip().upper()
    logger = setup_logger(coin)
    logger.info(f"=== START signals for {coin} ===")

    parquet_path = find_latest_parquet_for_coin(coin)
    if not parquet_path or not parquet_path.exists():
        msg = f"Файл parquet для {coin} не найден в {DATA_DIR}"
        print(msg); logger.error(msg); return

    df = load_parquet(parquet_path)
    one_m = df[df["interval"] == "1m"].sort_values("dt_utc").copy()
    if len(one_m) < 210:
        logger.info("Недостаточно минутных данных для чек-листа (нужно ~200+).")
    if df.empty:
        msg = "Паркет-файл пуст."
        print(msg); logger.error(msg); return

    now = datetime.now(SAMARA_TZ)
    last_1m = df[df["interval"] == "1m"]
    last_1h = df[df["interval"] == "1h"]
    if not last_1m.empty:
        cur_row = last_1m.sort_values("dt_utc").iloc[-1]
        atr_series = atr(last_1m.sort_values("dt_utc"), 14)
        cur_atr = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
    else:
        cur_row = last_1h.sort_values("dt_utc").iloc[-1]
        atr_series = atr(last_1h.sort_values("dt_utc"), 14)
        cur_atr = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")

    current_price = float(cur_row["close"])
    now_str = now.strftime("%H:%M:%S")

    signal = None
    entry_price = None
    stop_loss = None
    take_profit = None
    valid_from = now
    valid_to = now + timedelta(minutes=15)

    if len(one_m) >= 210:
        last2 = one_m.tail(2).copy()
        row_prev = last2.iloc[-2]
        row_cur  = last2.iloc[-1]

        close_series = one_m["close"]
        rsi_series = rsi(close_series, 14)
        rsi_last = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        trend, ema50s, ema200s = detect_trend(close_series)
        atr_series_full = atr(one_m, 14)
        atr_last = float(atr_series_full.iloc[-1]) if not atr_series_full.empty else float("nan")
        sup, res = find_levels(one_m, lookback=60)

        near_sup = near_level(current_price, sup, atr_last, mult=0.2)
        near_res = near_level(current_price, res, atr_last, mult=0.2)
        vol_ok = volume_rising(one_m["volume"], mult=1.2)

        long_ok  = (trend == "up")   and (near_sup or near_level(current_price, ema50s.iloc[-1], atr_last, 0.2)) \
                   and rsi_zone(rsi_last, long=True)  and candle_pattern_long(row_prev, row_cur)  and vol_ok
        short_ok = (trend == "down") and (near_res or near_level(current_price, ema50s.iloc[-1], atr_last, 0.2)) \
                   and rsi_zone(rsi_last, long=False) and candle_pattern_short(row_prev, row_cur) and vol_ok

        expected_ret_quick = (atr_last / current_price) if (not math.isnan(atr_last) and atr_last>0) else 0.0

        if long_ok and expected_ret_quick >= ENTRY_TRIGGER_PCT:
            signal = "BUY"
        elif short_ok and expected_ret_quick >= ENTRY_TRIGGER_PCT:
            signal = "SELL"

        if signal:
            if not math.isnan(atr_last) and atr_last>0 and not ema50s.isna().iloc[-1]:
                if signal == "BUY":
                    entry_price = max(0.0, min(current_price, ema50s.iloc[-1] - 0.15*atr_last))
                else:
                    entry_price = max(current_price, ema50s.iloc[-1] + 0.15*atr_last)
            else:
                entry_price = current_price

            min_pct, max_pct = 0.015, 0.02
            if signal == "BUY":
                raw_sl = (sup if sup else entry_price - (atr_last if not math.isnan(atr_last) else 0.02*entry_price)) - 0.3*(atr_last if not math.isnan(atr_last) else 0)
                sl_dist = max(min_pct*entry_price, min(max_pct*entry_price, entry_price - raw_sl))
                stop_loss = round(entry_price - sl_dist, 8)
            else:
                raw_sl = (res if res else entry_price + (atr_last if not math.isnan(atr_last) else 0.02*entry_price)) + 0.3*(atr_last if not math.isnan(atr_last) else 0)
                sl_dist = max(min_pct*entry_price, min(max_pct*entry_price, raw_sl - entry_price))
                stop_loss = round(entry_price + sl_dist, 8)

            tp = entry_price + (2.5*(entry_price - stop_loss)) if signal=="BUY" else entry_price - (2.5*(stop_loss - entry_price))
            take_profit = round(tp, 8)

            logger.info(f"CHECKLIST OK | signal={signal} | entry={entry_price:.8f} | sl={stop_loss:.8f} | tp={take_profit:.8f} | "
                        f"trend={trend} rsi={rsi_last:.2f} lvl_sup={sup} lvl_res={res} vol_ok={vol_ok} atr={atr_last}")
        else:
            msg = (f"Нет условий для входа (1m чек-лист). "
                f"trend={trend}, rsi={rsi_last:.2f}, near_sup={near_sup}, "
                f"near_res={near_res}, vol_ok={vol_ok}")
            print(msg)
            logger.info(f"CHECKLIST FAIL | trend={trend} rsi={rsi_last:.2f} "
                        f"near_sup={near_sup} near_res={near_res} vol_ok={vol_ok}")

    # --- обновление истории прогнозов ---
    metrics = load_metrics(coin)
    upd = update_history_with_actuals(coin, df)
    if isinstance(upd, tuple):
        mape_updates, summaries = upd
    else:
        mape_updates, summaries = upd, {}
    if "mape" not in metrics: metrics["mape"] = {}
    if "bias" not in metrics: metrics["bias"] = {}
    for k, v in mape_updates.items():
        if v is not None and not math.isnan(v):
            metrics["mape"][k] = float(v)
    for hz, agg in summaries.items():
        b = agg.get("bias")
        if b is not None and not math.isnan(b):
            metrics["bias"][hz] = float(b)
    save_metrics(coin, metrics)

    # --- прогнозы по горизонтам ---
    outputs = []
    hist_rows = []
    if not df.empty:
        # актуальные данные для ATR на лимит-вход
        one_m_sorted = df[df["interval"] == "1m"].sort_values("dt_utc").copy()
        if not one_m_sorted.empty:
            atr_series_all = atr(one_m_sorted, 14)
            cur_atr = float(atr_series_all.iloc[-1]) if not atr_series_all.empty else float("nan")
        else:
            cur_atr = float("nan")

    for hz_tag in HORIZONS.keys():
        base, steps, human, freq = horizon_config(hz_tag)
        X, y = load_history_and_targets(df, base, steps)

        if len(X) < 200:
            logger.info(f"{hz_tag}: мало данных ({len(X)}), используем эвристику")
            part = df[df["interval"] == base].sort_values("dt_utc").set_index("dt_utc")
            if part.empty:
                continue
            close = part["close"]
            ema12 = ema(close, 12)
            slope = (ema12.diff().iloc[-1]) if len(ema12) > 1 else 0.0
            pred = float(close.iloc[-1] + slope * steps)
        else:
            feat_cols = list(X.columns)
            Xn = X.values.astype(float)
            yn = y.values.astype(float)
            beta, predict = fit_linear_regression(Xn, yn)
            part = df[df["interval"] == base].sort_values("dt_utc").set_index("dt_utc")
            feats_now = build_features(part).iloc[[-1]]
            feats_now = feats_now[feat_cols]
            pred = float(predict(feats_now.values.astype(float))[0])

        last_bar_time = part.index[-1]
        target_time_utc = last_bar_time + pd.to_timedelta(steps, unit=("m" if base == "1m" else "h"))
        target_time_local = target_time_utc.tz_convert(SAMARA_TZ)

        mape = metrics.get("mape", {}).get(hz_tag)
        conf = compute_confidence(hz_tag, mape)
        expected_ret = (pred / current_price) - 1.0
        decision = decide(expected_ret, hz_tag)
        strength = strength_score(expected_ret, hz_tag, conf)

        tvx_price = None
        tvx_reason = ""
        entry_trigger = abs(expected_ret) >= ENTRY_TRIGGER_PCT
        if entry_trigger:
            if not math.isnan(cur_atr) and cur_atr > 0:
                if expected_ret > 0:
                    tvx_price = max(0.0, current_price - 0.25 * cur_atr)
                    tvx_reason = "BUY: ожидается рост >1%, лимит ниже рынка ~0.25*ATR"
                else:
                    tvx_price = current_price + 0.25 * cur_atr
                    tvx_reason = "SELL: ожидается падение >1%, лимит выше рынка ~0.25*ATR"
            else:
                tvx_price = current_price
                tvx_reason = "ATR недоступен — вход по рынку"

        outputs.append({
            "hz": hz_tag, "human": human,
            "pred": pred,
            "target_local": target_time_local,
            "decision": decision,
            "strength": strength,
            "conf": conf,
            "ret_pct": expected_ret * 100.0,
            "tvx_price": tvx_price,
            "tvx_trigger": entry_trigger,
            "tvx_reason": tvx_reason
        })

        hist_rec = {
            "coin": coin,
            "horizon": hz_tag,
            "made_at_local": now,
            "current_price": current_price,
            "predicted": pred,
            "target_time_local": target_time_local,
            "target_time_utc": target_time_utc.tz_convert("UTC"),
            "actual": np.nan,
            "ape": np.nan,
        }
        if tvx_price is not None:
            hist_rec["tvx_price"] = tvx_price
        hist_rows.append(hist_rec)

    if hist_rows:
        append_predictions_history(coin, hist_rows)

    print(f"\nМонета: {coin}")
    print(f"Текущая цена: {fmt_price(current_price)} | Время: {now_str} (Europe/Samara)\n")
    for row in outputs:
        ts = row["target_local"].strftime("%d-%m-%Y %H:%M")
        tvx_str = "да" if row["tvx_trigger"] else "нет"
        tvx_price_str = fmt_price(row["tvx_price"]) if row["tvx_price"] is not None else "-"
        note = f" | ТВХ: {tvx_price_str} ({row['tvx_reason']})" if row["tvx_trigger"] else ""
        print(
            f"{row['human']}: "
            f"open ~ {fmt_price(row['pred'])} | "
            f"{ts} | "
            f"{row['decision']} | "
            f"Δ% к тек: {row['ret_pct']:.2f}% | "
            f"ТВХ (>1%?): {tvx_str}{note} | "
            f"сила: {row['strength']:.1f}%"
        )

    metrics = load_metrics(coin)
    mape_strs, bias_strs = [], []
    for hz in HORIZONS.keys():
        m = metrics.get("mape", {}).get(hz)
        b = metrics.get("bias", {}).get(hz)
        if m is not None and not math.isnan(m):
            mape_strs.append(f"{hz}: {m*100:.2f}%")
        if b is not None and not math.isnan(b):
            sign = "+" if b >= 0 else ""
            bias_strs.append(f"{hz}: {sign}{b*100:.2f}%")

    learned_note = ("; ".join(mape_strs)) if mape_strs else "недостаточно фактических данных для оценки ошибки"
    bias_note = ("; ".join(bias_strs)) if bias_strs else "недостаточно данных для оценки смещения"

    print("\nОбучение/качество модели:")
    print(f"- Обновлены фактические значения прошлых прогнозов (если были).")
    print(f"- Текущая средняя ошибка (MAPE): {learned_note}")
    print(f"- Смещение прогноза (bias): {bias_note}")

    try:
        logger.info(f"Current price at start: {current_price:.8f} (local time {now_str})")
        for row in outputs:
            logger.info(
                f"{row['hz']} | pred={row['pred']:.8f} | target_local={row['target_local']} | "
                f"decision={row['decision']} | ret_pct={row['ret_pct']:.3f} | "
                f"TVX={'Y' if row['tvx_trigger'] else 'N'} tvx_price={row['tvx_price']} | "
                f"strength={row['strength']:.2f}% conf={row['conf']:.2f}%"
            )
        logger.info(f"MAPE summary: {learned_note}")
        logger.info(f"Bias summary: {bias_note}")
        logger.info("=== END ===\n")
    except Exception:
        pass

    if signal and (entry_price is not None) and (stop_loss is not None) and (take_profit is not None):
        time_window = f"{valid_from.strftime('%Y-%m-%d %H:%M')} — {valid_to.strftime('%Y-%m-%d %H:%M')} (Europe/Samara)"
        msg = (
            f"<b>Сделка по чек-листу</b>\n"
            f"Монета: <b>{coin}USDT</b>\n"
            f"Действие: <b>{signal}</b>\n"
            f"Цена входа: <b>{fmt_price(entry_price)}</b>\n"
            f"Стоп-лосс: <b>{fmt_price(stop_loss)}</b>\n"
            f"Время действия цены: <b>{time_window}</b>\n"
            f"Тейк-профит: <b>{fmt_price(take_profit)}</b>"
        )
        ok = send_telegram(msg)
        if not ok and TELEGRAM_CHAT_ID.startswith("@"):
            logger.warning("Telegram: не отправлено. Если это личный username, используйте числовой chat_id.")
        print("\nTelegram: отправлено ✅" if ok else "\nTelegram: не отправлено (нет токена/ошибка)")

def main():
    # если монеты передали аргументами → используем их
    if len(sys.argv) > 1:
        coins = [c.strip().upper() for c in sys.argv[1:] if c.strip()]
    else:
        coins_path = pathlib.Path(__file__).parent / "coins.txt"
        coins = read_coins_from_file(coins_path)

    if not coins:
        print("Список монет пуст. Добавьте монеты в coins.txt или передайте аргументами.")
        return

    for coin in coins:
        try:
            logger = setup_logger(coin)
            logger.info(f"=== START signals for {coin} ===")
            process_coin(coin)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[signals] {coin}: ошибка: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
