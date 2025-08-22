# bybit_kline_collector.py
import os
import sys
import time
import shutil
import pathlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv

BYBIT_BASE_URL = "https://api.bybit.com"
KLINE_ENDPOINT = "/v5/market/kline"  # public v5
SAMARA_TZ = ZoneInfo("Europe/Samara")

# ----------------- УТИЛИТЫ -----------------

def read_coins_from_file(path: pathlib.Path) -> list[str]:
    if not path.exists():
        print(f"Файл с монетами не найден: {path}")
        return []
    coins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        coin = line.strip().upper()
        if not coin or coin.startswith("#"):
            continue
        coins.append(coin)
    # уберём дубликаты, сохраняя порядок
    seen = set(); out = []
    for c in coins:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def sleep_until_next_full_minute():
    now = datetime.now(SAMARA_TZ)
    nxt = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    time_to_sleep = (nxt - now).total_seconds()
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

def local_now_truncated():
    now_local = datetime.now(SAMARA_TZ)
    last_full_hour = now_local.replace(minute=0, second=0, microsecond=0)
    if now_local == last_full_hour:
        last_full_hour -= timedelta(hours=1)

    last_full_minute = now_local.replace(second=0, microsecond=0)
    if now_local == last_full_minute:
        last_full_minute -= timedelta(minutes=1)

    return now_local, last_full_hour, last_full_minute

def dt_local_to_ms_utc(dt_local: datetime) -> int:
    dt_utc = dt_local.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)

def ensure_data_dir() -> pathlib.Path:
    data_dir = pathlib.Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def latest_parquet_or_none(data_dir: pathlib.Path, coin: str):
    files = sorted(
        data_dir.glob(f"{coin}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[0] if files else None

def current_filename(coin: str) -> str:
    now_local = datetime.now(SAMARA_TZ)
    return f"{coin}_{now_local.strftime('%Y%m%d_%H%M')}.parquet"

def normalize_symbol(user_coin: str) -> str:
    coin = (user_coin or "").strip().upper()
    if not coin:
        return ""
    return f"{coin}USDT"

def parse_kline_rows(rows):
    out = []
    for r in rows or []:
        if isinstance(r, dict):
            ts = int(r.get("start"))
            o = float(r.get("open", 0))
            h = float(r.get("high", 0))
            l = float(r.get("low", 0))
            c = float(r.get("close", 0))
            v = float(r.get("volume", 0))
            tq = float(r.get("turnover", 0))
        else:
            ts = int(r[0])
            o, h, l, c = map(float, r[1:5])
            v = float(r[5])
            tq = float(r[6])
        out.append({
            "ts": ts,
            "open": o,
            "max": h,
            "min": l,
            "close": c,
            "volume": v,
            "volume_USDT": tq
        })
    return out

def fetch_kline_all(symbol: str, interval: str, start_ms: int, end_ms: int,
                    category: str = "linear", limit: int = 1000, pause: float = 0.15):
    url = BYBIT_BASE_URL + KLINE_ENDPOINT
    cursor = None
    total = []
    session = requests.Session()

    while True:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.0)
            continue

        if resp.status_code != 200:
            time.sleep(1.0)
            continue

        try:
            data = resp.json()
        except ValueError:
            time.sleep(0.5)
            continue

        if data.get("retCode") != 0:
            time.sleep(1.0)
            continue

        result = data.get("result", {}) or {}
        rows = result.get("list") or result.get("klineList") or []
        total.extend(parse_kline_rows(rows))

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

        time.sleep(pause)

    total.sort(key=lambda x: x["ts"])
    return total

def build_time_ranges_ms_utc():
    _, last_full_hour_local, last_full_minute_local = local_now_truncated()

    hour_end_local = last_full_hour_local
    hour_start_local = hour_end_local - timedelta(days=90) + timedelta(hours=1)

    min_end_local = last_full_minute_local
    min_start_local = min_end_local - timedelta(days=5) + timedelta(minutes=1)

    return {
        "1h": (dt_local_to_ms_utc(hour_start_local), dt_local_to_ms_utc(hour_end_local)),
        "1m": (dt_local_to_ms_utc(min_start_local), dt_local_to_ms_utc(min_end_local)),
    }

def load_existing_df(path: pathlib.Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        cols = ["ts","open","min","max","close","volume","volume_USDT","interval","symbol"]
        return pd.DataFrame(columns=cols)

def upsert_and_sort(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old is None or df_old.empty:
        base = df_new.copy()
    else:
        base = pd.concat([df_old, df_new], ignore_index=True)
    base.drop_duplicates(subset=["ts", "interval", "symbol"], keep="last", inplace=True)
    base.sort_values(by=["interval", "ts"], inplace=True)
    base.reset_index(drop=True, inplace=True)
    return base

def filter_needed_from_existing(existing: pd.DataFrame, symbol: str, interval_tag: str,
                                start_ms: int, end_ms: int):
    if existing is None or existing.empty:
        return start_ms, end_ms
    df_part = existing[(existing["symbol"] == symbol) & (existing["interval"] == interval_tag)]
    if df_part.empty:
        return start_ms, end_ms
    have_max_ts = int(df_part["ts"].max())
    if have_max_ts >= end_ms:
        return None, None
    step_ms = 60_000 if interval_tag == "1m" else 3_600_000
    new_start = max(start_ms, have_max_ts + step_ms)
    if new_start > end_ms:
        return None, None
    return new_start, end_ms

# ----------------- ОСНОВНАЯ ЛОГИКА -----------------

def process_coin(user_coin: str):
    coin = user_coin.strip().upper()
    symbol = normalize_symbol(coin)
    if not symbol:
        print("Монета не указана."); return

    print(f"\n[collector] Europe/Samara | {coin} → {symbol}")
    ranges = build_time_ranges_ms_utc()
    h_start, h_end = ranges["1h"]
    m_start, m_end = ranges["1m"]

    data_dir = ensure_data_dir()
    existing_path = latest_parquet_or_none(data_dir, coin)
    if existing_path:
        print(f"Найден файл: {existing_path.name}")
        df_exist = load_existing_df(existing_path)
    else:
        df_exist = pd.DataFrame(columns=["ts","open","min","max","close","volume","volume_USDT","interval","symbol"])

    need_h = filter_needed_from_existing(df_exist, symbol, "1h", h_start, h_end)
    need_m = filter_needed_from_existing(df_exist, symbol, "1m", m_start, m_end)

    dfs_to_merge = []
    if need_h != (None, None):
        s, e = need_h
        print("  → Гружу часовые свечи...")
        rows_h = fetch_kline_all(symbol, "60", s, e)
        if rows_h:
            df_h = pd.DataFrame(rows_h)
            df_h["interval"] = "1h"; df_h["symbol"] = symbol
            dfs_to_merge.append(df_h)
    if need_m != (None, None):
        s, e = need_m
        print("  → Гружу минутные свечи...")
        rows_m = fetch_kline_all(symbol, "1", s, e)
        if rows_m:
            df_m = pd.DataFrame(rows_m)
            df_m["interval"] = "1m"; df_m["symbol"] = symbol
            dfs_to_merge.append(df_m)

    if dfs_to_merge:
        df_new = pd.concat(dfs_to_merge, ignore_index=True)
        df_new = df_new[["ts","open","min","max","close","volume","volume_USDT","interval","symbol"]]
        df_new["ts"] = df_new["ts"].astype("int64")
        for col in ["open","min","max","close","volume","volume_USDT"]:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
        merged = upsert_and_sort(df_exist, df_new)
        save_path = existing_path if existing_path else (data_dir / current_filename(coin))
        merged.to_parquet(save_path, index=False)
        new_name = current_filename(coin)
        final_path = save_path.with_name(new_name)
        if final_path != save_path:
            shutil.move(str(save_path), str(final_path))
        print(f"  ✓ Готово: {final_path.name}, строк: {len(merged)}")
    else:
        if existing_path:
            new_name = current_filename(coin)
            final_path = existing_path.with_name(new_name)
            if final_path != existing_path:
                shutil.move(str(existing_path), str(final_path))
                print(f"  ⟳ Переименован: {final_path.name}")
            else:
                print("  ⟲ Имя файла уже актуальное.")
        else:
            print("  ⓘ Нет данных для сохранения.")

def main():
    load_dotenv()
    # если монеты передали аргументами → используем их
    if len(sys.argv) > 1:
        coins = [c.strip().upper() for c in sys.argv[1:] if c.strip()]
    else:
        # иначе читаем coins.txt
        coins_path = pathlib.Path(__file__).parent / "coins.txt"
        coins = read_coins_from_file(coins_path)

    if not coins:
        print("Список монет пуст. Добавьте монеты в coins.txt или передайте аргументами.")
        sys.exit(1)

    for coin in coins:
        try:
            process_coin(coin)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[collector] {coin}: ошибка: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено.")
