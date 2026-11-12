"""Pure EMA7 + SMA20 growth-sequence calculation used by StockLab.

The sequence starts on a close below both moving averages. It continues while
the close is at or above at least one average, and stops on a later close
below both averages. Missing market sessions interrupt an active sequence.
"""
from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {"date", "symbol", "close", "EMA7", "SMA20"}


def calculate_growth_sequences(history: pd.DataFrame) -> dict:
    """Return current, completed, interrupted and per-session growth sequences.

    A sequence is considered current only when its ticker has a row on the
    latest market date present in the dataset. A ticker missing an intervening
    market date is not allowed to bridge that gap.
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history phải là pandas.DataFrame")

    missing = _REQUIRED_COLUMNS - set(history.columns)
    if missing:
        raise ValueError("Thiếu cột lịch sử: " + ", ".join(sorted(missing)))

    gh = history.copy()
    gh["date"] = pd.to_datetime(gh["date"], errors="coerce").dt.normalize()
    for column in ("close", "EMA7", "SMA20"):
        gh[column] = pd.to_numeric(gh[column], errors="coerce")
    gh["symbol"] = gh["symbol"].astype(str).str.upper().str.strip()
    gh = gh.dropna(subset=["date", "symbol", "close", "EMA7", "SMA20"])
    gh = gh[gh["symbol"].ne("")].sort_values(["symbol", "date"])
    gh = gh.drop_duplicates(["symbol", "date"], keep="last")

    if gh.empty:
        return {
            "latest_market_date": None,
            "current": [],
            "detail": [],
            "completed": [],
            "interrupted": [],
        }

    market_dates = sorted(gh["date"].unique())
    latest_market_date = market_dates[-1]
    detail, completed, interrupted, current = [], [], [], []

    for symbol, part in gh.groupby("symbol", sort=True):
        part = part.sort_values("date").reset_index(drop=True)
        ticker_dates = set(part["date"])
        active, count, start_date, previous_date = False, 0, None, None

        for _, row in part.iterrows():
            day = row["date"]

            if previous_date is not None and active:
                has_gap = any(
                    previous_date < market_day < day and market_day not in ticker_dates
                    for market_day in market_dates
                )
                if has_gap:
                    interrupted.append({
                        "Mã": symbol,
                        "Ngày bắt đầu": start_date,
                        "Ngày dữ liệu cuối": previous_date,
                        "Số ngày": count,
                        "Trạng thái": "Gián đoạn · thiếu phiên dữ liệu",
                    })
                    active, count, start_date = False, 0, None

            below_both = row["close"] < row["EMA7"] and row["close"] < row["SMA20"]

            if not active:
                if below_both:
                    active, count, start_date = True, 1, day
                    detail.append({
                        "Mã": symbol,
                        "Ngày": day,
                        "Sự kiện": "Bắt đầu · Ngày 1",
                        "Số ngày": count,
                        "Đóng cửa": row["close"],
                        "EMA7": row["EMA7"],
                        "SMA20": row["SMA20"],
                    })
            elif below_both:
                completed.append({
                    "Mã": symbol,
                    "Ngày bắt đầu": start_date,
                    "Ngày kết thúc": previous_date,
                    "Số ngày": count,
                    "Trạng thái": "Đã kết thúc · đóng cửa dưới cả EMA7 và SMA20",
                })
                # The stopping session is excluded from the completed sequence.
                active, count, start_date = False, 0, None
            else:
                # Continue when close >= EMA7 OR close >= SMA20.
                count += 1
                detail.append({
                    "Mã": symbol,
                    "Ngày": day,
                    "Sự kiện": "Tiếp tục",
                    "Số ngày": count,
                    "Đóng cửa": row["close"],
                    "EMA7": row["EMA7"],
                    "SMA20": row["SMA20"],
                })

            previous_date = day

        if active and previous_date == latest_market_date:
            current.append({
                "Mã": symbol,
                "Ngày bắt đầu": start_date,
                "Ngày cập nhật": previous_date,
                "Số ngày": count,
            })
        elif active:
            interrupted.append({
                "Mã": symbol,
                "Ngày bắt đầu": start_date,
                "Ngày dữ liệu cuối": previous_date,
                "Số ngày": count,
                "Trạng thái": "Chưa xác nhận · mã thiếu phiên mới nhất",
            })

    return {
        "latest_market_date": latest_market_date,
        "current": current,
        "detail": detail,
        "completed": completed,
        "interrupted": interrupted,
    }
