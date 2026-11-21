"""Daily market state for the StockLab buy and sell indicator."""
from pathlib import Path

import pandas as pd

from lightweight_chart import calculate_stochastic_trial, stochastic_trade_events

STATE_PATH = Path(__file__).with_name("stocklab_trade_signals.csv")
STATE_COLUMNS = [
    "symbol", "exchange", "data_date", "last_close", "buy_date",
    "buy_price", "sessions", "ready", "fresh",
]


def load_signal_state():
    if not STATE_PATH.exists():
        return {}
    rows = pd.read_csv(STATE_PATH, keep_default_na=False).to_dict("records")
    return {row["symbol"]: row for row in rows}


def update_signal_state(symbol, exchange, bars, previous):
    """Preserve active entries over rolling history and replay new candles."""
    dates = bars["date"].dt.strftime("%Y-%m-%d").tolist()
    if not dates or (previous and previous["data_date"] not in dates):
        raise ValueError("Không có phiên mốc để nối tiếp trạng thái")

    previous_date = previous["data_date"] if previous else ""
    entry_date = previous["buy_date"] if previous else ""
    entry_price = float(previous["buy_price"] or 0) if previous else 0.0
    sessions = int(previous["sessions"]) if previous else 0
    events, ready = stochastic_trade_events(
        calculate_stochastic_trial(bars), bars["date"], after=previous_date,
        holding=bool(entry_date),
        ready_to_buy=bool(int(previous["ready"])) if previous else False,
    )
    indexes = {day: i for i, day in enumerate(dates)}
    if entry_date:
        sessions += len(dates) - 1 - indexes[previous_date]
    for event in events:
        event_date = event["time"]
        if event["text"] == "Mua":
            entry_date = event_date
            entry_price = float(bars.iloc[indexes[event_date]]["close"])
            sessions = len(dates) - 1 - indexes[event_date]
        else:
            entry_date, entry_price, sessions = "", 0.0, 0
    return {
        "symbol": symbol, "exchange": exchange, "data_date": dates[-1],
        "last_close": float(bars.iloc[-1]["close"]), "buy_date": entry_date,
        "buy_price": entry_price, "sessions": sessions, "ready": int(ready),
        "fresh": 1,
    }
