"""Vietnam stock scanner and technical charts."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO
import os
import pandas as pd
import requests
import streamlit as st

EXCHANGES = {"HOSE", "HNX", "UPCOM"}
VNSTOCK_SOURCE_KEYS = {"dnse": ("DNSE_API_KEY", "DNSE_API_SECRET"),
                       "fhsc": ("FHSC_API_KEY",), "ssi": ("SSI_API_KEY", "SSI_API_SECRET"),
                       "tcbs": ("TCBS_API_KEY",)}
COMPARE_INDICES = {"VN-Index": "VNINDEX", "VN30": "VN30"}
LONG_HISTORY_DAYS = (5 * 365, 10 * 365, 20 * 365, 30 * 365)
# Add dates only after the exchanges confirm the equity trading schedule.
MARKET_HOLIDAY_PERIODS = (
    ("Tết Dương lịch", "2026-01-01", "2026-01-02"),
    ("Tết Nguyên đán", "2026-02-16", "2026-02-20"),
    ("Giỗ Tổ Hùng Vương", "2026-04-27", "2026-04-27"),
    ("30/4 và Quốc tế Lao động", "2026-04-30", "2026-05-01"),
    ("Quốc khánh", "2026-08-31", "2026-09-02"),
    ("Ngày Văn hóa Việt Nam", "2026-11-24", "2026-11-24"),
)
MARKET_HOLIDAYS = {
    day for _, start, end in MARKET_HOLIDAY_PERIODS for day in pd.date_range(start, end)
}


def vietnam_market_date() -> date:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()


def drop_market_holiday_bars(bars: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    return bars.loc[~dates.isin(MARKET_HOLIDAYS)].copy()


def show_market_holiday(today: date) -> None:
    for name, first_day, last_day in MARKET_HOLIDAY_PERIODS:
        first, last = date.fromisoformat(first_day), date.fromisoformat(last_day)
        if not first - timedelta(days=2) <= today <= last:
            continue
        reopening = last + timedelta(days=1)
        while reopening.weekday() >= 5 or pd.Timestamp(reopening) in MARKET_HOLIDAYS:
            reopening += timedelta(days=1)
        period = f"{first:%d/%m/%Y}" if first == last else f"{first:%d/%m/%Y}–{last:%d/%m/%Y}"
        status = "Sắp nghỉ giao dịch" if today < first else "Thị trường nghỉ giao dịch"
        st.info(f"{status} dịp {name} ({period}). Giao dịch trở lại ngày {reopening:%d/%m/%Y}. "
                "Biểu đồ và tín hiệu vẫn xem được đến phiên có dữ liệu hợp lệ gần nhất.")


def history_label(days: int) -> str:
    return f"{days // 365} năm" if days in LONG_HISTORY_DAYS else f"{days} ngày"
class MarketDataError(RuntimeError):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_instruments() -> pd.DataFrame:
    """Build HOSE/HNX/UPCOM universe from Cophieu68 public listings."""
    from cophieu68_import import fetch_cophieu68_listing
    try:
        return fetch_cophieu68_listing()
    except Exception as exc:
        raise MarketDataError(f"Không lấy được danh sách mã từ Cophieu68: {exc}") from exc

def _normalize_quotes(payload) -> pd.DataFrame:
    df = pd.DataFrame(payload).copy()
    mapping = {"time": "date", "date": "date", "open": "open",
               "high": "high", "low": "low", "close": "close", "volume": "volume"}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MarketDataError("Thiếu trường dữ liệu OHLCV: " + ", ".join(missing))
    df = df[required].copy()
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    # Keep date-only/naive market dates as Vietnam trading dates. Convert only
    # genuinely timezone-aware timestamps, then remove the time component.
    if isinstance(parsed_dates.dtype, pd.DatetimeTZDtype):
        parsed_dates = parsed_dates.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    df["date"] = parsed_dates.dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_quotes(symbol: str, start_date: str, end_date: str, interval: str = "1D",
                       is_index: bool = False) -> tuple[pd.DataFrame, str]:
    """Load OHLCV from a configured Vnstock 6 provider."""
    sources = [name for name, keys in VNSTOCK_SOURCE_KEYS.items()
               if all(os.getenv(key, "").strip() for key in keys)]
    if not sources:
        raise MarketDataError("Chưa cấu hình nguồn Vnstock 6.")
    from vnstock import Market

    for source in sources:
        try:
            market = Market(source)
            asset = market.index(symbol) if is_index else market.equity(symbol)
            raw = asset.ohlcv(start=start_date, end=end_date, interval=interval)
            quotes = _normalize_quotes(raw).dropna(subset=["volume"])
            if not quotes.empty:
                return quotes, f"Vnstock {source.upper()}"
        except Exception:
            continue
    raise MarketDataError(f"Nguồn Vnstock đã cấu hình không trả dữ liệu cho {symbol}.")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_chart_quotes(symbol: str, start_date: str, end_date: str, interval: str = "1D") -> tuple[pd.DataFrame, str]:
    """Use public Cophieu68 daily history first; use Vnstock for intraday candles."""
    if interval == "1D":
        try:
            from cophieu68_import import fetch_cophieu68_history
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            max_pages = 120 if (end - start).days > 730 else 30
            history = fetch_cophieu68_history(symbol, max_pages=max_pages, start_date=start)
            history = history[(history["date"] >= start) & (history["date"] <= end)]
            normalized = _normalize_quotes(history)
            if len(normalized) >= 20:
                return normalized, "Cophieu68 · dữ liệu ngày"
        except Exception as exc:
            cp68_error = str(exc)
        else:
            cp68_error = "Cophieu68 trả về không đủ dữ liệu"
    else:
        cp68_error = "Cophieu68 chỉ dùng cho biểu đồ ngày 1D"
    try:
        return fetch_daily_quotes(symbol, start_date, end_date, interval)
    except Exception as exc:
        raise MarketDataError(f"Cophieu68: {cp68_error}. Vnstock: {exc}") from exc

@st.cache_data(ttl=900, show_spinner=False)
def fetch_index_chart_quotes(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """Load index candles from public chart feeds, then configured Vnstock."""
    start = pd.Timestamp(start_date, tz="Asia/Ho_Chi_Minh")
    end = pd.Timestamp(end_date, tz="Asia/Ho_Chi_Minh") + pd.Timedelta(days=1)
    params = {"symbol": symbol, "resolution": "D", "from": int(start.timestamp()),
              "to": int(end.timestamp()) - 1}
    providers = (
        ("DNSE", "https://services.entrade.com.vn/chart-api/v2/ohlcs/index", "1D"),
        ("VNDIRECT", "https://dchart-api.vndirect.com.vn/dchart/history", "D"),
        ("VPS", "https://histdatafeed.vps.com.vn/tradingview/history", "D"),
    )
    columns = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    for source, url, resolution in providers:
        try:
            response = requests.get(url, params={**params, "resolution": resolution}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            bars = pd.DataFrame({name: payload[key] for key, name in columns.items()})
            bars["time"] = pd.to_datetime(bars["time"], unit="s", utc=True, errors="coerce")
            quotes = _normalize_quotes(bars).dropna(subset=["volume"])
            quotes = quotes[quotes["date"].between(start_date, end_date)]
            if not quotes.empty:
                return quotes, f"{source} · chỉ số"
        except (requests.RequestException, KeyError, ValueError, TypeError):
            continue

    try:
        response = requests.get(f"https://simplize.vn/chi-so/{symbol}/lich-su-gia", timeout=10)
        response.raise_for_status()
        for table in pd.read_html(StringIO(response.text), flavor="lxml"):
            columns = {"Ngày": "time", "Mở cửa": "open", "Cao": "high", "Thấp": "low",
                       "Đóng cửa": "close", "Tổng khối lượng": "volume"}
            if not set(columns).issubset(table.columns):
                continue
            bars = table[list(columns)].rename(columns=columns)
            bars["time"] = pd.to_datetime(bars["time"], format="%d/%m/%Y", errors="coerce")
            for column in ("open", "high", "low", "close", "volume"):
                bars[column] = bars[column].astype(str).str.replace(",", "", regex=False)
            quotes = _normalize_quotes(bars).dropna(subset=["volume"])
            quotes = quotes[quotes["date"].between(start_date, end_date)]
            if not quotes.empty:
                return quotes, "Simplize · số phiên công khai"
    except (requests.RequestException, ValueError):
        pass

    try:
        slug = {"VNINDEX": "vn-index", "VN30": "vn-30"}[symbol]
        response = requests.get(f"https://www.investing.com/indices/{slug}-historical-data", timeout=10)
        response.raise_for_status()
        for table in pd.read_html(StringIO(response.text), flavor="lxml"):
            columns = {"Date": "time", "Open": "open", "High": "high", "Low": "low",
                       "Price": "close", "Vol.": "volume"}
            if not set(columns).issubset(table.columns):
                continue
            bars = table[list(columns)].rename(columns=columns)
            bars["time"] = pd.to_datetime(bars["time"], format="%b %d, %Y", errors="coerce")
            volume = bars["volume"].astype(str).str.strip().str.extract(r"^([0-9.,]+)\s*([KMB]?)$")
            bars["volume"] = (pd.to_numeric(volume[0].str.replace(",", "", regex=False), errors="coerce")
                              * volume[1].map({"": 1, "K": 1e3, "M": 1e6, "B": 1e9}))
            quotes = _normalize_quotes(bars).dropna(subset=["volume"])
            quotes = quotes[quotes["date"].between(start_date, end_date)]
            if not quotes.empty:
                return quotes, "Investing.com · số phiên công khai"
    except (requests.RequestException, ValueError):
        pass

    try:
        return fetch_daily_quotes(symbol, start_date, end_date, is_index=True)
    except Exception as exc:
        raise MarketDataError(f"Nguồn dữ liệu chưa trả được nến cho {symbol}.") from exc


def sanitize_daily_chart_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Normalize trading dates and omit a suspicious latest one-lot flat candle."""
    out = drop_market_holiday_bars(df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(out) < 21:
        return out, None
    latest = out.iloc[-1]
    prior_median_volume = pd.to_numeric(out.iloc[-21:-1]["volume"], errors="coerce").median()
    price_values = [float(latest[c]) for c in ("open", "high", "low", "close")]
    flat_candle = max(price_values) - min(price_values) <= max(abs(float(latest["close"])) * 1e-8, 1e-8)
    low_volume = pd.notna(prior_median_volume) and float(latest["volume"]) <= max(100.0, float(prior_median_volume) * 0.01)
    if flat_candle and low_volume:
        dropped_date = latest["date"].strftime("%d/%m/%Y")
        dropped_volume = float(latest["volume"])
        out = out.iloc[:-1].reset_index(drop=True)
        return out, (
            f"Tạm loại phiên {dropped_date} khỏi chỉ báo: nến đứng giá và khối lượng "
            f"{dropped_volume:,.0f} cp thấp bất thường (≤ 1% trung vị 20 phiên trước). "
            "Phiên này vẫn cần đối chiếu lại với nguồn dữ liệu."
        )
    return out, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA7"] = out["close"].ewm(span=7, adjust=False, min_periods=7).mean()
    out["SMA20"] = out["close"].rolling(20, min_periods=20).mean()
    out["EMA20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["EMA50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["EMA100"] = out["close"].ewm(span=100, adjust=False, min_periods=100).mean()
    # FireAnt Mom: length 10, source Close; value is a price difference, not a percentage.
    out["MOM10"] = out["close"] - out["close"].shift(10)
    # ROC14 is percentage change from 14 trading sessions earlier.
    out["ROC14"] = (out["close"] / out["close"].shift(14) - 1) * 100
    out["VOL20"] = out["volume"].rolling(20, min_periods=20).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    out["RSI14"] = 100 - (100 / (1 + rs))
    out.loc[(loss == 0) & (gain > 0), "RSI14"] = 100
    out.loc[(loss == 0) & (gain == 0), "RSI14"] = 50
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out

def chart_with_warmed_ema100(daily_quotes, timeframe, display_start):
    """Calculate EMA100 on earlier candles while keeping the chosen display range."""
    from stock_compare import aggregate_candles

    history = aggregate_candles(daily_quotes, timeframe)
    ema100 = history.set_index("date")["close"].ewm(
        span=100, adjust=False, min_periods=100
    ).mean()
    visible_quotes = daily_quotes.loc[daily_quotes["date"] >= pd.Timestamp(display_start)]
    chart = add_indicators(aggregate_candles(visible_quotes, timeframe))
    chart["EMA100"] = chart["date"].map(ema100)
    return chart


# Deliberately not cached: scan_market updates a live Streamlit progress element.
def _scan_market_uncached_v2(instruments: pd.DataFrame, lookback_days: int = 260) -> tuple[pd.DataFrame, dict]:
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()
    rows, errors = [], []
    symbols = instruments.to_dict("records")
    for i, item in enumerate(symbols):
        symbol = item["symbol"]
        try:
            quotes, _ = fetch_daily_quotes(symbol, start, end)
            hist = add_indicators(quotes)
            valid = hist.dropna(subset=["EMA100", "MOM10", "ROC14", "VOL20"])
            if not valid.empty:
                last = valid.iloc[-1]
                rows.append({
                    "Mã": symbol, "Sàn": item.get("exchange", ""),
                    "Giá đóng cửa": last["close"], "EMA100": last["EMA100"],
                    "MOM10 (FireAnt)": last["MOM10"], "ROC14 (%)": last["ROC14"], "Khối lượng": last["volume"],
                    "KLGD TB20": last["VOL20"], "Ngày dữ liệu": last["date"],
                    "Đạt ROC14": bool(last["ROC14"] >= 4),
                    "Đạt EMA100": bool(last["close"] >= last["EMA100"]),
                    "Đạt MOM10": bool(last["MOM10"] >= 4),
                    "Đạt bộ lọc": bool(last["MOM10"] >= 4 and last["ROC14"] >= 4 and last["close"] >= last["EMA100"]),
                })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Đạt bộ lọc", "ROC14 (%)"], ascending=[False, False]).reset_index(drop=True)
    return result, {"scanned": len(symbols), "success": len(rows), "errors": errors}

def load_saved_market_scan() -> tuple[pd.DataFrame | None, dict | None]:
    """Load the latest automated Cophieu68 scan committed by GitHub Actions."""
    from pathlib import Path
    path = Path(__file__).resolve().parent / "stocklab_market_all.csv"
    passed_path = Path(__file__).resolve().parent / "stocklab_market_passed.csv"
    if not path.exists():
        return None, None
    try:
        raw = pd.read_csv(path)
        required = {"symbol", "exchange", "date", "close", "EMA100", "MOM10", "ROC14", "VOL20"}
        if raw.empty or not required.issubset(raw.columns):
            raise ValueError("CSV thiếu cột bắt buộc hoặc không có dữ liệu")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for col in ("close", "EMA100", "MOM10", "ROC14", "VOL20"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna(subset=["symbol", "date", "close", "EMA100", "MOM10", "ROC14", "VOL20"])
        if raw.empty:
            raise ValueError("Không còn dòng OHLCV hợp lệ sau khi chuẩn hóa")
        result = pd.DataFrame({
            "Mã": raw["symbol"].astype(str).str.upper(),
            "Sàn": raw["exchange"].astype(str).str.upper(),
            "Giá đóng cửa": raw["close"],
            "EMA100": raw["EMA100"],
            "MOM10 (FireAnt)": raw["MOM10"],
            "ROC14 (%)": raw["ROC14"],
            "Khối lượng": pd.to_numeric(raw["volume"], errors="coerce") if "volume" in raw.columns else pd.NA,
            "KLGD TB20": raw["VOL20"],
            "Ngày dữ liệu": raw["date"],
        })
        updated = str(raw["updated_at_utc"].iloc[0]) if "updated_at_utc" in raw.columns else "không rõ"
        passed_count = None
        if passed_path.exists():
            passed = pd.read_csv(passed_path)
            passed_count = len(passed)
        stats = {
            "scanned": len(raw), "success": len(raw), "total": len(raw),
            "complete": True, "errors": [], "source": "GitHub Actions · Cophieu68",
            "updated_at_utc": updated, "saved_passed_count": passed_count,
        }
        return result, stats
    except Exception as exc:
        raise MarketDataError(f"Không đọc được bản quét tự động ({path.name}): {exc}") from exc


def filter_signal_columns(table: pd.DataFrame, key: str) -> pd.DataFrame:
    """Apply optional filters selected per displayed column."""
    filters = st.multiselect("Cột cần lọc", list(table.columns), key=f"{key}_columns")
    matches = pd.Series(True, index=table.index)
    for column in filters:
        values = table[column]
        widget_key = f"{key}_{column}"
        left, right = st.columns(2)
        if pd.api.types.is_numeric_dtype(values):
            lower = left.number_input(f"{column} từ", value=None, key=f"{widget_key}_from")
            upper = right.number_input(f"{column} đến", value=None, key=f"{widget_key}_to")
        elif column.startswith("Ngày "):
            lower = left.date_input(f"{column} từ", value=None, key=f"{widget_key}_from")
            upper = right.date_input(f"{column} đến", value=None, key=f"{widget_key}_to")
            values = pd.to_datetime(values, errors="coerce").dt.date
        elif column in {"Sàn", "Dữ liệu"}:
            selected = left.multiselect(column, sorted(values.dropna().unique()), key=widget_key)
            if selected:
                matches &= values.isin(selected)
            continue
        else:
            query = left.text_input(f"Tìm {column.lower()}", key=widget_key).strip()
            if query:
                matches &= values.str.contains(query, case=False, regex=False, na=False)
            continue
        if lower is not None:
            matches &= values.ge(lower)
        if upper is not None:
            matches &= values.le(upper)
    return table.loc[matches]


def render_stock_dashboard():
    st.header("📈 STOCKLAB VIỆT NAM")
    show_market_holiday(vietnam_market_date())
    st.caption("Bảng giá · Bộ lọc cổ phiếu · Biểu đồ nến · EMA · RSI · MACD · Khối lượng")

    st.caption("Nguồn dữ liệu: Cophieu68 cho bảng quét và biểu đồ cổ phiếu ngày. Biểu đồ chỉ số ưu tiên nguồn công khai.")

    left, right = st.columns([1, 1])
    with left:
        days = st.selectbox("Số ngày lịch sử", [260, 365, 520, 730], index=1)
    with right:
        min_volume = st.number_input("Khối lượng tối thiểu (cổ phiếu)", min_value=0, value=50000, step=10000)
    mom10_floor = st.number_input("MOM10 tối thiểu (giá)", value=4.0, step=0.5)
    require_ema = st.checkbox("Chỉ lấy giá đóng cửa ≥ EMA100", value=True)
    st.caption("Dữ liệu thị trường được cập nhật tự động từ Cophieu68. Nút bên dưới chỉ nạp lại bản quét đầy đủ, không gọi Vnstock hàng loạt.")

    scan_col, reset_col = st.columns([3, 1])
    with reset_col:
        if st.button("Xóa kết quả", use_container_width=True):
            for key in ("vnstock_scan_offset", "vnstock_scan_all", "vnstock_scan", "vnstock_scan_stats", "vnstock_instruments", "vnstock_days"):
                st.session_state.pop(key, None)
            st.rerun()
    with scan_col:
        scan_clicked = st.button("🔄 Cập nhật dữ liệu và quét thị trường", type="primary", use_container_width=True)
    if scan_clicked:
        try:
            # Prefer the complete scheduled Cophieu68 snapshot. Do not start
            # hundreds of Vnstock requests from a mobile button: that quickly
            # exhausts the per-minute quota and leaves a misleading partial scan.
            saved_result, saved_stats = load_saved_market_scan()
            if saved_result is None or saved_result.empty:
                raise MarketDataError(
                    "Chưa có dữ liệu quét tự động. Hãy chờ GitHub Actions cập nhật dữ liệu thị trường."
                )
            for key in (
                "vnstock_scan_offset", "vnstock_scan_all", "vnstock_scan",
                "vnstock_scan_stats", "vnstock_scan_config", "vnstock_instruments",
                "vnstock_days", "vnstock_scan_source",
            ):
                st.session_state.pop(key, None)
            st.session_state["vnstock_scan"] = saved_result
            st.session_state["vnstock_scan_all"] = saved_result
            st.session_state["vnstock_scan_stats"] = saved_stats
            st.session_state["vnstock_scan_source"] = "automatic"
            st.success(
                f"Đã nạp dữ liệu quét tự động: {len(saved_result):,} mã có dữ liệu. "
                "Không gọi Vnstock hàng loạt nên không tiêu hao hạn mức API."
            )
        except Exception as exc:
            st.error(f"Không thể nạp dữ liệu quét tự động: {exc}")
            st.stop()

    result = st.session_state.get("vnstock_scan")
    stats = st.session_state.get("vnstock_scan_stats")
    if result is None or result.empty:
        try:
            result, stats = load_saved_market_scan()
            if result is not None:
                st.session_state["vnstock_scan"] = result
                st.session_state["vnstock_scan_stats"] = stats
                st.session_state["vnstock_scan_all"] = result
                st.session_state["vnstock_scan_source"] = "automatic"
        except MarketDataError as exc:
            st.warning(str(exc))
    if result is None or result.empty:
        st.info("Chưa có bản quét tự động. Workflow GitHub Actions sẽ cập nhật theo lịch ngày giao dịch; mày cũng có thể nhấn nút cập nhật để quét thủ công.")
        from cophieu68_import import render_cophieu68_import
        render_cophieu68_import()
        return

    if stats and stats.get("source"):
        st.success(
            f"Bản quét tự động: {stats['success']:,}/{stats['total']:,} mã có dữ liệu · "
            f"{stats.get('saved_passed_count', '—')} mã đạt bộ lọc mặc định · "
            f"cập nhật UTC: {stats.get('updated_at_utc', 'không rõ')}"
        )

    result = result.copy()
    result["Đạt MOM10"] = result["MOM10 (FireAnt)"] >= mom10_floor
    result["Đạt EMA100"] = result["Giá đóng cửa"] >= result["EMA100"]
    result["Đạt khối lượng"] = result["KLGD TB20"] >= min_volume
    # Two independent momentum screens; share only volume and optional EMA constraints.
    common_pass = result["Đạt khối lượng"]
    if require_ema:
        common_pass = common_pass & result["Đạt EMA100"]
    result["Đạt MOM10 filter"] = result["Đạt MOM10"] & common_pass
    filtered = result[common_pass].copy()

    if stats:
        a, b, c = st.columns(3)
        a.metric("Mã đã quét", f"{stats['scanned']:,}", help=f"Tổng danh sách: {stats.get('total', stats['scanned']):,}")
        b.metric("Mã có dữ liệu", f"{stats['success']:,}")
        c.metric("Mã đạt bộ lọc", f"{len(filtered):,}")
        if stats.get("complete") and not stats.get("source"):
            st.success("Đã quét hết danh sách mã hiện tại.")
        else:
            remaining = max(0, stats.get("total", stats["scanned"]) - stats["scanned"])
            st.info(f"Còn {remaining:,} mã chưa quét. Nhấn nút cập nhật để tiếp tục lượt kế tiếp.")
        if stats.get("errors"):
            with st.expander(f"Lỗi dữ liệu trong các lượt quét ({len(stats['errors'])})"):
                st.write(stats["errors"][-50:])

    tab_scan, tab_chart, tab_compare, tab_all = st.tabs(["🔎 Bộ lọc cổ phiếu", "📊 Biểu đồ kỹ thuật", "🕯️ So sánh 2 CP", "📋 Toàn bộ thị trường"])
    with tab_scan:
        mom_tab, roc_tab, signals_tab = st.tabs(["MOM10 ≥ ngưỡng", "ROC14 ≥ ngưỡng", "Tín hiệu mua bán"])
        with roc_tab:
            st.subheader("Danh sách ROC14 độc lập · phần trăm thay đổi 14 phiên")
            st.caption("Không yêu cầu MOM10 đạt ngưỡng. Vẫn áp dụng khối lượng tối thiểu và EMA100 nếu bật.")
            roc_floor = st.number_input("ROC14 tối thiểu (%)", value=4.0, step=0.5)
            result["Đạt ROC14"] = result["ROC14 (%)"] >= roc_floor
            result["Đạt ROC14 filter"] = result["Đạt ROC14"] & common_pass
            filtered_roc14 = result[result["Đạt ROC14 filter"]].copy().sort_values("ROC14 (%)", ascending=False).reset_index(drop=True)
            st.metric("Số mã đạt ROC14", f"{len(filtered_roc14):,}")
            st.dataframe(filtered_roc14[["Mã", "Sàn", "Giá đóng cửa", "MOM10 (FireAnt)", "ROC14 (%)", "EMA100", "Khối lượng", "KLGD TB20", "Ngày dữ liệu"]], use_container_width=True, hide_index=True)
            st.download_button("Tải CSV ROC14", filtered_roc14.to_csv(index=False).encode("utf-8-sig"), "co_phieu_ROC14_dat_nguong.csv", "text/csv")
        filtered_mom10 = result[result["Đạt MOM10 filter"]].copy().sort_values("MOM10 (FireAnt)", ascending=False).reset_index(drop=True)
        with mom_tab:
            st.subheader("Danh sách MOM10 độc lập · FireAnt Length 10, Close")
            st.caption("Không yêu cầu ROC14 đạt ngưỡng. Vẫn áp dụng khối lượng tối thiểu và EMA100 nếu bật.")
            st.metric("Số mã đạt MOM10", f"{len(filtered_mom10):,}")
            st.dataframe(filtered_mom10[["Mã", "Sàn", "Giá đóng cửa", "MOM10 (FireAnt)", "ROC14 (%)", "EMA100", "Khối lượng", "KLGD TB20", "Ngày dữ liệu"]], use_container_width=True, hide_index=True)
            st.download_button("Tải CSV MOM10", filtered_mom10.to_csv(index=False).encode("utf-8-sig"), "co_phieu_MOM10_dat_nguong.csv", "text/csv")

        with signals_tab:
            from stochastic_signals import load_signal_state

            signal_rows = load_signal_state()
            if signal_rows:
                signals = pd.DataFrame(signal_rows.values())
                latest = signals.loc[signals["fresh"].eq(1), "data_date"].max()
                st.caption(f"Bộ tín hiệu mua bán 4 · Khung D1 · Phiên dữ liệu mới nhất: {latest} · {signals['fresh'].sum():,} mã cập nhật.")
                volume_floor = st.number_input("Khối lượng phiên gần nhất từ (cổ phiếu)",
                                               min_value=100000, value=100000, step=10000,
                                               key="signals_volume_floor_100k")
                tracked = signals["buy_date"].ne("") & signals["buy_price"].gt(0)
                missing_volume = (tracked & signals["last_volume"].isna()).sum()
                if missing_volume:
                    st.info(f"{missing_volume:,} mã đang theo dõi chưa có khối lượng để áp dụng bộ lọc.")
                active = signals.loc[tracked & signals["last_volume"].ge(volume_floor)].copy()
                active["% lợi nhuận"] = (active["last_close"] / active["buy_price"] - 1) * 100
                active["Dữ liệu"] = "Cũ"
                active.loc[active["fresh"].eq(1) & active["data_date"].eq(latest), "Dữ liệu"] = "Mới nhất"
                active = active.rename(columns={
                    "symbol": "Mã", "exchange": "Sàn", "buy_date": "Ngày báo mua",
                    "buy_type": "Tín hiệu",
                    "buy_price": "Giá báo mua", "last_close": "Giá gần nhất",
                    "sessions": "Số phiên", "data_date": "Ngày giá gần nhất", "last_volume": "Khối lượng",
                })
                buys = active.loc[active["Ngày báo mua"].eq(latest) & active["Dữ liệu"].eq("Mới nhất")]
                st.subheader("Tín hiệu mua")
                buy_columns = ["Mã", "Sàn", "Tín hiệu", "Ngày báo mua", "Giá báo mua", "Khối lượng"]
                with st.expander("Lọc cột bảng Tín hiệu mua"):
                    buys = filter_signal_columns(buys[buy_columns], "signals_buy")
                st.caption(f"{len(buys):,} mã phù hợp")
                st.dataframe(buys.sort_values("Mã"), hide_index=True, use_container_width=True)

                st.subheader("Đang theo dõi")
                st.caption("Giá báo mua là giá đóng cửa của phiên có tín hiệu. Số phiên bắt đầu từ 0. % lợi nhuận chưa tính phí.")
                tracking_columns = ["Mã", "Sàn", "Ngày báo mua", "Giá báo mua", "Giá gần nhất",
                                    "Khối lượng", "% lợi nhuận", "Số phiên", "Ngày giá gần nhất", "Dữ liệu"]
                with st.expander("Lọc cột bảng Đang theo dõi"):
                    active = filter_signal_columns(active[tracking_columns], "signals_tracking")
                st.caption(f"{len(active):,} mã phù hợp")
                st.dataframe(
                    active.sort_values(["Ngày báo mua", "Mã"], ascending=[False, True]),
                    column_config={"% lợi nhuận": st.column_config.NumberColumn(format="%.2f%%")},
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("Hai bảng sẽ có dữ liệu sau lượt quét D1 đầu tiên của Bộ tín hiệu mua bán 4.")

    available_symbols = sorted({
        str(symbol).strip().upper()
        for symbol in result["Mã"].dropna()
        if str(symbol).strip()
    })
    index_codes = set(COMPARE_INDICES.values())
    symbol_options = [*COMPARE_INDICES, *(
        symbol for symbol in available_symbols if symbol.replace("-", "") not in index_codes
    )]

    default_chart_symbol = "VN-Index"
    with tab_chart:
        selected_symbols = st.multiselect(
            "Tìm mã cổ phiếu hoặc chỉ số", symbol_options,
            default=[default_chart_symbol],
            placeholder="Gõ chữ đầu của mã, ví dụ F hoặc VN",
            help="Chọn một hoặc nhiều mã. Mã đầu tiên sẽ hiện trên biểu đồ.",
            accept_new_options=True, filter_mode="prefix",
            key="stocklab_chart_symbol_picker",
        )
        chart_day_options = [260, 365, 520, 730, *LONG_HISTORY_DAYS]
        chart_days = st.selectbox(
            "Số ngày lịch sử biểu đồ", chart_day_options,
            index=chart_day_options.index(days),
            format_func=history_label,
            key="stocklab_chart_days",
        )
        timeframe_labels = {"1D": "D1", "2D": "D2", "3D": "D3", "1W": "Tuần", "1M": "Tháng"}
        timeframe = st.radio(
            "Khung nến biểu đồ", list(timeframe_labels), format_func=timeframe_labels.__getitem__,
            horizontal=True, key="stocklab_chart_timeframe",
        )
        index_labels = {code: label for label, code in COMPARE_INDICES.items()}
        chart_symbols = []
        for item in selected_symbols:
            symbol = item.strip().upper()
            symbol = index_labels.get(symbol.replace("-", ""), symbol)
            if symbol and symbol not in chart_symbols:
                chart_symbols.append(symbol)
        selected = chart_symbols[0] if chart_symbols else default_chart_symbol
        st.caption("Biểu đồ hiển thị mã đầu tiên. Bảng đối chiếu bên dưới dùng dữ liệu quét cổ phiếu.")
        try:
            today = vietnam_market_date()
            start = (today - timedelta(days=chart_days)).isoformat()
            end = today.isoformat()
            # Short display ranges reuse the same EMA100 calculation history.
            warmup_days = {"1D": 365, "2D": 730, "3D": 1095, "1W": 1460, "1M": 4383}
            history_start = (
                today - timedelta(days=max(chart_days, 730) + warmup_days[timeframe])
            ).isoformat()
            chart_fetch = fetch_index_chart_quotes if selected in COMPARE_INDICES else fetch_chart_quotes
            ticker = COMPARE_INDICES.get(selected, selected)
            try:
                raw_chart, chart_source = chart_fetch(ticker, history_start, end)
            except MarketDataError:
                raw_chart, chart_source = chart_fetch(ticker, start, end)
            if selected in COMPARE_INDICES:
                clean_chart, chart_warning = drop_market_holiday_bars(raw_chart), None
            else:
                clean_chart, chart_warning = sanitize_daily_chart_bars(raw_chart)
            st.caption(f"Nguồn dữ liệu biểu đồ: {chart_source}.")
            if chart_warning:
                st.warning(chart_warning)
            chart = chart_with_warmed_ema100(clean_chart, timeframe, start)
            chart = chart.dropna(subset=["date", "open", "high", "low", "close"])
            if chart.empty:
                st.warning("Chưa có dữ liệu OHLC hợp lệ cho mã này.")
            else:
                last = chart.iloc[-1]
                bar_unit = "phiên" if timeframe == "1D" else "nến"
                first_date = chart["date"].min()
                st.caption(f"Dữ liệu thực tế: {first_date:%d/%m/%Y} đến {last['date']:%d/%m/%Y} · {len(chart):,} {bar_unit}.")
                price_metric, volume_metric = st.columns(2)
                close_label = "Điểm đóng cửa" if selected in COMPARE_INDICES else "Giá đóng cửa"
                price_metric.metric(close_label, f"{last['close']:,.2f}")
                volume_metric.metric("Khối lượng", f"{last['volume']:,.0f}")
                from lightweight_chart import render_lightweight_chart
                render_lightweight_chart(chart, selected, timeframe=timeframe)
                st.caption("Mũi tên của Bộ chỉ báo mua bán được tính theo khung nến đang chọn.")
                st.dataframe(chart.tail(30).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Không tải được biểu đồ {selected}: {exc}")

        stock_symbols = [symbol for symbol in chart_symbols if symbol not in COMPARE_INDICES]
        if stock_symbols:
            st.subheader("Đối chiếu các mã cổ phiếu đã nhập")
            comparison = result[result["Mã"].astype(str).str.upper().isin(stock_symbols)].copy()
            if comparison.empty:
                st.warning("Chưa có dữ liệu quét cho các mã vừa nhập. Kiểm tra lại mã hoặc cập nhật dữ liệu thị trường.")
            else:
                order = {symbol: index for index, symbol in enumerate(stock_symbols)}
                comparison["_order"] = comparison["Mã"].map(order)
                comparison = comparison.sort_values("_order").drop(columns=["_order"])
                st.dataframe(
                    comparison[["Mã", "Sàn", "Giá đóng cửa", "ROC14 (%)", "EMA100", "Khối lượng", "KLGD TB20", "Ngày dữ liệu"]],
                    use_container_width=True,
                    hide_index=True,
                )
            missing_symbols = [symbol for symbol in stock_symbols if symbol not in set(comparison["Mã"].astype(str).str.upper())]
            if missing_symbols:
                st.caption("Chưa có dữ liệu đối chiếu cho: " + ", ".join(missing_symbols))

    with tab_compare:
        st.subheader("So sánh hai cổ phiếu · Nến Nhật")
        # Searchable symbol selectors: Streamlit filters options while typing,
        # including on mobile. Keep the market index available as an option.
        col_a, col_b = st.columns(2)
        with col_a:
            ticker_a = st.selectbox(
                "Mã khung trên", symbol_options,
                index=symbol_options.index("FPT") if "FPT" in symbol_options else 0,
                key="compare_ticker_a_search",
                help="Chạm vào ô và gõ vài chữ đầu của mã để lọc danh sách.",
            )
        with col_b:
            ticker_b = st.selectbox(
                "Mã khung dưới", symbol_options,
                index=symbol_options.index("HPG") if "HPG" in symbol_options else 0,
                key="compare_ticker_b_search",
                help="Chạm vào ô và gõ vài chữ đầu của mã để lọc danh sách.",
            )
        compare_timeframe = st.radio(
            "Khung nến", options=["1D", "2D", "3D", "1W", "1M"],
            format_func=lambda x: {"1D": "Ngày", "2D": "2 ngày", "3D": "3 ngày",
                                   "1W": "Tuần", "1M": "Tháng"}[x],
            horizontal=True, key="compare_timeframe",
        )
        compare_days = st.select_slider(
            "Khoảng thời gian", options=[90, 180, 365, 540, 730, *LONG_HISTORY_DAYS], value=365,
            format_func=history_label, key="compare_days",
        )
        if not ticker_a or not ticker_b or ticker_a == ticker_b:
            st.info("Nhập hai mã cổ phiếu khác nhau.")
        elif not all(t in COMPARE_INDICES or (3 <= len(t) <= 6 and t.isalnum()) for t in (ticker_a, ticker_b)):
            st.warning("Mỗi mã phải có 3–6 ký tự chữ hoặc số.")
        else:
            try:
                today = vietnam_market_date()
                start_date = (today - timedelta(days=compare_days)).isoformat()
                end_date = today.isoformat()
                charts = []
                sources = []
                from stock_compare import aggregate_candles, render_stock_comparison
                for ticker in (ticker_a, ticker_b):
                    if ticker in COMPARE_INDICES:
                        clean, source = fetch_index_chart_quotes(COMPARE_INDICES[ticker], start_date, end_date)
                        clean = drop_market_holiday_bars(clean)
                        warning = None
                    else:
                        raw, source = fetch_chart_quotes(ticker, start_date, end_date, interval="1D")
                        clean, warning = sanitize_daily_chart_bars(raw)
                    if warning:
                        st.warning(f"{ticker}: {warning}")
                    if clean.empty:
                        raise MarketDataError(f"{ticker} không có dữ liệu nến D1 hợp lệ")
                    candles = aggregate_candles(clean, compare_timeframe)
                    charts.append(add_indicators(candles))
                    sources.append(source)
                render_stock_comparison(charts[0], ticker_a, charts[1], ticker_b)
                for ticker, candles in zip((ticker_a, ticker_b), charts):
                    dates = candles["date"]
                    st.caption(f"{ticker}: {dates.min():%d/%m/%Y} đến {dates.max():%d/%m/%Y} · {len(candles):,} nến.")
                overlap = pd.merge(
                    charts[0][["date", "close"]].rename(columns={"close": "first"}),
                    charts[1][["date", "close"]].rename(columns={"close": "second"}),
                    on="date", how="inner",
                ).sort_values("date")
                if len(overlap) >= 2:
                    change_a = (overlap["first"].iloc[-1] / overlap["first"].iloc[0] - 1) * 100
                    change_b = (overlap["second"].iloc[-1] / overlap["second"].iloc[0] - 1) * 100
                    a, b, diff = st.columns(3)
                    a.metric(ticker_a, f"{change_a:+.2f}%")
                    b.metric(ticker_b, f"{change_b:+.2f}%")
                    diff.metric("Chênh lệch", f"{change_a-change_b:+.2f} điểm %")
                    st.caption(f"Biến động giá đóng cửa trên {len(overlap)} phiên chung; chưa điều chỉnh cổ tức.")
                else:
                    st.info("Không đủ hai phiên giao dịch chung để tính biến động.")
                st.caption(f"Khung {compare_timeframe}: 2D/3D tính theo phiên giao dịch; 1W/1M theo tuần/tháng. Nến đang hình thành có thể chưa đủ phiên.")
                st.caption(f"Nguồn: {ticker_a} — {sources[0]}; {ticker_b} — {sources[1]}.")
            except Exception as exc:
                st.error(f"Không tải được so sánh: {exc}")

    with tab_all:
        st.subheader("Bảng dữ liệu toàn thị trường")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Tải toàn bộ kết quả quét CSV", result.to_csv(index=False).encode("utf-8-sig"),
            "ket_qua_quet_thi_truong.csv", "text/csv"
        )
        if stats and stats.get("errors"):
            with st.expander(f"Lỗi ở {len(stats['errors'])} mã"):
                st.code("\n".join(stats["errors"][:100]))

    with tab_all:
        from cophieu68_import import render_cophieu68_import
        render_cophieu68_import()

