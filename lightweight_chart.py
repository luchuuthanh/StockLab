"""Mobile-friendly TradingView Lightweight Charts embed for StockLab."""
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def calculate_stochastic_trial(chart: pd.DataFrame) -> pd.DataFrame:
    """Calculate the trial oscillator from chronological OHLC bars."""
    high = chart["high"].rolling(14).max()
    low = chart["low"].rolling(14).min()
    spread = high - low
    raw = (100 * (chart["close"] - low) / spread.where(spread > 0)).clip(0, 100)
    green = raw.ewm(span=3, adjust=False).mean().where(raw.notna())
    yellow = green.ewm(span=3, adjust=False).mean().where(green.notna())
    red = yellow.ewm(span=9, adjust=False).mean().where(yellow.notna())
    return pd.DataFrame({
        "%K thô": raw,
        "EMA 3": green,
        "EMA 3 lần hai": yellow,
        "EMA 9": red,
        "Biên trên": pd.Series(80.0, index=chart.index),
        "Biên dưới": pd.Series(20.0, index=chart.index),
    })


def stochastic_trade_markers(stochastic: pd.DataFrame, dates, *, buy_threshold=25) -> list[dict[str, str]]:
    """Mark crossings of the oscillator candle close and the red EMA 9 line."""
    markers, _ = stochastic_trade_events(stochastic, dates, buy_threshold=buy_threshold)
    return markers


def stochastic_trade_events(stochastic, dates, *, after="", holding=False, ready_to_buy=False, buy_threshold=25):
    """Replay the chart's signal rules, optionally resuming a previous scan."""
    closes = stochastic["EMA 3"]
    red = stochastic["EMA 9"]
    markers = []
    for day, previous, current, previous_red, current_red in zip(
        dates, closes.shift(), closes, red.shift(), red,
    ):
        signal_date = day.strftime("%Y-%m-%d")
        if signal_date <= after or pd.isna(current) or pd.isna(current_red):
            continue
        if not holding and current < buy_threshold:
            ready_to_buy = True
        if pd.isna(previous) or pd.isna(previous_red):
            continue
        if holding and previous >= previous_red and current < current_red and current < previous:
            markers.append({"time": signal_date, "position": "aboveBar",
                            "color": "#ff6068", "shape": "arrowDown", "text": "Bán"})
            holding = False
            ready_to_buy = current < buy_threshold
        elif not holding and ready_to_buy and previous <= previous_red and current > current_red and current > previous:
            markers.append({"time": signal_date, "position": "belowBar",
                            "color": "#32d981", "shape": "arrowUp", "text": "Mua"})
            holding = True
            ready_to_buy = False
    return markers, ready_to_buy


def ichimoku_spans(bars):
    hi9 = bars["high"].rolling(9, min_periods=9).max()
    lo9 = bars["low"].rolling(9, min_periods=9).min()
    hi26 = bars["high"].rolling(26, min_periods=26).max()
    lo26 = bars["low"].rolling(26, min_periods=26).min()
    hi52 = bars["high"].rolling(52, min_periods=52).max()
    lo52 = bars["low"].rolling(52, min_periods=52).min()
    tenkan = (hi9 + lo9) / 2
    kijun = (hi26 + lo26) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (hi52 + lo52) / 2
    return tenkan, kijun, span_a, span_b


def rejected_heikin_ashi_dates(bars, buy_dates):
    """Apply the same green-body and wick rules as the chart."""
    previous = None
    rejected = set()
    for bar in bars.itertuples(index=False):
        close = (bar.open + bar.high + bar.low + bar.close) / 4
        opening = sum(previous) / 2 if previous else (bar.open + bar.close) / 2
        previous = opening, close
        if bar.date.strftime("%Y-%m-%d") not in buy_dates or close <= opening:
            continue
        high, low = max(bar.high, opening, close), min(bar.low, opening, close)
        body, spread = close - opening, high - low
        tolerance = 4 * 2**-52 * max(abs(high), abs(low))
        long_body = body + tolerance >= 0.01 * opening and body + tolerance >= 0.50 * spread
        if body <= 0.10 * spread + tolerance or (
            not long_body and max(high - close, opening - low) > 0.35 * body + tolerance
        ):
            rejected.add(bar.date.strftime("%Y-%m-%d"))
    return rejected


def signal_set_four_events(bars, stochastic, *, after="", trade="", ready=False, spans=None):
    """Return Set 4 markers plus the oscillator and breakout state for the next scan."""
    raw_events, ready = stochastic_trade_events(
        stochastic, bars["date"], after=after, holding=bool(trade),
        ready_to_buy=ready, buy_threshold=32,
    )
    by_date = {event["time"]: event for event in raw_events}
    rejected = rejected_heikin_ashi_dates(
        bars, {day for day, event in by_date.items() if event["text"] == "Mua"},
    )
    _, _, span_a, span_b = ichimoku_spans(bars) if spans is None else spans
    events = []
    for date, price, a, b in zip(
        bars["date"], bars["close"], span_a.shift(26).round(6), span_b.shift(26).round(6),
    ):
        day = date.strftime("%Y-%m-%d")
        if day <= after:
            continue
        event = by_date.get(day)
        has_cloud = pd.notna(a) and pd.notna(b)
        if trade == "pending":
            trade = "fake"
            if event is None and has_cloud and price > max(a, b):
                trade = "vm"
                events.append({"time": day, "position": "belowBar", "color": "#32d981",
                               "shape": "arrowUp", "text": "Mua VM", "price": float(price)})
        if event is None:
            continue
        if event["text"] == "Mua":
            if has_cloud and min(a, b) <= price <= max(a, b):
                trade = "pending"
            else:
                trade = "fake" if day in rejected else "normal"
            text = "Mua" if trade == "normal" else "Mua giả"
        else:
            text = {"vm": "Bán VM", "fake": "Bán giả"}.get(trade, "Bán")
            trade = ""
        events.append({**event, "text": text, "price": float(price)})
    return events, ready, trade


def early_trade_markers(chart, stochastic, set_four_events):
    """Wait for a Set 4 entry after an eligible early buy crossing."""
    set_four_entries = {
        event["time"]: event["text"] for event in set_four_events
        if event["text"] in {"Mua", "Mua giả", "Mua VM"}
    }
    candle = stochastic["EMA 3"]
    fast, slow = stochastic["EMA 4"], stochastic["EMA 8"]
    ha_close = chart[["open", "high", "low", "close"]].mean(axis=1)
    ha_confirmed = ha_close.gt(ha_close.shift()) & ha_close.gt(ha_close.shift(2))
    ema100 = chart["EMA100"]
    gap = (100 * (ema100 - chart["close"]) / ema100).round(10)
    buy_cross = candle.gt(slow) & candle.shift().le(slow.shift())
    sell_cross = fast.lt(slow) & fast.shift().ge(slow.shift())
    below_both = candle.lt(fast) & candle.lt(slow)
    markers = []
    holding = ""
    pending = False
    bearish_cross = False
    warned = False
    watching = False
    watch_crossed = False
    for day, buy, cross, below, bearish, ha_ok, distance in zip(
        chart["date"], buy_cross, sell_cross, below_both, fast.lt(slow), ha_confirmed, gap,
    ):
        time = day.strftime("%Y-%m-%d")
        in_watch_band = 5 <= distance <= 8.5
        if watching:
            watch_crossed = bearish and (watch_crossed or cross)
            if below and watch_crossed:
                markers.append({"time": time, "position": "aboveBar", "color": "#f6c453",
                                "shape": "circle", "size": 0, "text": "◆ Bán theo dõi"})
                watching = False
        elif buy and ha_ok and in_watch_band:
            markers.append({"time": time, "position": "belowBar", "color": "#f6c453",
                            "shape": "circle", "size": 0, "text": "◆ Mua theo dõi"})
            watching = True
            watch_crossed = False

        if holding:
            bearish_cross = bearish and (bearish_cross or cross)
            if below and bearish_cross:
                label = {"Mua giả": "Bán giả", "Mua VM": "Bán VM"}.get(holding, "Bán")
                markers.append({"time": time, "position": "aboveBar", "color": "#ff646a",
                                "shape": "arrowDown", "text": label})
                holding = ""
                bearish_cross = False
                warned = False
            elif below:
                if not warned:
                    markers.append({"time": time, "position": "aboveBar", "color": "#ff646a",
                                    "shape": "circle", "size": 0, "text": "✕"})
                warned = True
            else:
                warned = False
            continue
        if pending:
            bearish_cross = bearish and (bearish_cross or cross)
            if below and bearish_cross:
                pending = False
                bearish_cross = False
                continue
        elif buy and ha_ok and pd.notna(distance) and not in_watch_band:
            if 0 < distance < 5:
                holding = "Mua giả"
                markers.append({"time": time, "position": "belowBar", "color": "#32d981",
                                "shape": "arrowUp", "text": holding})
                continue
            pending = True
            markers.append({"time": time, "position": "belowBar", "color": "#f6c453",
                            "shape": "circle", "size": 0, "text": "Chờ mua"})

        if pending:
            confirmation = set_four_entries.get(time)
            if confirmation == "Mua giả":
                markers.append({"time": time, "position": "belowBar", "color": "#f6c453",
                                "shape": "circle", "size": 0, "text": "Mua giả"})
            elif confirmation in {"Mua", "Mua VM"}:
                holding = confirmation
                pending = False
                bearish_cross = False
                markers.append({"time": time, "position": "belowBar", "color": "#32d981",
                                "shape": "arrowUp", "text": holding})
    return markers


def render_lightweight_chart(chart: pd.DataFrame, symbol: str, timeframe: str = "1D") -> None:
    """Render interactive candles with touch navigation, volume and selectable overlays."""
    # Standard Ichimoku (9, 26, 52), with 26-period projection.
    ordered = chart.sort_values("date").drop_duplicates("date", keep="last").copy()
    tenkan, kijun, span_a, span_b = ichimoku_spans(ordered)
    dates = list(ordered["date"])
    projection_step = {
        "1D": pd.offsets.BusinessDay(),
        "2D": pd.offsets.BusinessDay(2),
        "3D": pd.offsets.BusinessDay(3),
        "1W": pd.offsets.Week(weekday=4),
        "1M": pd.offsets.BMonthEnd(),
    }[timeframe]
    future = pd.date_range(start=dates[-1] + projection_step, periods=26, freq=projection_step)
    projected_dates = dates + list(future)
    def ichipoints(values, shift=0):
        return [{"time": projected_dates[i + shift].strftime("%Y-%m-%d"), "value": round(float(v), 6)}
                for i, v in enumerate(values) if pd.notna(v) and 0 <= i + shift < len(projected_dates)]
    ichimoku = {"Tenkan": ichipoints(tenkan), "Kijun": ichipoints(kijun),
                "SenkouA": ichipoints(span_a, 26), "SenkouB": ichipoints(span_b, 26),
                "Chikou": ichipoints(ordered["close"], -26)}
    close_window = ordered["close"].rolling(20)
    bb_middle = close_window.mean()
    bb_width = 2 * close_window.std(ddof=0)
    bollinger = {
        "upper": ichipoints(bb_middle + bb_width),
        "middle": ichipoints(bb_middle),
        "lower": ichipoints(bb_middle - bb_width),
    }
    chart = ordered.copy()
    def points(column):
        return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 6)}
                for d, v in zip(chart["date"], chart[column]) if pd.notna(v)]
    candles = [{"time": row.date.strftime("%Y-%m-%d"),
                "open": float(row.open), "high": float(row.high),
                "low": float(row.low), "close": float(row.close)}
               for row in chart.itertuples(index=False)]
    volume = [{"time": row.date.strftime("%Y-%m-%d"), "value": float(row.volume),
               "color": "#26a69a88" if row.close >= row.open else "#ef535088"}
              for row in chart.itertuples(index=False) if pd.notna(row.volume)]
    stochastic = calculate_stochastic_trial(chart)
    # Signal-pane averages of the oscillator candle close.
    stochastic["EMA 4"] = stochastic["EMA 3"].ewm(span=4, adjust=False).mean().where(stochastic["EMA 3"].notna())
    stochastic["EMA 8"] = stochastic["EMA 3"].ewm(span=8, adjust=False).mean().where(stochastic["EMA 3"].notna())
    stochastic_bars = []
    green = stochastic["EMA 3"]
    for day, previous, current in zip(dates, green.shift(), green):
        bar = {"time": day.strftime("%Y-%m-%d")}
        if pd.notna(previous) and pd.notna(current):
            bar.update(open=round(float(previous), 6), close=round(float(current), 6),
                       high=round(float(max(previous, current)), 6),
                       low=round(float(min(previous, current)), 6))
        stochastic_bars.append(bar)
    volume_sma10 = ichipoints(chart["volume"].rolling(10).mean())
    set_four_events, _, _ = signal_set_four_events(
        chart, stochastic, spans=(tenkan, kijun, span_a, span_b),
    )
    payload = json.dumps({"candles": candles, "volume": volume, "volumeSma10": volume_sma10,
                          "overlays": {k: points(k) for k in ("EMA7", "SMA20", "EMA20", "EMA50", "EMA100")},
                          "rsi": points("RSI14"), "macd": points("MACD"), "mom": points("MOM10"),
                          "signal": points("MACD_SIGNAL"), "hist": points("MACD_HIST"),
                          "symbol": symbol, "barUnit": "phiên" if timeframe == "1D" else "nến",
                          "ichimoku": ichimoku, "bollinger": bollinger,
                          "timeline": [day.strftime("%Y-%m-%d") for day in projected_dates],
                          "stochastic": {name: ichipoints(values) for name, values in stochastic.items()},
                          "stochasticBars": stochastic_bars,
                          "earlyMarkers": early_trade_markers(chart, stochastic, set_four_events),
                          "stochasticMarkers4": set_four_events,
                          "stochasticMarkers32": stochastic_trade_markers(stochastic, dates, buy_threshold=32),
                          "stochasticMarkers": stochastic_trade_markers(stochastic, dates)}, ensure_ascii=True).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body{margin:0;background:#101722;color:#e4eaf2;font:13px system-ui,sans-serif}
#toolbar{display:flex;align-items:center;gap:6px;overflow-x:auto;padding:8px;white-space:nowrap}
.indicator-header{display:flex;align-items:center;justify-content:space-between;gap:8px;height:36px;padding:0 8px;background:#172233}
.indicator-header h2{margin:0;font-size:13px}
#candle-focus{padding:5px 9px}
#error:empty{display:none}
button{background:#263446;color:#e4eaf2;border:1px solid #415168;border-radius:7px;padding:9px 12px}
button.active{background:#186b7c;border-color:#32c7c5}#status{padding:4px 10px;min-height:18px;color:#b5c7d8}
#error{padding:12px;color:#f6bd60}
#price,#volume,.indicator-chart{position:relative}
.session-line,.session-values{position:absolute;z-index:2;pointer-events:none}
.session-line{top:0;bottom:0;border-left:1px dashed #d7e0ed}
.has-time-axis .session-line{bottom:26px}
.session-values{top:4px;left:6px;max-width:calc(100% - 90px);padding:4px 6px;border-radius:4px;background:#101722ee;color:#edf3fb;line-height:1.35;font-size:12px}
@media(max-width:600px){
    .indicator-header{height:30px}
    #candle-focus{padding:4px 8px}
}
#trade-history{box-sizing:border-box;height:355px;padding:10px;overflow:hidden}
#trade-history h2{margin:0 0 8px;font-size:15px}
#trade-history select{color:inherit;background:#263446;border:1px solid #415168;border-radius:5px;padding:4px}
#trade-history p{margin:7px 0;font-size:12px}
#trade-rows-scroll{height:205px;overflow:auto}
#trade-history table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}
#trade-history th,#trade-history td{padding:6px 8px;text-align:left;border-bottom:1px solid #35465a}
#trade-history th{position:sticky;top:0;background:#263446}
</style></head><body><div id="toolbar"></div><div id="status">Chạm vào nến để xem giá · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng</div>
<div id="price"></div><div id="volume"></div>
<section id="indicator-1" aria-labelledby="indicator-title-1"><div class="indicator-header"><h2 id="indicator-title-1"></h2><button id="candle-focus" type="button" aria-pressed="false" hidden>Chỉ xem nến</button></div><div class="indicator-chart" id="osc"></div></section>
<section id="indicator-2" aria-labelledby="indicator-title-2" hidden><div class="indicator-header"><h2 id="indicator-title-2"></h2></div><div class="indicator-chart" id="osc-2"></div></section>
<section id="trade-history" aria-labelledby="trade-history-title">
<h2 id="trade-history-title">Phân tích tín hiệu mua bán</h2>
<label>Bộ chỉ báo <select id="trade-set"><option value="0">Bộ 1</option><option value="1">Bộ 2</option><option value="2">Bộ 3</option><option value="3" selected>Bộ 4</option><option value="4">Mua bán sớm</option></select></label>
<label>Loại <select id="trade-type"><option>Tất cả</option><option>Thường</option><option>Giả</option><option>VM</option></select></label>
<p>Giá đóng cửa tại tín hiệu. Lợi nhuận chưa tính phí. <strong id="trade-total" aria-live="polite"></strong></p>
<div id="trade-rows-scroll" tabindex="0"><table><thead><tr><th>Loại</th><th>Ngày mua</th><th>Giá mua</th><th>Ngày bán</th><th>Giá bán</th><th>% lợi nhuận</th><th>Trạng thái</th></tr></thead><tbody id="trade-rows"></tbody></table></div>
</section>
<div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' '+data.barUnit+' · Đang khởi tạo biểu đồ';
if(!window.LightweightCharts){document.getElementById('error').textContent='Lỗi tải thư viện biểu đồ từ CDN. Vui lòng kiểm tra kết nối mạng.';}
else {try {
const L=window.LightweightCharts;
const compactScreen=window.matchMedia('(max-width: 600px)');
const chartFrame=window.frameElement;
let signalExtraHeight=0;
const baseAxisWidth=84;
function chartSizes(){
    if(!compactScreen.matches)return {price:410,volume:90,indicator:220,header:36};
    const header=30;
    const controlsHeight=document.getElementById('toolbar').offsetHeight
        +document.getElementById('status').offsetHeight;
    const frameHeight=chartFrame?chartFrame.clientHeight-signalExtraHeight-355:window.innerHeight-355;
    const plotHeight=Math.max(390,frameHeight-controlsHeight-2*header-8);
    const price=Math.round(plotHeight*0.44);
    const volume=Math.round(plotHeight*0.11);
    return {price,volume,indicator:Math.floor((plotHeight-price-volume)/2),header};
}
const base={crosshair:{vertLine:{visible:false,labelVisible:false},horzLine:{visible:true,labelVisible:true}},layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},rightPriceScale:{borderColor:'#35465a',minimumWidth:baseAxisWidth},timeScale:{borderColor:'#35465a',timeVisible:false},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const price=L.createChart(document.getElementById('price'),{...base,width:Math.max(320,document.getElementById('price').clientWidth),height:chartSizes().price});
const candleStyle={upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderVisible:false};
const candle=price.addCandlestickSeries(candleStyle);
candle.setData(data.candles);
const averageColors={EMA7:'#f8d05a',SMA20:'#37c3ff',EMA20:'#b48bff',EMA50:'#f59e73',EMA100:'#e5e7eb'};
const ichimokuColors={Tenkan:'#ff9d43',Kijun:'#e95bb5',SenkouA:'#29bd88',SenkouB:'#f87171',Chikou:'#9acaff'};
const ichimokuLabel='Ichimoku (9,26,52)';
const bandLabels={upper:'BB trên',middle:'BB giữa',lower:'BB dưới'};
const enabledPriceOverlays=new Set(['EMA7','SMA20','EMA100']);
function createPriceOverlays(chart){
    function addLine(name,points,options){
        const series=chart.addLineSeries({lineWidth:2,priceLineVisible:false,
            lastValueVisible:false,visible:false,...options});
        series.setData(points);
        return [name,series];
    }
    return {
        BB:Object.entries(data.bollinger).map(([band,points])=>{
            const middle=band==='middle';
            return addLine(bandLabels[band],points,{color:middle?'#8eb9f3':'#678ff2',
                lineWidth:middle?1:2,lineStyle:middle?2:0});
        }),
        [ichimokuLabel]:Object.entries(data.ichimoku).map(([name,points])=>
            addLine(name,points,{color:ichimokuColors[name]})),
        ...Object.fromEntries(Object.entries(data.overlays).map(([name,points])=>[
            name,[addLine(name,points,{color:averageColors[name],lastValueVisible:true})]
        ]))
    };
}
const priceOverlays=createPriceOverlays(price);
const vol=L.createChart(document.getElementById('volume'),{...base,width:Math.max(320,document.getElementById('volume').clientWidth),height:chartSizes().volume,timeScale:{...base.timeScale,visible:false}});
vol.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'right',priceLineVisible:false,lastValueVisible:false}).setData(data.volume);
const volumeAverage=vol.addLineSeries({color:'#f8d05a',lineWidth:2,priceFormat:{type:'volume'},
    priceLineVisible:false,lastValueVisible:false});
volumeAverage.setData(data.volumeSma10);
const stochasticMode='Bộ chỉ báo mua bán';
const filteredStochasticMode='Bộ tín hiệu mua bán 2';
const heikinAshiMode='Nến Heikin Ashi';
const thirdStochasticMode='Bộ tín hiệu mua bán 3';
const fourthStochasticMode='Bộ tín hiệu mua bán 4';
const earlyTradeMode='Bộ chỉ số mua bán sớm';
const tallIndicatorModes=new Set([stochasticMode,filteredStochasticMode,thirdStochasticMode,fourthStochasticMode,earlyTradeMode,heikinAshiMode]);
function calculateHeikinAshi(candles){
    let previous;
    return candles.map(bar=>{
        const close=(bar.open+bar.high+bar.low+bar.close)/4;
        const open=previous?(previous.open+previous.close)/2:(bar.open+bar.close)/2;
        const high=Math.max(bar.high,open,close);
        const low=Math.min(bar.low,open,close);
        previous={time:bar.time,open,high,low,close};
        return previous;
    });
}
const heikinAshiBars=calculateHeikinAshi(data.candles);
const indexByDate=points=>new Map(points.map(point=>[point.time,point]));
const heikinAshiByDate=indexByDate(heikinAshiBars);
function isRejectedGreenHeikinAshi(bar){
    if(!bar||bar.close<=bar.open)return false;
    const body=bar.close-bar.open;
    const range=bar.high-bar.low;
    const tolerance=4*Number.EPSILON*Math.max(Math.abs(bar.high),Math.abs(bar.low));
    if(body<=0.10*range+tolerance)return true;
    const longBody=body+tolerance>=0.01*bar.open&&body+tolerance>=0.50*range;
    if(longBody)return false;
    const upperWick=bar.high-bar.close;
    const lowerWick=bar.open-bar.low;
    return Math.max(upperWick,lowerWick)>0.35*body+tolerance;
}
function classifyStochasticMarkers(markers,barsByDate){
    let fakeBuy=false;
    return markers.map(marker=>{
        const isBuy=marker.shape==='arrowUp';
        if(isBuy)fakeBuy=isRejectedGreenHeikinAshi(barsByDate.get(marker.time));
        const classified=fakeBuy?{...marker,text:isBuy?'Mua giả':'Bán giả'}:marker;
        if(!isBuy)fakeBuy=false;
        return classified;
    });
}
const thirdStochasticMarkers=classifyStochasticMarkers(data.stochasticMarkers32,heikinAshiByDate);
const vmPriceFormat=new Intl.NumberFormat('vi-VN',{minimumFractionDigits:2,maximumFractionDigits:2});
const signalMarkers=new Map([
    [stochasticMode,data.stochasticMarkers],
    [filteredStochasticMode,classifyStochasticMarkers(data.stochasticMarkers,heikinAshiByDate)],
    [thirdStochasticMode,thirdStochasticMarkers],
    [fourthStochasticMode,data.stochasticMarkers4.map(({price,...marker})=>
        marker.text.endsWith(' VM')?{...marker,text:marker.text+' · '+vmPriceFormat.format(price)}:marker)],
    [earlyTradeMode,data.earlyMarkers]
]);
const tradeSet=document.getElementById('trade-set');
const tradeType=document.getElementById('trade-type');
const closingPrices=new Map(data.candles.map(bar=>[bar.time,bar.close]));
function tradeHistory(events){
    const rows=[];
    let active=null;
    for(const event of events){
        const label=event.text.split(' · ')[0];
        const price=closingPrices.get(event.time);
        if(event.shape==='arrowUp'){
            if(active&&label==='Mua VM')active.status='Chuyển sang Mua VM';
            active={type:label==='Mua giả'?'Giả':label==='Mua VM'?'VM':'Thường',
                start:event.time,entry:price,status:'Chưa bán'};
            rows.push(active);
        } else if(active&&event.shape==='arrowDown'){
            active.end=event.time;
            active.exit=price;
            active.status=label;
            active=null;
        }
    }
    return rows.reverse();
}
const tradeHistories=Array.from(signalMarkers.values(),tradeHistory);
const tradeDate=day=>day?day.split('-').reverse().join('/'):'—';
const tradeNumber=value=>Number.isFinite(value)?vmPriceFormat.format(value):'—';
function renderTradeHistory(){
    const body=document.getElementById('trade-rows');
    body.replaceChildren();
    const rows=tradeHistories[Number(tradeSet.value)].filter(row=>
        tradeType.value==='Tất cả'||row.type===tradeType.value);
    if(!rows.length){
        const cell=body.insertRow().insertCell();
        cell.colSpan=7;
        cell.textContent='Không có tín hiệu phù hợp trong lịch sử đang xem.';
    }
    let total=0;
    let closedCount=0;
    const bySale=new Map([['Bán',[0,0]],['Bán giả',[0,0]],['Bán VM',[0,0]]]);
    for(const trade of rows){
        const row=body.insertRow();
        const result=trade.entry>0&&Number.isFinite(trade.exit)?100*(trade.exit/trade.entry-1):null;
        if(result!==null){
            total+=result;
            closedCount++;
            const subtotal=bySale.get(trade.status);
            if(subtotal){
                subtotal[0]+=result;
                subtotal[1]++;
            }
        }
        const values=[trade.type,tradeDate(trade.start),tradeNumber(trade.entry),tradeDate(trade.end),
            tradeNumber(trade.exit),result===null?'—':tradeNumber(result)+'%',trade.status];
        values.forEach(value=>{row.insertCell().textContent=value;});
    }
    const summaries=[closedCount
        ? 'Tổng % lợi nhuận: '+tradeNumber(total)+'% ('+closedCount+' cặp đã bán)'
        : 'Tổng % lợi nhuận: — (chưa có cặp đã bán)'];
    for(const [label,[sum,count]] of bySale){
        const name=label==='Bán'?'Bán thường':label;
        summaries.push(name+': '+(count?tradeNumber(sum)+'%':'—'));
    }
    document.getElementById('trade-total').textContent=summaries.join(' · ');
}
tradeSet.addEventListener('change',renderTradeHistory);
tradeType.addEventListener('change',renderTradeHistory);
renderTradeHistory();
const candleFocus=document.getElementById('candle-focus');
function createIndicatorPane(id, sectionId, titleId){
    const root=document.getElementById(id);
    const title=document.getElementById(titleId);
    const section=document.getElementById(sectionId);
    const osc=L.createChart(root,{...base,width:Math.max(320,root.clientWidth),height:chartSizes().indicator});
    const rsi=osc.addLineSeries({color:'#dba5ff',lineWidth:2,priceLineVisible:false});rsi.setData(data.rsi);
    const rsi30=osc.addLineSeries({color:'#f59e0b',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false});
    const rsi70=osc.addLineSeries({color:'#f59e0b',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false});
    rsi30.setData(data.rsi.map(p=>({time:p.time,value:30})));
    rsi70.setData(data.rsi.map(p=>({time:p.time,value:70})));
    const macd=osc.addLineSeries({color:'#36c6e9',lineWidth:2,priceLineVisible:false});macd.setData(data.macd);
    const signal=osc.addLineSeries({color:'#f8d05a',lineWidth:2,priceLineVisible:false});signal.setData(data.signal);
    const hist=osc.addHistogramSeries({color:'#5aa59c',priceLineVisible:false});hist.setData(data.hist);
    const stochasticBars=osc.addCandlestickSeries({
        upColor:'#32d981',downColor:'#ff6068',wickVisible:false,borderVisible:false,
        priceLineVisible:false,lastValueVisible:false,visible:false
    });
    stochasticBars.setData(data.stochasticBars);
    const stochasticColorMap={'%K thô':'#f3f5fa','EMA 3':'#48da87','EMA 3 lần hai':'#fce06a','EMA 9':'#ff646a','EMA 4':'#00e5ff','EMA 8':'#ff6ec7','Biên trên':'#ec91ad','Biên dưới':'#69aaff'};
    const stochasticLines=Object.entries(data.stochastic).map(([name,points],i)=>{
        const series=osc.addLineSeries({
            color:stochasticColorMap[name]||"#f3f5fa",lineWidth:1,lineStyle:name.startsWith("Biên")?2:0,
            priceLineVisible:false,lastValueVisible:false,visible:false
        });
        series.setData(points);
        return [name,series];
    });
    const mom=osc.addLineSeries({color:'#69c6f0',lineWidth:2,priceLineVisible:false,visible:false});
    mom.setData(data.mom);
    const heikinAshi=osc.addCandlestickSeries({...candleStyle,priceLineVisible:false,visible:false});
    heikinAshi.setData(heikinAshiBars);
    const modes={RSI:[rsi,rsi30,rsi70],MACD:[macd,signal,hist],MOM:[mom],
        [stochasticMode]:[stochasticBars,...stochasticLines.map(([,series])=>series)],
        [heikinAshiMode]:[heikinAshi]};
    const candleFocusSeries=new Set([stochasticBars,...stochasticLines
        .filter(([name])=>name==='Biên trên'||name==='Biên dưới').map(([,series])=>series)]);
    const display={RSI:[['RSI',rsi]],MACD:[['MACD',macd],['Signal',signal],['Histogram',hist]],
        MOM:[['MOM',mom]],[stochasticMode]:stochasticLines};
    return {id,root,title,section,chart:osc,modes,display,stochasticLines,stochasticBars,candleFocusSeries,
        priceOverlays:createPriceOverlays(osc)};
}
const indicatorPanes=[
    createIndicatorPane('osc','indicator-1','indicator-title-1'),
    createIndicatorPane('osc-2','indicator-2','indicator-title-2')
];
const selectedModes=['RSI','MACD'];
const signalLineNames=['%K thô','EMA 3','EMA 3 lần hai','EMA 9','EMA 4','EMA 8'];
const enabledSignalLines=new Set(['%K thô','EMA 3','EMA 3 lần hai','EMA 9']);
const earlySignalLines=new Set(['EMA 4','EMA 8']);
function syncSignalLineControls(pane,mode){
    const active=signalMarkers.has(mode);
    if(!pane.signalControls){
        const controls=document.createElement('div');
        controls.className='signal-line-controls';
        controls.style.cssText='display:flex;flex-wrap:wrap;gap:4px;padding:4px 6px';
        for(const name of signalLineNames){
            const button=document.createElement('button');
            button.type='button';
            button.textContent=name;
            button.onclick=()=>{
                const lines=pane.enabledSignalLines;
                if(lines.has(name))lines.delete(name);
                else lines.add(name);
                renderIndicators();
            };
            controls.appendChild(button);
        }
        pane.section.insertBefore(controls,pane.root);
        pane.signalControls=controls;
    }
    pane.signalControls.hidden=!active;
    // Inline display:flex overrides the HTML hidden attribute on some browsers.
    pane.signalControls.style.display=active?'flex':'none';
    pane.enabledSignalLines=mode===earlyTradeMode?earlySignalLines:enabledSignalLines;
    for(const button of pane.signalControls.children){
        const on=pane.enabledSignalLines.has(button.textContent);
        button.classList.toggle('active',on);
        button.setAttribute('aria-pressed',String(on));
    }
}
const buttons=new Map();
function syncPriceOverlays(){
    function applyVisibility(groups,active){
        for(const [name,lines] of Object.entries(groups)){
            const visible=active&&enabledPriceOverlays.has(name);
            for(const [,series] of lines)series.applyOptions({visible});
        }
    }
    applyVisibility(priceOverlays,true);
    indicatorPanes.forEach((pane,i)=>{
        applyVisibility(pane.priceOverlays,selectedModes[i]===heikinAshiMode);
    });
}
function resizeChartPanes(){
    const {price:priceHeight,volume:volumeHeight,indicator,header}=chartSizes();
    const lastIndicator=selectedModes.length-1;
    const resize=(root,chart,height,showAxis)=>{
        root.classList.toggle('has-time-axis',showAxis);
        chart.applyOptions({width:root.clientWidth||320,height,timeScale:{visible:showAxis}});
    };
    resize(document.getElementById('price'),price,priceHeight,!compactScreen.matches);
    resize(document.getElementById('volume'),vol,volumeHeight,false);
    const tallCount=lastIndicator===1?selectedModes.filter(mode=>tallIndicatorModes.has(mode)).length:0;
    const extraHeight=chartFrame?indicator*tallCount:0;
    if(extraHeight!==signalExtraHeight){
        signalExtraHeight=extraHeight;
        chartFrame.style.setProperty('--stocklab-signal-extra', extraHeight+'px');
    }
    const indicatorHeight=chartFrame?indicator:Math.floor(indicator*2/(2+tallCount));
    indicatorPanes.slice(0,selectedModes.length).forEach((pane,i)=>{
        const height=lastIndicator===0?indicator*2+header:
            indicatorHeight*(tallIndicatorModes.has(selectedModes[i])?2:1);
        resize(pane.root,pane.chart,height,!compactScreen.matches||i===lastIndicator);
    });
}
function renderIndicators(){
    const range=price.timeScale().getVisibleLogicalRange();
    const candlesOnly=candleFocus.getAttribute('aria-pressed')==='true';
    const firstSignalPane=selectedModes.findIndex(name=>signalMarkers.has(name));
    rangeUpdateInProgress=true;
    try {
        indicatorPanes.forEach((pane,i)=>{
            const mode=selectedModes[i];
            pane.section.hidden=!mode;
            if(!mode)return;
            const stochasticActive=signalMarkers.has(mode);
            const seriesMode=stochasticActive?stochasticMode:mode;
            pane.title.textContent=mode;
            syncSignalLineControls(pane,mode);
            if(stochasticActive){
                if(i===firstSignalPane)pane.title.parentElement.appendChild(candleFocus);
                const markers=signalMarkers.get(mode);
                if(pane.stochasticMarkers!==markers){
                    pane.stochasticBars.setMarkers(markers);
                    pane.stochasticMarkers=markers;
                }
            }
            for(const [name,series] of Object.entries(pane.modes)){
                for(const item of series){
                    const lineName=stochasticActive?pane.stochasticLines.find(([,line])=>line===item)?.[0]:null;
                    const lineEnabled=!lineName||!signalLineNames.includes(lineName)||pane.enabledSignalLines.has(lineName);
                    item.applyOptions({visible:name===seriesMode&&lineEnabled&&(!stochasticActive||!candlesOnly||pane.candleFocusSeries.has(item))});
                }
            }
            pane.chart.priceScale('right').applyOptions({autoScale:true});
        });
        syncPriceOverlays();
        resizeChartPanes();
        if(range){
            for(const [,chart] of panes)chart.timeScale().setVisibleLogicalRange(range);
        }
    } finally {
        rangeUpdateInProgress=false;
    }
    candleFocus.hidden=firstSignalPane<0;
    for(const [name,button] of buttons){
        const selected=selectedModes.includes(name);
        button.classList.toggle('active',selected);
        button.setAttribute('aria-pressed',String(selected));
    }
}
function toggleMode(name){
    const index=selectedModes.indexOf(name);
    if(index!==-1){
        if(selectedModes.length===1)return;
        selectedModes.splice(index,1);
    } else if(selectedModes.length===2){
        selectedModes[1]=name;
    } else {
        selectedModes.push(name);
    }
    renderIndicators();
}
candleFocus.onclick=()=>{
    const active=candleFocus.getAttribute('aria-pressed')!=='true';
    candleFocus.setAttribute('aria-pressed',String(active));
    candleFocus.classList.toggle('active',active);
    candleFocus.textContent=active?'Hiện lại các đường':'Chỉ xem nến';
    renderIndicators();
    drawSession();
};
const toolbar=document.getElementById('toolbar');
const notesButton=document.createElement('button');
notesButton.textContent='Chú thích';
notesButton.classList.add('active');
notesButton.setAttribute('aria-pressed','true');
notesButton.onclick=()=>{
    const on=notesButton.getAttribute('aria-pressed')!=='true';
    notesButton.setAttribute('aria-pressed',String(on));
    notesButton.classList.toggle('active',on);
};
toolbar.appendChild(notesButton);
for(const name of Object.keys(priceOverlays)){
    const button=document.createElement('button');
    button.type='button';
    button.textContent=name;
    if(name==='BB')button.title='Bollinger Bands (20, 2)';
    const updateButton=()=>{
        const enabled=enabledPriceOverlays.has(name);
        button.classList.toggle('active',enabled);
        button.setAttribute('aria-pressed',String(enabled));
    };
    button.onclick=()=>{
        if(enabledPriceOverlays.has(name))enabledPriceOverlays.delete(name);
        else enabledPriceOverlays.add(name);
        updateButton();
        syncPriceOverlays();
    };
    updateButton();
    toolbar.appendChild(button);
}
const indicatorNames=Object.keys(indicatorPanes[0].modes).flatMap(name=>
    name===stochasticMode?[name,filteredStochasticMode,thirdStochasticMode,fourthStochasticMode,earlyTradeMode]:[name]);
for(const name of indicatorNames){
    const button=document.createElement('button');
    button.type='button';
    button.textContent=name;
    button.onclick=()=>toggleMode(name);
    buttons.set(name,button);
    toolbar.appendChild(button);
}
const panes=[['price',price],['volume',vol],...indicatorPanes.map(({id,chart})=>[id,chart])];
// Every pane needs identical logical indexes, including Ichimoku projection dates.
const timeline=data.timeline.map(time=>({time,value:0}));
for(const [,chart] of panes){
    chart.addLineSeries({visible:false,priceLineVisible:false,lastValueVisible:false}).setData(timeline);
}
const scales=panes.map(([,chart])=>chart.timeScale());
const pricesByDate=indexByDate(data.candles);
const volumeByDate=indexByDate(data.volume);
const volumeAverageByDate=indexByDate(data.volumeSma10);
const valueSeries=Object.fromEntries(Object.entries({
    ...data.overlays,...data.ichimoku,
    'BB trên':data.bollinger.upper,'BB giữa':data.bollinger.middle,'BB dưới':data.bollinger.lower,
    RSI:data.rsi,MACD:data.macd,Signal:data.signal,Histogram:data.hist,MOM:data.mom,...data.stochastic
}).map(([name,points])=>[name,indexByDate(points)]));
function overlayValues(overlays){
    const {BB,[ichimokuLabel]:ichimoku,...averages}=overlays;
    return visibleValues([...Object.values(averages).flat(),...ichimoku,...BB],2);
}
const readouts=panes.map(([id,chart])=>{
    const root=document.getElementById(id);
    const line=document.createElement('div');
    const note=document.createElement('div');
    line.className='session-line';
    note.className='session-values';
    line.hidden=note.hidden=true;
    root.append(line,note);
    return {root,chart,line,note};
});
const qty=new Intl.NumberFormat('vi-VN');
let selectedDate;
function visibleValues(series,digits){
    return series.flatMap(([name,seriesApi])=>{
        const point=valueSeries[name].get(selectedDate);
        return point&&seriesApi.options().visible?[name+' '+point.value.toFixed(digits)]:[];
    });
}
function candleValues(bar){
    return ['open','high','low','close'].map(key=>key[0].toUpperCase()+' '+bar[key].toFixed(2));
}
function drawSession(){
    if(!selectedDate)return;
    const bar=pricesByDate.get(selectedDate);
    const volPoint=volumeByDate.get(selectedDate);
    const average=volumeAverageByDate.get(selectedDate);
    const day=selectedDate.split('-').reverse().join('/');
    const notes=[
        [day,...candleValues(bar),
            ...overlayValues(priceOverlays)].join(' · '),
        [volPoint?'Khối lượng '+qty.format(volPoint.value):'Khối lượng —',
            average?'SMA10 '+qty.format(average.value):'SMA10 —'].join(' · '),
        ...selectedModes.map((mode,i)=>mode===heikinAshiMode
            ? [day,...candleValues(heikinAshiByDate.get(selectedDate)),...overlayValues(indicatorPanes[i].priceOverlays)].join(' · ')
            : [
            day,...(signalMarkers.has(mode)&&candleFocus.getAttribute('aria-pressed')==='true'&&valueSeries['EMA 3'].has(selectedDate)
                ? ['Nến '+valueSeries['EMA 3'].get(selectedDate).value.toFixed(2)] : []),
            ...visibleValues(indicatorPanes[i].display[signalMarkers.has(mode)?stochasticMode:mode],mode==='MACD'?3:2)
        ].join(' · '))
    ];
    const x=price.timeScale().timeToCoordinate(selectedDate);
    readouts.forEach(({root,chart,line,note},i)=>{
        const inView=x!==null&&x>=0&&x<root.clientWidth-chart.priceScale('right').width();
        line.hidden=!inView;
        note.hidden=!inView||notesButton.getAttribute('aria-pressed')!=='true';
        if(inView){
            line.style.left=x+'px';
            note.textContent=notes[i];
        }
    });
}
function selectSession(point){
    if(!pricesByDate.has(point.time))return;
    selectedDate=point.time;
    drawSession();
    scheduleLayoutUpdate();
}
for(const [,chart] of panes){
    chart.subscribeClick(selectSession);
    chart.subscribeCrosshairMove(selectSession);
}
toolbar.addEventListener('click',scheduleLayoutUpdate);
function equalRange(a,b){
    return a&&b&&Math.abs(a.from-b.from)<1e-6&&Math.abs(a.to-b.to)<1e-6;
}
let rangeUpdateInProgress=false;
function syncRange(source,range){
    if(!range||rangeUpdateInProgress||!equalRange(range,source.getVisibleLogicalRange()))return;
    rangeUpdateInProgress=true;
    try {
        for(const target of scales){
            if(target!==source&&!equalRange(range,target.getVisibleLogicalRange())){
                target.setVisibleLogicalRange(range);
            }
        }
    } finally {
        rangeUpdateInProgress=false;
    }
}
let layoutFrame=0;
let sharedAxisWidth=baseAxisWidth;
function scheduleLayoutUpdate(){
    if(layoutFrame)return;
    layoutFrame=requestAnimationFrame(()=>{
        layoutFrame=0;
        const visiblePanes=readouts.filter(({root})=>root.clientWidth>0);
        const axisWidth=Math.max(sharedAxisWidth,...visiblePanes.map(({chart})=>chart.priceScale('right').width()));
        if(axisWidth>sharedAxisWidth){
            sharedAxisWidth=axisWidth;
            for(const [,chart] of panes)chart.priceScale('right').applyOptions({minimumWidth:axisWidth});
            scheduleLayoutUpdate();
        }
        drawSession();
    });
}
for(const scale of scales)scale.subscribeVisibleLogicalRangeChange(range=>{
    syncRange(scale,range);
    scheduleLayoutUpdate();
});
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' '+data.barUnit+' · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng';
renderIndicators();
let initialRangeApplied=false;
const observer=new ResizeObserver(()=>{
    const range=initialRangeApplied?price.timeScale().getVisibleLogicalRange():
        {from:-1,to:timeline.length-1};
    rangeUpdateInProgress=true;
    try {
        for(const [id,chart] of panes){
            const width=document.getElementById(id).clientWidth;
            if(width>0)chart.applyOptions({width});
            if(width>0&&range)chart.timeScale().setVisibleLogicalRange(range);
        }
        initialRangeApplied=document.getElementById('price').clientWidth>0;
    } finally {
        rangeUpdateInProgress=false;
    }
});
for(const [id] of panes)observer.observe(document.getElementById(id));
window.addEventListener('resize',()=>{
    renderIndicators();
    scheduleLayoutUpdate();
});
scheduleLayoutUpdate();
} catch(e) {document.getElementById('error').textContent='Lỗi khởi tạo biểu đồ: '+(e && e.message ? e.message : String(e)); console.error(e);}
}
</script></body></html>""".replace("__DATA__", payload)
    with st.container(key="stocklab_technical_chart"):
        st.html("""
            <style>
                .st-key-stocklab_technical_chart { container-type: inline-size; }
                .st-key-stocklab_technical_chart iframe {
                    --stocklab-chart-base-height: 1100px;
                    height: calc(var(--stocklab-chart-base-height) + var(--stocklab-signal-extra, 0px) + 355px) !important;
                }
                @container (max-width: 600px) {
                    .st-key-stocklab_technical_chart iframe {
                        --stocklab-chart-base-height: 750px;
                        --stocklab-chart-base-height: clamp(650px, calc(100svh - 36px), 780px);
                    }
                }
            </style>
        """)
        components.html(html, height=1455, scrolling=False)
