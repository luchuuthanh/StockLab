"""Mobile-friendly TradingView Lightweight Charts embed for StockLab."""
import json
import pandas as pd
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


def render_lightweight_chart(chart: pd.DataFrame, symbol: str) -> None:
    """Render interactive candles with touch navigation, volume and selectable overlays."""
    # Standard Ichimoku (9, 26, 52), with 26-period projection.
    ordered = chart.sort_values("date").drop_duplicates("date", keep="last").copy()
    hi9 = ordered["high"].rolling(9, min_periods=9).max()
    lo9 = ordered["low"].rolling(9, min_periods=9).min()
    hi26 = ordered["high"].rolling(26, min_periods=26).max()
    lo26 = ordered["low"].rolling(26, min_periods=26).min()
    hi52 = ordered["high"].rolling(52, min_periods=52).max()
    lo52 = ordered["low"].rolling(52, min_periods=52).min()
    tenkan = (hi9 + lo9) / 2
    kijun = (hi26 + lo26) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (hi52 + lo52) / 2
    dates = list(ordered["date"])
    future = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=26)
    projected_dates = dates + list(future)
    def ichipoints(values, shift=0):
        return [{"time": projected_dates[i + shift].strftime("%Y-%m-%d"), "value": round(float(v), 6)}
                for i, v in enumerate(values) if pd.notna(v) and 0 <= i + shift < len(projected_dates)]
    ichimoku = {"Tenkan": ichipoints(tenkan), "Kijun": ichipoints(kijun),
                "SenkouA": ichipoints(span_a, 26), "SenkouB": ichipoints(span_b, 26),
                "Chikou": ichipoints(ordered["close"], -26)}
    chart = ordered
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
    stochastic_bars = []
    green = stochastic["EMA 3"]
    for day, previous, current in zip(dates, green.shift(), green):
        bar = {"time": day.strftime("%Y-%m-%d")}
        if pd.notna(previous) and pd.notna(current):
            bar.update(open=round(float(previous), 6), close=round(float(current), 6),
                       high=round(float(max(previous, current)), 6),
                       low=round(float(min(previous, current)), 6))
        stochastic_bars.append(bar)
    payload = json.dumps({"candles": candles, "volume": volume,
                          "overlays": {k: points(k) for k in ("EMA7", "SMA20", "EMA20", "EMA50", "EMA100")},
                          "rsi": points("RSI14"), "macd": points("MACD"),
                          "signal": points("MACD_SIGNAL"), "hist": points("MACD_HIST"),
                          "symbol": symbol, "ichimoku": ichimoku,
                          "timeline": [day.strftime("%Y-%m-%d") for day in projected_dates],
                          "stochastic": {name: ichipoints(values) for name, values in stochastic.items()},
                          "stochasticBars": stochastic_bars}, ensure_ascii=True).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body{margin:0;background:#101722;color:#e4eaf2;font:13px system-ui,sans-serif}
#toolbar{display:flex;gap:6px;overflow-x:auto;padding:8px;white-space:nowrap}
button{background:#263446;color:#e4eaf2;border:1px solid #415168;border-radius:7px;padding:9px 12px}
button.active{background:#186b7c;border-color:#32c7c5}#status{padding:4px 10px;min-height:18px;color:#b5c7d8}
#price{height:410px}#volume{height:90px}#osc{height:220px}#osc-tools{padding:4px 8px}#osc-tools[hidden]{display:none}#error{padding:12px;color:#f6bd60}
</style></head><body><div id="toolbar"></div><div id="status">Chạm vào nến để xem giá · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng</div>
<div id="price"></div><div id="volume"></div><div id="osc-tools" hidden><button id="candle-focus" type="button" aria-pressed="false">Chỉ xem nến</button></div><div id="osc"></div><div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' phiên · Đang khởi tạo biểu đồ';
if(!window.LightweightCharts){document.getElementById('error').textContent='Lỗi tải thư viện biểu đồ từ CDN. Vui lòng kiểm tra kết nối mạng.';}
else {try {
const L=window.LightweightCharts;
const base={layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},rightPriceScale:{borderColor:'#35465a',minimumWidth:76},timeScale:{borderColor:'#35465a',timeVisible:false},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const price=L.createChart(document.getElementById('price'),{...base,width:Math.max(320,document.getElementById('price').clientWidth),height:410});
const candle=price.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderVisible:false});
candle.setData(data.candles);
const colors={EMA7:'#f8d05a',SMA20:'#37c3ff',EMA20:'#b48bff',EMA50:'#f59e73',EMA100:'#e5e7eb'};
const overlay={};
const ichiColors={Tenkan:'#ff9d43',Kijun:'#e95bb5',SenkouA:'#29bd88',SenkouB:'#f87171',Chikou:'#9acaff'};
const ichi={};
for(const [name,points] of Object.entries(data.ichimoku)){const series=price.addLineSeries({color:ichiColors[name],lineWidth:2,priceLineVisible:false,lastValueVisible:false,visible:false});series.setData(points);ichi[name]=series;}
for(const [name,points] of Object.entries(data.overlays)){const s=price.addLineSeries({color:colors[name],lineWidth:2,priceLineVisible:false,lastValueVisible:true});s.setData(points);overlay[name]=s;}
const vol=L.createChart(document.getElementById('volume'),{...base,width:Math.max(320,document.getElementById('volume').clientWidth),height:90,timeScale:{...base.timeScale,visible:false}});
vol.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'right',priceLineVisible:false,lastValueVisible:false}).setData(data.volume);
const osc=L.createChart(document.getElementById('osc'),{...base,width:Math.max(320,document.getElementById('osc').clientWidth),height:220});
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
const stochasticColors=['#f3f5fa','#48da87','#fce06a','#ff646a','#ec91ad','#69aaff'];
const stochasticLines=Object.entries(data.stochastic).map(([name,points],i)=>{
    const series=osc.addLineSeries({
        color:stochasticColors[i],lineWidth:1,lineStyle:i<4?0:2,
        priceLineVisible:false,lastValueVisible:false,visible:false
    });
    series.setData(points);
    return [name,series];
});
const stochasticMode='Stochastic thử';
const modes={RSI:[rsi,rsi30,rsi70],MACD:[macd,signal,hist],
    [stochasticMode]:[stochasticBars,...stochasticLines.map(([,series])=>series)]};
const candleFocusSeries=new Set([stochasticBars,...stochasticLines
    .filter(([name])=>name==='Biên trên'||name==='Biên dưới').map(([,series])=>series)]);
osc.subscribeCrosshairMove(point=>{
    if(current!==stochasticMode||!point.time)return;
    const values=stochasticLines.flatMap(([name,series])=>{
        const value=point.seriesData.get(series)?.value;
        return value===undefined?[]:[name+' '+value.toFixed(2)];
    });
    document.getElementById('status').textContent=data.symbol+' · '+point.time+' · '+values.join(' · ');
});
let current='RSI';
const oscTools=document.getElementById('osc-tools');
const candleFocus=document.getElementById('candle-focus');
function setMode(name){
    current=name;
    const stochasticActive=name===stochasticMode;
    const candlesOnly=stochasticActive&&candleFocus.getAttribute('aria-pressed')==='true';
    for(const [key,series] of Object.entries(modes)){
        for(const item of series){
            item.applyOptions({visible:key===name&&(!candlesOnly||candleFocusSeries.has(item))});
        }
    }
    oscTools.hidden=!stochasticActive;
    document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===name));
}
candleFocus.onclick=()=>{
    const active=candleFocus.getAttribute('aria-pressed')!=='true';
    candleFocus.setAttribute('aria-pressed',String(active));
    candleFocus.classList.toggle('active',active);
    candleFocus.textContent=active?'Hiện lại các đường':'Chỉ xem nến';
    osc.priceScale('right').applyOptions({autoScale:true});
    setMode(stochasticMode);
};
const toolbar=document.getElementById('toolbar');
const ichiButton=document.createElement('button');ichiButton.textContent='Ichimoku (9,26,52)';ichiButton.onclick=()=>{const on=!ichiButton.classList.contains('active');ichiButton.classList.toggle('active',on);for(const series of Object.values(ichi))series.applyOptions({visible:on});};toolbar.appendChild(ichiButton);
for(const name of Object.keys(overlay)){const b=document.createElement('button');b.textContent=name;b.className=['EMA7','SMA20','EMA100'].includes(name)?'active':'';overlay[name].applyOptions({visible:b.className==='active'});b.onclick=()=>{const on=!b.classList.contains('active');b.classList.toggle('active',on);overlay[name].applyOptions({visible:on});};toolbar.appendChild(b);}
for(const name of Object.keys(modes)){const b=document.createElement('button');b.textContent=name;b.dataset.mode=name;b.onclick=()=>setMode(name);toolbar.appendChild(b);}
setMode('RSI');
price.subscribeCrosshairMove(p=>{if(!p.time)return;const v=p.seriesData.get(candle);if(v)document.getElementById('status').textContent=data.symbol+' · '+p.time+' · O '+v.open+' H '+v.high+' L '+v.low+' C '+v.close;});
const panes=[['price',price],['volume',vol],['osc',osc]];
// Every pane needs identical logical indexes, including Ichimoku projection dates.
const timeline=data.timeline.map(time=>({time,value:0}));
for(const [,chart] of panes){
    chart.addLineSeries({visible:false,priceLineVisible:false,lastValueVisible:false}).setData(timeline);
}
const scales=panes.map(([,chart])=>chart.timeScale());
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
for(const scale of scales)scale.subscribeVisibleLogicalRangeChange(range=>syncRange(scale,range));
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' phiên · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng';
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
        initialRangeApplied=panes.every(([id])=>document.getElementById(id).clientWidth>0);
    } finally {
        rangeUpdateInProgress=false;
    }
});
for(const [id] of panes)observer.observe(document.getElementById(id));
} catch(e) {document.getElementById('error').textContent='Lỗi khởi tạo biểu đồ: '+(e && e.message ? e.message : String(e)); console.error(e);}
}
</script></body></html>""".replace("__DATA__", payload)
    components.html(html, height=875, scrolling=False)
