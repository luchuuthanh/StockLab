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


def render_lightweight_chart(chart: pd.DataFrame, symbol: str, timeframe: str = "1D") -> None:
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
    volume_sma10 = ichipoints(chart["volume"].rolling(10).mean())
    payload = json.dumps({"candles": candles, "volume": volume, "volumeSma10": volume_sma10,
                          "overlays": {k: points(k) for k in ("EMA7", "SMA20", "EMA20", "EMA50", "EMA100")},
                          "rsi": points("RSI14"), "macd": points("MACD"),
                          "signal": points("MACD_SIGNAL"), "hist": points("MACD_HIST"),
                          "symbol": symbol, "barUnit": "phiên" if timeframe == "1D" else "nến",
                          "ichimoku": ichimoku, "bollinger": bollinger,
                          "timeline": [day.strftime("%Y-%m-%d") for day in projected_dates],
                          "stochastic": {name: ichipoints(values) for name, values in stochastic.items()},
                          "stochasticBars": stochastic_bars}, ensure_ascii=True).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body{margin:0;background:#101722;color:#e4eaf2;font:13px system-ui,sans-serif}
#toolbar{display:flex;gap:6px;overflow-x:auto;padding:8px;white-space:nowrap}
button{background:#263446;color:#e4eaf2;border:1px solid #415168;border-radius:7px;padding:9px 12px}
button.active{background:#186b7c;border-color:#32c7c5}#status{padding:4px 10px;min-height:18px;color:#b5c7d8}
#price{height:410px}#volume{height:90px}#osc{height:220px}#osc-tools{padding:4px 8px}#osc-tools[hidden]{display:none}#error{padding:12px;color:#f6bd60}
#price,#volume,#osc{position:relative}
.session-line,.session-values{position:absolute;z-index:2;pointer-events:none}
.session-line{top:0;bottom:0;border-left:1px dashed #d7e0ed}
#price .session-line,#osc .session-line{bottom:26px}
.session-values{top:4px;left:6px;max-width:calc(100% - 90px);padding:4px 6px;border-radius:4px;background:#101722ee;color:#edf3fb;line-height:1.35;font-size:12px}
</style></head><body><div id="toolbar"></div><div id="status">Chạm vào nến để xem giá · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng</div>
<div id="price"></div><div id="volume"></div><div id="osc-tools" hidden><button id="candle-focus" type="button" aria-pressed="false">Chỉ xem nến</button></div><div id="osc"></div><div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' '+data.barUnit+' · Đang khởi tạo biểu đồ';
if(!window.LightweightCharts){document.getElementById('error').textContent='Lỗi tải thư viện biểu đồ từ CDN. Vui lòng kiểm tra kết nối mạng.';}
else {try {
const L=window.LightweightCharts;
const base={crosshair:{vertLine:{visible:false,labelVisible:false},horzLine:{visible:true,labelVisible:true}},layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},rightPriceScale:{borderColor:'#35465a',minimumWidth:76},timeScale:{borderColor:'#35465a',timeVisible:false},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const price=L.createChart(document.getElementById('price'),{...base,width:Math.max(320,document.getElementById('price').clientWidth),height:410});
const candle=price.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderVisible:false});
candle.setData(data.candles);
const colors={EMA7:'#f8d05a',SMA20:'#37c3ff',EMA20:'#b48bff',EMA50:'#f59e73',EMA100:'#e5e7eb'};
const overlay={};
const bbSeries=Object.entries(data.bollinger).map(([band,points])=>{
    const middle=band==='middle';
    const series=price.addLineSeries({color:middle?'#8eb9f3':'#678ff2',
        lineWidth:middle?1:2,lineStyle:middle?2:0,
        priceLineVisible:false,lastValueVisible:false,visible:false});
    series.setData(points);
    return series;
});
const ichiColors={Tenkan:'#ff9d43',Kijun:'#e95bb5',SenkouA:'#29bd88',SenkouB:'#f87171',Chikou:'#9acaff'};
const ichi={};
for(const [name,points] of Object.entries(data.ichimoku)){const series=price.addLineSeries({color:ichiColors[name],lineWidth:2,priceLineVisible:false,lastValueVisible:false,visible:false});series.setData(points);ichi[name]=series;}
for(const [name,points] of Object.entries(data.overlays)){const s=price.addLineSeries({color:colors[name],lineWidth:2,priceLineVisible:false,lastValueVisible:true});s.setData(points);overlay[name]=s;}
const vol=L.createChart(document.getElementById('volume'),{...base,width:Math.max(320,document.getElementById('volume').clientWidth),height:90,timeScale:{...base.timeScale,visible:false}});
vol.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'right',priceLineVisible:false,lastValueVisible:false}).setData(data.volume);
const volumeAverage=vol.addLineSeries({color:'#f8d05a',lineWidth:2,priceFormat:{type:'volume'},
    priceLineVisible:false,lastValueVisible:false});
volumeAverage.setData(data.volumeSma10);
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
const bbButton=document.createElement('button');
bbButton.textContent='BB';
bbButton.title='Bollinger Bands (20, 2)';
bbButton.setAttribute('aria-pressed','false');
bbButton.onclick=()=>{
    const on=bbButton.getAttribute('aria-pressed')!=='true';
    bbButton.setAttribute('aria-pressed',String(on));
    bbButton.classList.toggle('active',on);
    for(const series of bbSeries)series.applyOptions({visible:on});
};
const notesButton=document.createElement('button');
notesButton.textContent='Chú thích';
notesButton.classList.add('active');
notesButton.setAttribute('aria-pressed','true');
notesButton.onclick=()=>{
    const on=notesButton.getAttribute('aria-pressed')!=='true';
    notesButton.setAttribute('aria-pressed',String(on));
    notesButton.classList.toggle('active',on);
};
toolbar.append(notesButton,bbButton);
const ichiButton=document.createElement('button');ichiButton.textContent='Ichimoku (9,26,52)';ichiButton.onclick=()=>{const on=!ichiButton.classList.contains('active');ichiButton.classList.toggle('active',on);for(const series of Object.values(ichi))series.applyOptions({visible:on});};toolbar.appendChild(ichiButton);
for(const name of Object.keys(overlay)){const b=document.createElement('button');b.textContent=name;b.className=['EMA7','SMA20','EMA100'].includes(name)?'active':'';overlay[name].applyOptions({visible:b.className==='active'});b.onclick=()=>{const on=!b.classList.contains('active');b.classList.toggle('active',on);overlay[name].applyOptions({visible:on});};toolbar.appendChild(b);}
for(const name of Object.keys(modes)){const b=document.createElement('button');b.textContent=name;b.dataset.mode=name;b.onclick=()=>setMode(name);toolbar.appendChild(b);}
setMode('RSI');
const panes=[['price',price],['volume',vol],['osc',osc]];
// Every pane needs identical logical indexes, including Ichimoku projection dates.
const timeline=data.timeline.map(time=>({time,value:0}));
for(const [,chart] of panes){
    chart.addLineSeries({visible:false,priceLineVisible:false,lastValueVisible:false}).setData(timeline);
}
const scales=panes.map(([,chart])=>chart.timeScale());
const indexByDate=points=>new Map(points.map(point=>[point.time,point]));
const pricesByDate=indexByDate(data.candles);
const volumeByDate=indexByDate(data.volume);
const volumeAverageByDate=indexByDate(data.volumeSma10);
const valueSeries=Object.fromEntries(Object.entries({
    ...data.overlays,...data.ichimoku,
    'BB trên':data.bollinger.upper,'BB giữa':data.bollinger.middle,'BB dưới':data.bollinger.lower,
    RSI:data.rsi,MACD:data.macd,Signal:data.signal,Histogram:data.hist,...data.stochastic
}).map(([name,points])=>[name,indexByDate(points)]));
const displayPrice=Object.entries(overlay).concat(Object.entries(ichi),
    ['BB trên','BB giữa','BB dưới'].map((name,i)=>[name,bbSeries[i]]));
const displayOsc={RSI:[['RSI',rsi]],MACD:[['MACD',macd],['Signal',signal],['Histogram',hist]],
    [stochasticMode]:stochasticLines};
const readouts=panes.map(([id,chart])=>{
    const root=document.getElementById(id);
    const line=document.createElement('div');
    const note=document.createElement('div');
    line.className='session-line';
    note.className='session-values';
    line.hidden=note.hidden=true;
    root.append(line,note);
    return {root,chart,scale:chart.timeScale(),line,note};
});
const qty=new Intl.NumberFormat('vi-VN');
let selectedDate;
function visibleValues(series,digits){
    return series.flatMap(([name,seriesApi])=>{
        const point=valueSeries[name].get(selectedDate);
        return point&&seriesApi.options().visible?[name+' '+point.value.toFixed(digits)]:[];
    });
}
function drawSession(){
    if(!selectedDate)return;
    const bar=pricesByDate.get(selectedDate);
    const volPoint=volumeByDate.get(selectedDate);
    const average=volumeAverageByDate.get(selectedDate);
    const day=selectedDate.split('-').reverse().join('/');
    const notes=[
        [day,...['open','high','low','close'].map(key=>key[0].toUpperCase()+' '+bar[key].toFixed(2)),
            ...visibleValues(displayPrice,2)].join(' · '),
        [volPoint?'Khối lượng '+qty.format(volPoint.value):'Khối lượng —',
            average?'SMA10 '+qty.format(average.value):'SMA10 —'].join(' · '),
        [day,...(current===stochasticMode&&candleFocus.getAttribute('aria-pressed')==='true'&&valueSeries['EMA 3'].has(selectedDate)
            ? ['Nến '+valueSeries['EMA 3'].get(selectedDate).value.toFixed(2)] : []),
            ...visibleValues(displayOsc[current],current==='MACD'?3:2)].join(' · ')
    ];
    readouts.forEach(({root,chart,scale,line,note},i)=>{
        const x=scale.timeToCoordinate(selectedDate);
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
}
for(const [,chart] of panes){
    chart.subscribeClick(selectSession);
    chart.subscribeCrosshairMove(selectSession);
}
toolbar.addEventListener('click',drawSession);
oscTools.addEventListener('click',drawSession);
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
for(const scale of scales)scale.subscribeVisibleLogicalRangeChange(range=>{
    syncRange(scale,range);
    drawSession();
});
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' '+data.barUnit+' · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng';
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
