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


def stochastic_trade_markers(stochastic: pd.DataFrame, dates) -> list[dict[str, str]]:
    """Mark crossings of the oscillator candle close and the red EMA 9 line."""
    markers, _ = stochastic_trade_events(stochastic, dates)
    return markers


def stochastic_trade_events(stochastic, dates, *, after="", holding=False, ready_to_buy=False):
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
        if not holding and current < 25:
            ready_to_buy = True
        if pd.isna(previous) or pd.isna(previous_red):
            continue
        if holding and previous >= previous_red and current < current_red and current < previous:
            markers.append({"time": signal_date, "position": "aboveBar",
                            "color": "#ff6068", "shape": "arrowDown", "text": "Bán"})
            holding = False
            ready_to_buy = current < 25
        elif not holding and ready_to_buy and previous <= previous_red and current > current_red and current > previous:
            markers.append({"time": signal_date, "position": "belowBar",
                            "color": "#32d981", "shape": "arrowUp", "text": "Mua"})
            holding = True
            ready_to_buy = False
    return markers, ready_to_buy


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
                          "rsi": points("RSI14"), "macd": points("MACD"), "mom": points("MOM10"),
                          "signal": points("MACD_SIGNAL"), "hist": points("MACD_HIST"),
                          "symbol": symbol, "barUnit": "phiên" if timeframe == "1D" else "nến",
                          "ichimoku": ichimoku, "bollinger": bollinger,
                          "timeline": [day.strftime("%Y-%m-%d") for day in projected_dates],
                          "stochastic": {name: ichipoints(values) for name, values in stochastic.items()},
                          "stochasticBars": stochastic_bars,
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
</style></head><body><div id="toolbar"></div><div id="status">Chạm vào nến để xem giá · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng</div>
<div id="price"></div><div id="volume"></div>
<section id="indicator-1" aria-labelledby="indicator-title-1"><div class="indicator-header"><h2 id="indicator-title-1"></h2><button id="candle-focus" type="button" aria-pressed="false" hidden>Chỉ xem nến</button></div><div class="indicator-chart" id="osc"></div></section>
<section id="indicator-2" aria-labelledby="indicator-title-2" hidden><div class="indicator-header"><h2 id="indicator-title-2"></h2></div><div class="indicator-chart" id="osc-2"></div></section>
<div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' '+data.barUnit+' · Đang khởi tạo biểu đồ';
if(!window.LightweightCharts){document.getElementById('error').textContent='Lỗi tải thư viện biểu đồ từ CDN. Vui lòng kiểm tra kết nối mạng.';}
else {try {
const L=window.LightweightCharts;
const compactScreen=window.matchMedia('(max-width: 600px)');
const chartSizes=()=>compactScreen.matches
    ? {price:225,volume:55,indicator:116,header:30}
    : {price:410,volume:90,indicator:220,header:36};
const base={crosshair:{vertLine:{visible:false,labelVisible:false},horzLine:{visible:true,labelVisible:true}},layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},rightPriceScale:{borderColor:'#35465a',minimumWidth:76},timeScale:{borderColor:'#35465a',timeVisible:false},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const price=L.createChart(document.getElementById('price'),{...base,width:Math.max(320,document.getElementById('price').clientWidth),height:chartSizes().price});
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
const vol=L.createChart(document.getElementById('volume'),{...base,width:Math.max(320,document.getElementById('volume').clientWidth),height:chartSizes().volume,timeScale:{...base.timeScale,visible:false}});
vol.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'right',priceLineVisible:false,lastValueVisible:false}).setData(data.volume);
const volumeAverage=vol.addLineSeries({color:'#f8d05a',lineWidth:2,priceFormat:{type:'volume'},
    priceLineVisible:false,lastValueVisible:false});
volumeAverage.setData(data.volumeSma10);
const stochasticMode='Bộ chỉ báo mua bán';
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
    stochasticBars.setMarkers(data.stochasticMarkers);
    const stochasticColors=['#f3f5fa','#48da87','#fce06a','#ff646a','#ec91ad','#69aaff'];
    const stochasticLines=Object.entries(data.stochastic).map(([name,points],i)=>{
        const series=osc.addLineSeries({
            color:stochasticColors[i],lineWidth:1,lineStyle:i<4?0:2,
            priceLineVisible:false,lastValueVisible:false,visible:false
        });
        series.setData(points);
        return [name,series];
    });
    const mom=osc.addLineSeries({color:'#69c6f0',lineWidth:2,priceLineVisible:false,visible:false});
    mom.setData(data.mom);
    const modes={RSI:[rsi,rsi30,rsi70],MACD:[macd,signal,hist],MOM:[mom],
        [stochasticMode]:[stochasticBars,...stochasticLines.map(([,series])=>series)]};
    const candleFocusSeries=new Set([stochasticBars,...stochasticLines
        .filter(([name])=>name==='Biên trên'||name==='Biên dưới').map(([,series])=>series)]);
    const display={RSI:[['RSI',rsi]],MACD:[['MACD',macd],['Signal',signal],['Histogram',hist]],
        MOM:[['MOM',mom]],[stochasticMode]:stochasticLines};
    return {id,root,title,section,chart:osc,modes,display,candleFocusSeries};
}
const indicatorPanes=[
    createIndicatorPane('osc','indicator-1','indicator-title-1'),
    createIndicatorPane('osc-2','indicator-2','indicator-title-2')
];
const selectedModes=['RSI','MACD'];
const buttons=new Map();
function resizeChartPanes(){
    const {price:priceHeight,volume:volumeHeight,indicator,header}=chartSizes();
    const lastIndicator=selectedModes.length-1;
    const resize=(root,chart,height,showAxis)=>{
        root.classList.toggle('has-time-axis',showAxis);
        chart.applyOptions({width:root.clientWidth||320,height,timeScale:{visible:showAxis}});
    };
    resize(document.getElementById('price'),price,priceHeight,!compactScreen.matches);
    resize(document.getElementById('volume'),vol,volumeHeight,false);
    indicatorPanes.slice(0,selectedModes.length).forEach((pane,i)=>{
        const height=lastIndicator===0?indicator*2+header:indicator;
        resize(pane.root,pane.chart,height,!compactScreen.matches||i===lastIndicator);
    });
}
function renderIndicators(){
    const range=price.timeScale().getVisibleLogicalRange();
    const candlesOnly=candleFocus.getAttribute('aria-pressed')==='true';
    rangeUpdateInProgress=true;
    try {
        indicatorPanes.forEach((pane,i)=>{
            const mode=selectedModes[i];
            pane.section.hidden=!mode;
            if(!mode)return;
            const stochasticActive=mode===stochasticMode;
            pane.title.textContent=mode;
            if(stochasticActive)pane.title.parentElement.appendChild(candleFocus);
            for(const [name,series] of Object.entries(pane.modes)){
                for(const item of series){
                    item.applyOptions({visible:name===mode&&(!stochasticActive||!candlesOnly||pane.candleFocusSeries.has(item))});
                }
            }
            pane.chart.priceScale('right').applyOptions({autoScale:true});
        });
        resizeChartPanes();
        if(range){
            for(const [,chart] of panes)chart.timeScale().setVisibleLogicalRange(range);
        }
    } finally {
        rangeUpdateInProgress=false;
    }
    candleFocus.hidden=!selectedModes.includes(stochasticMode);
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
for(const name of Object.keys(indicatorPanes[0].modes)){
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
const indexByDate=points=>new Map(points.map(point=>[point.time,point]));
const pricesByDate=indexByDate(data.candles);
const volumeByDate=indexByDate(data.volume);
const volumeAverageByDate=indexByDate(data.volumeSma10);
const valueSeries=Object.fromEntries(Object.entries({
    ...data.overlays,...data.ichimoku,
    'BB trên':data.bollinger.upper,'BB giữa':data.bollinger.middle,'BB dưới':data.bollinger.lower,
    RSI:data.rsi,MACD:data.macd,Signal:data.signal,Histogram:data.hist,MOM:data.mom,...data.stochastic
}).map(([name,points])=>[name,indexByDate(points)]));
const displayPrice=Object.entries(overlay).concat(Object.entries(ichi),
    ['BB trên','BB giữa','BB dưới'].map((name,i)=>[name,bbSeries[i]]));
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
        ...selectedModes.map((mode,i)=>[
            day,...(mode===stochasticMode&&candleFocus.getAttribute('aria-pressed')==='true'&&valueSeries['EMA 3'].has(selectedDate)
                ? ['Nến '+valueSeries['EMA 3'].get(selectedDate).value.toFixed(2)] : []),
            ...visibleValues(indicatorPanes[i].display[mode],mode==='MACD'?3:2)
        ].join(' · '))
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
compactScreen.addEventListener('change',()=>{
    renderIndicators();
    drawSession();
});
} catch(e) {document.getElementById('error').textContent='Lỗi khởi tạo biểu đồ: '+(e && e.message ? e.message : String(e)); console.error(e);}
}
</script></body></html>""".replace("__DATA__", payload)
    with st.container(key="stocklab_technical_chart"):
        st.html("""
            <style>
                .st-key-stocklab_technical_chart { container-type: inline-size; }
                @container (max-width: 600px) {
                    .st-key-stocklab_technical_chart iframe { height: 650px !important; }
                }
            </style>
        """)
        components.html(html, height=1100, scrolling=False)
