"""Two-stock stacked D1 candlestick comparison using TradingView Lightweight Charts."""
import json
import pandas as pd
import streamlit.components.v1 as components


def render_stock_comparison(first: pd.DataFrame, first_symbol: str,
                            second: pd.DataFrame, second_symbol: str) -> None:
    def pack(frame):
        frame = frame.sort_values("date").drop_duplicates("date").copy()
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        def line(values):
            return [{"time": day, "value": round(float(v), 6)}
                    for day, v in zip(dates, values) if pd.notna(v)]
        close = frame["close"]
        middle = close.rolling(20, min_periods=20).mean()
        width = close.rolling(20, min_periods=20).std(ddof=0) * 2
        return {
            "candles": [{"time": day, "open": float(row.open), "high": float(row.high),
                         "low": float(row.low), "close": float(row.close)}
                        for day, row in zip(dates, frame.itertuples(index=False))],
            "ema7": line(frame["EMA7"]),
            "sma20": line(frame["SMA20"]),
            "ema100": line(frame["EMA100"]),
            "bbUpper": line(middle + width),
            "bbLower": line(middle - width),
        }
    data = {"symbols": [first_symbol, second_symbol],
            "series": [pack(first), pack(second)]}
    data["timeline"] = sorted(set(p["time"] for item in data["series"] for p in item["candles"]))
    payload = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;background:#101722;color:#e4eaf2;font:13px system-ui,sans-serif}
#tools{display:flex;gap:6px;padding:8px;overflow-x:auto;white-space:nowrap}
button{background:#263446;color:#e4eaf2;border:1px solid #415168;border-radius:7px;padding:9px 12px}
button.active{background:#186b7c;border-color:#32c7c5}
.header{padding:8px 10px;font-weight:700;color:#dbe9f4}
.pane{height:340px;position:relative}
.session-line{position:absolute;top:0;bottom:26px;border-left:1px dashed #e1e8f4;pointer-events:none;z-index:3}
.session-note{position:absolute;top:4px;left:8px;max-width:calc(100% - 95px);padding:5px 7px;background:#101722ee;z-index:3;pointer-events:none;border-radius:4px;font-size:12px;line-height:1.35}
#hint{padding:6px 10px;color:#a8bed0}#error{padding:8px;color:#f6bd60}
</style></head><body>
<div id="tools"><button id="ma" class="active">EMA7 / SMA20 / EMA100</button><button id="bb">BB</button><button id="notes" class="active">Chú thích</button></div>
<div class="header" id="title0"></div><div id="pane0" class="pane"></div>
<div class="header" id="title1"></div><div id="pane1" class="pane"></div>
<div id="hint">Vuốt ngang hoặc chụm hai ngón để thay đổi cả hai biểu đồ · Chạm nến để xem cùng ngày</div><div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
if(!window.LightweightCharts){document.getElementById('error').textContent='Không tải được thư viện biểu đồ.';}
else {try {
const L=window.LightweightCharts;
const base={layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},
grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},
rightPriceScale:{borderColor:'#35465a',minimumWidth:76},
timeScale:{borderColor:'#35465a',timeVisible:false},
crosshair:{vertLine:{visible:false,labelVisible:false},horzLine:{visible:true,labelVisible:true}},
handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},
handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const items=data.series.map((entry,i)=>{
 document.getElementById('title'+i).textContent=data.symbols[i]+' · Nến D1';
 const root=document.getElementById('pane'+i);
 const chart=L.createChart(root,{...base,width:Math.max(320,root.clientWidth),height:340});
 const candle=chart.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderVisible:false});
 candle.setData(entry.candles);
 const overlay=[
  ['ema7','#f8d05a'],['sma20','#37c3ff'],['ema100','#e5e7eb']
 ].map(([field,color])=>{const s=chart.addLineSeries({color,lineWidth:2,priceLineVisible:false,lastValueVisible:true});s.setData(entry[field]);return s;});
 const bands=['bbUpper','bbLower'].map(field=>{const s=chart.addLineSeries({color:'#7897ed',lineWidth:1,lineStyle:2,visible:false,priceLineVisible:false,lastValueVisible:false});s.setData(entry[field]);return s;});
 // A common invisible calendar ensures matching logical bar positions on both charts.
 const anchor=chart.addLineSeries({visible:false,priceLineVisible:false,lastValueVisible:false});
 anchor.setData(data.timeline.map(time=>({time,value:0})));
 const line=document.createElement('div');line.className='session-line';line.hidden=true;
 const note=document.createElement('div');note.className='session-note';note.hidden=true;
 root.append(line,note);
 return {root,chart,overlay,bands,line,note,byDate:new Map(entry.candles.map(p=>[p.time,p]))};
});
const controls={ma:document.getElementById('ma'),bb:document.getElementById('bb'),notes:document.getElementById('notes')};
for(const [key,seriesKey] of [['ma','overlay'],['bb','bands']]){
 controls[key].onclick=()=>{
  const active=!controls[key].classList.contains('active');controls[key].classList.toggle('active',active);
  items.forEach(item=>item[seriesKey].forEach(s=>s.applyOptions({visible:active})));
 };
}
controls.notes.onclick=()=>{controls.notes.classList.toggle('active');draw();};
let selected=null;
function draw(){
 items.forEach(item=>{
  const bar=item.byDate.get(selected);
  const x=selected?item.chart.timeScale().timeToCoordinate(selected):null;
  const visible=x!==null&&x>=0&&x<item.root.clientWidth-76;
  item.line.hidden=!visible;
  item.note.hidden=!visible||!controls.notes.classList.contains('active');
  if(visible){
   item.line.style.left=x+'px';
   item.note.textContent=bar?selected.split('-').reverse().join('/')+' · O '+bar.open.toFixed(2)+' · H '+bar.high.toFixed(2)+' · L '+bar.low.toFixed(2)+' · C '+bar.close.toFixed(2):selected.split('-').reverse().join('/')+' · Không giao dịch';
  }
 });
}
function select(point){
 if(!point||!point.time)return;
 const time=typeof point.time==='string'?point.time:null;
 if(time&&time!==selected){selected=time;draw();}
}
items.forEach(item=>{item.chart.subscribeCrosshairMove(select);item.chart.subscribeClick(select);});
let syncing=false;
function equal(a,b){return a&&b&&Math.abs(a.from-b.from)<0.0001&&Math.abs(a.to-b.to)<0.0001;}
items.forEach(source=>source.chart.timeScale().subscribeVisibleLogicalRangeChange(range=>{
 if(syncing||!range)return;
 syncing=true;
 try{items.forEach(target=>{if(target!==source&&!equal(range,target.chart.timeScale().getVisibleLogicalRange()))target.chart.timeScale().setVisibleLogicalRange(range);});}finally{syncing=false;}
 draw();
}));
let initialized=false;
const observer=new ResizeObserver(()=>{
 const range=initialized?items[0].chart.timeScale().getVisibleLogicalRange():{from:Math.max(-1,data.timeline.length-90),to:data.timeline.length-1};
 syncing=true;
 try{items.forEach(item=>{if(item.root.clientWidth>0){item.chart.applyOptions({width:item.root.clientWidth});if(range)item.chart.timeScale().setVisibleLogicalRange(range);}});}finally{syncing=false;}
 initialized=items.every(item=>item.root.clientWidth>0);
 draw();
});
items.forEach(item=>observer.observe(item.root));
}catch(e){document.getElementById('error').textContent='Lỗi biểu đồ: '+(e.message||String(e));console.error(e);}}
</script></body></html>""".replace("__DATA__", payload)
    components.html(html, height=810, scrolling=False)
