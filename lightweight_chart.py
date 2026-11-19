"""Mobile-friendly TradingView Lightweight Charts embed for StockLab."""
import json
import pandas as pd
import streamlit.components.v1 as components


def render_lightweight_chart(chart: pd.DataFrame, symbol: str) -> None:
    """Render interactive candles with touch navigation, volume and selectable overlays."""
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
    payload = json.dumps({"candles": candles, "volume": volume,
                          "overlays": {k: points(k) for k in ("EMA7", "SMA20", "EMA20", "EMA50", "EMA100")},
                          "rsi": points("RSI14"), "macd": points("MACD"),
                          "signal": points("MACD_SIGNAL"), "hist": points("MACD_HIST"),
                          "symbol": symbol}, ensure_ascii=True).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body{margin:0;background:#101722;color:#e4eaf2;font:13px system-ui,sans-serif}
#toolbar{display:flex;gap:6px;overflow-x:auto;padding:8px;white-space:nowrap}
button{background:#263446;color:#e4eaf2;border:1px solid #415168;border-radius:7px;padding:9px 12px}
button.active{background:#186b7c;border-color:#32c7c5}#status{padding:4px 10px;min-height:18px;color:#b5c7d8}
#price{height:410px}#volume{height:90px}#osc{height:145px}#error{padding:12px;color:#f6bd60}
</style></head><body><div id="toolbar"></div><div id="status">Chạm vào nến để xem giá · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng</div>
<div id="price"></div><div id="volume"></div><div id="osc"></div><div id="error"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const data=__DATA__;
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' phiên · Đang khởi tạo biểu đồ';
if(!window.LightweightCharts){document.getElementById('error').textContent='Lỗi tải thư viện biểu đồ từ CDN. Vui lòng kiểm tra kết nối mạng.';}
else {try {
const L=window.LightweightCharts;
const base={layout:{background:{type:'solid',color:'#101722'},textColor:'#c8d2df'},grid:{vertLines:{color:'#1b2736'},horzLines:{color:'#1b2736'}},rightPriceScale:{borderColor:'#35465a'},timeScale:{borderColor:'#35465a',timeVisible:false},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true}};
const price=L.createChart(document.getElementById('price'),{...base,width:Math.max(320,document.getElementById('price').clientWidth),height:410});
const candle=price.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderVisible:false});
candle.setData(data.candles);
const colors={EMA7:'#f8d05a',SMA20:'#37c3ff',EMA20:'#b48bff',EMA50:'#f59e73',EMA100:'#e5e7eb'};
const overlay={};
for(const [name,points] of Object.entries(data.overlays)){const s=price.addLineSeries({color:colors[name],lineWidth:2,priceLineVisible:false,lastValueVisible:true});s.setData(points);overlay[name]=s;}
const vol=L.createChart(document.getElementById('volume'),{...base,width:Math.max(320,document.getElementById('volume').clientWidth),height:90,timeScale:{...base.timeScale,visible:false},rightPriceScale:{visible:false}});
vol.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:''}).setData(data.volume);
const osc=L.createChart(document.getElementById('osc'),{...base,width:Math.max(320,document.getElementById('osc').clientWidth),height:145});
const rsi=osc.addLineSeries({color:'#dba5ff',lineWidth:2,priceLineVisible:false});rsi.setData(data.rsi);
const macd=osc.addLineSeries({color:'#36c6e9',lineWidth:2,priceLineVisible:false});macd.setData(data.macd);
const signal=osc.addLineSeries({color:'#f8d05a',lineWidth:2,priceLineVisible:false});signal.setData(data.signal);
const hist=osc.addHistogramSeries({color:'#5aa59c',priceLineVisible:false});hist.setData(data.hist);
const modes={RSI:[rsi],MACD:[macd,signal,hist]};
let current='RSI';
function setMode(name){current=name;for(const [key,series] of Object.entries(modes))for(const s of series)s.applyOptions({visible:key===name});document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===name));osc.timeScale().fitContent();}
const toolbar=document.getElementById('toolbar');
for(const name of Object.keys(overlay)){const b=document.createElement('button');b.textContent=name;b.className=['EMA7','SMA20','EMA100'].includes(name)?'active':'';overlay[name].applyOptions({visible:b.className==='active'});b.onclick=()=>{const on=!b.classList.contains('active');b.classList.toggle('active',on);overlay[name].applyOptions({visible:on});};toolbar.appendChild(b);}
for(const name of ['RSI','MACD']){const b=document.createElement('button');b.textContent=name;b.dataset.mode=name;b.onclick=()=>setMode(name);toolbar.appendChild(b);}
setMode('RSI');
price.subscribeCrosshairMove(p=>{if(!p.time)return;const v=p.seriesData.get(candle);if(v)document.getElementById('status').textContent=data.symbol+' · '+p.time+' · O '+v.open+' H '+v.high+' L '+v.low+' C '+v.close;});
let syncing=false;
function sync(from,to){from.timeScale().subscribeVisibleLogicalRangeChange(range=>{if(!range||syncing)return;syncing=true;to.timeScale().setVisibleLogicalRange(range);syncing=false;});}
sync(price,vol);sync(price,osc);
price.timeScale().fitContent();vol.timeScale().fitContent();osc.timeScale().fitContent();
document.getElementById('status').textContent=data.symbol+' · '+data.candles.length+' phiên · Vuốt để xem lịch sử · Chụm hai ngón để thu phóng';
const observer=new ResizeObserver(()=>{for(const [id,c] of [['price',price],['volume',vol],['osc',osc]])c.applyOptions({width:document.getElementById(id).clientWidth});});
for(const id of ['price','volume','osc'])observer.observe(document.getElementById(id));
setTimeout(()=>{for(const [id,c] of [['price',price],['volume',vol],['osc',osc]])c.applyOptions({width:Math.max(320,document.getElementById(id).clientWidth)});price.timeScale().fitContent();},500);
} catch(e) {document.getElementById('error').textContent='Lỗi khởi tạo biểu đồ: '+(e && e.message ? e.message : String(e)); console.error(e);}
}
</script></body></html>""".replace("__DATA__", payload)
    components.html(html, height=735, scrolling=False)
