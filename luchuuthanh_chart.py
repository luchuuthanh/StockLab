"""LUCHUUTHANH experimental technical chart, isolated from existing scanner."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _rsi(close, n=14):
    d=close.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    down=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    r=100-100/(1+up/down.replace(0,np.nan))
    return r.mask((down==0)&(up>0),100).mask((down==0)&(up==0),50)

def _signals(df):
    """Experimental stochastic-RSI blend, NOT original TOANDAITUONG AFL."""
    c=df["close"]
    r=_rsi(c)
    lo=r.rolling(14,min_periods=14).min()
    hi=r.rolling(14,min_periods=14).max()
    raw=(100*(r-lo)/(hi-lo).replace(0,np.nan)).clip(0,100)
    fast=raw.ewm(span=3,adjust=False).mean()
    yellow=fast.ewm(span=3,adjust=False).mean()
    slow=yellow.ewm(span=9,adjust=False).mean()
    dev=fast.rolling(14,min_periods=5).std().fillna(0)
    df["LCT_FAST"]=fast
    df["LCT_YELLOW"]=yellow
    df["LCT_SLOW"]=slow
    df["LCT_UPPER"]=(yellow+1.5*dev).clip(0,100)
    df["LCT_LOWER"]=(yellow-1.5*dev).clip(0,100)
    cross_up=(fast>slow)&(fast.shift(1)<=slow.shift(1))&(fast<35)
    cross_down=(fast<slow)&(fast.shift(1)>=slow.shift(1))&(fast>65)
    df["LCT_BUY"]=fast.where(cross_up)
    df["LCT_SELL"]=fast.where(cross_down)
    return df

def render_luchuuthanh_chart():
    from stock_dashboard import fetch_chart_quotes, sanitize_daily_chart_bars, add_indicators
    st.subheader("📊 LUCHUUTHANH – Biểu đồ kỹ thuật")
    st.caption("Khung sáng là mô hình tái tạo thử nghiệm từ ảnh, chưa xác minh công thức AFL gốc.")
    symbol=st.text_input("Mã cổ phiếu",value="FPT",key="lct_symbol").strip().upper()
    days=st.selectbox("Khoảng thời gian hiển thị (ngày)",[30,60,90,120,180,260,365,520,730],index=2,key="lct_days_v2")
    # Always render the original five stacked panels; no stale per-panel widget state.
    view="Tổng hợp 5 biểu đồ"
    st.caption("Chế độ mặc định: Tổng hợp 5 biểu đồ (Giá/EMA · LUCHUUTHANH · MACD · RSI · Khối lượng)")
    if not symbol or not symbol.isalnum():
        st.warning("Nhập mã cổ phiếu hợp lệ.")
        return
    start=(date.today()-timedelta(days=days+160)).isoformat()
    try:
        raw,source=fetch_chart_quotes(symbol,start,date.today().isoformat(),interval="1D")
        clean,warning=sanitize_daily_chart_bars(raw)
        if warning: st.warning(warning)
        df=_signals(add_indicators(clean.copy()))
        if len(df)<40:
            st.warning("Cần tối thiểu 40 phiên có dữ liệu để hiển thị chỉ báo.")
            return
        df["EMA10_PRICE"]=df["close"].ewm(span=10,adjust=False).mean()
        df["EMA10_VOLUME"]=df["volume"].ewm(span=10,adjust=False).mean()
        df=df[df["date"]>=pd.Timestamp(date.today()-timedelta(days=days))].copy()
        views={"Giá và EMA":1,"LUCHUUTHANH":2,"MACD":3,"RSI":4,"Khối lượng":5}
        selected=views.get(view)
        panels=[selected] if selected else [1,2,3,4,5]
        def panel(original):
            return panels.index(original)+1 if original in panels else None
        fig=make_subplots(rows=len(panels),cols=1,shared_xaxes=True,vertical_spacing=.025,
            row_heights=([.38,.23,.13,.12,.14] if len(panels)==5 else None))
        def add(trace,original):
            if panel(original) is not None:
                fig.add_trace(trace,row=panel(original),col=1)
        def hline(level,original):
            if panel(original) is not None:
                fig.add_hline(y=level,line_dash="dot",line_color="#64748b",row=panel(original),col=1)
        add(go.Candlestick(x=df.date,open=df.open,high=df.high,low=df.low,close=df.close,name="Nến giá"),1)
        # Ichimoku 9/26/52: spans projected 26 sessions ahead; cloud shaded between spans.
        if st.checkbox("Hiển thị Ichimoku (9, 26, 52)", value=False, key="lct_ichimoku"):
            base=clean.sort_values("date").drop_duplicates("date").copy()
            h9=base.high.rolling(9).max(); l9=base.low.rolling(9).min()
            h26=base.high.rolling(26).max(); l26=base.low.rolling(26).min()
            h52=base.high.rolling(52).max(); l52=base.low.rolling(52).min()
            tenkan=(h9+l9)/2; kijun=(h26+l26)/2
            span_a=(tenkan+kijun)/2; span_b=(h52+l52)/2
            forward=pd.bdate_range(base.date.iloc[-1]+pd.Timedelta(days=1),periods=26)
            future_dates=list(base.date)+list(forward)
            projected=pd.to_datetime(future_dates[26:26+len(base)])
            visible_start=df.date.min()
            cloud=pd.DataFrame({"date":projected,"a":span_a.to_numpy(),"b":span_b.to_numpy()}).dropna()
            cloud=cloud[cloud.date>=visible_start]
            add(go.Scatter(x=cloud.date,y=cloud.a,name="Senkou Span A",mode="lines",line=dict(color="#22c55e",width=1)),1)
            add(go.Scatter(x=cloud.date,y=cloud.b,name="Senkou Span B",mode="lines",line=dict(color="#ef4444",width=1),fill="tonexty",fillcolor="rgba(45,180,120,0.13)"),1)
            add(go.Scatter(x=base.date,y=tenkan,name="Tenkan-sen",mode="lines",line=dict(color="#fb923c",width=1.5)),1)
            add(go.Scatter(x=base.date,y=kijun,name="Kijun-sen",mode="lines",line=dict(color="#e879f9",width=1.5)),1)
            add(go.Scatter(x=base.date.iloc[:-26],y=base.close.iloc[26:].to_numpy(),name="Chikou Span",mode="lines",line=dict(color="#93c5fd",width=1),visible="legendonly"),1)
        for col,color in [("EMA10_PRICE","#f59e0b"),("EMA20","#f472b6"),("EMA50","#22c55e"),("EMA100","#a855f7")]:
            add(go.Scatter(x=df.date,y=df[col],name=col,mode="lines",line=dict(color=color,width=1.5)),1)
        for col,name,color,width in [
            ("LCT_FAST","LUCHUUTHANH nhanh","#16a34a",2),
            ("LCT_YELLOW","Tín hiệu vàng","#eab308",1.6),
            ("LCT_SLOW","Chậm","#dc2626",2),
            ("LCT_UPPER","Dải trắng trên","#f8fafc",1),
            ("LCT_LOWER","Dải trắng dưới","#f8fafc",1)]:
            add(go.Scatter(x=df.date,y=df[col],name=name,mode="lines",line=dict(color=color,width=width)),2)
        for col,name,color,symbol_marker in [
            ("LCT_BUY","Mua thử nghiệm","#22c55e","triangle-up"),
            ("LCT_SELL","Bán thử nghiệm","#ef4444","triangle-down")]:
            pts=df.dropna(subset=[col])
            add(go.Scatter(x=pts.date,y=pts[col],mode="markers",name=name,
                marker=dict(color=color,size=11,symbol=symbol_marker)),2)
        for level in [20,80]:
            hline(level,2)
        add(go.Scatter(x=df.date,y=df.MACD,name="MACD 12,26",line=dict(color="#38bdf8")),3)
        add(go.Scatter(x=df.date,y=df.MACD_SIGNAL,name="Signal 9",line=dict(color="#f59e0b")),3)
        add(go.Bar(x=df.date,y=df.MACD_HIST,name="MACD Histogram",marker_color=np.where(df.MACD_HIST>=0,"#22c55e","#ef4444")),3)
        add(go.Scatter(x=df.date,y=df.RSI14,name="RSI14",line=dict(color="#a78bfa")),4)
        for level in [30,50,70]:
            hline(level,4)
        add(go.Bar(x=df.date,y=df.volume,name="Khối lượng",marker_color=np.where(df.close>=df.open,"#22c55e","#ef4444")),5)
        add(go.Scatter(x=df.date,y=df.EMA10_VOLUME,name="EMA10 khối lượng",line=dict(color="#facc15",width=2)),5)
        for original in (2,4):
            if panel(original) is not None:
                fig.update_yaxes(range=[0,100],row=panel(original),col=1)
        fig.update_layout(height=1050 if len(panels)==5 else 480,template="plotly_dark",xaxis_rangeslider_visible=False,
            legend=dict(orientation="h",y=-.15,font=dict(size=10)),margin=dict(l=4,r=4,t=40,b=90),
            title=f"{symbol} · {view} · D1")
        st.caption(f"Nguồn: {source} · Dữ liệu D1 · Tín hiệu khung sáng chỉ dùng nghiên cứu.")
        st.plotly_chart(fig,use_container_width=True,key="lct_plot")
    except Exception as exc:
        st.error(f"Không tải hoặc dựng được biểu đồ {symbol}: {exc}")
