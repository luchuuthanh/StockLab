"""CV03 research-only causal daily alerts; no orders or web integration."""
import numpy as np
import pandas as pd

def features(data):
    d=data.copy().sort_index()
    if not d.index.is_unique: raise ValueError("duplicate sessions")
    c=d.close.astype(float); v=d.volume.astype(float)
    for n in (20,30): d[f"sma{n}"]=c.rolling(n).mean()
    d["ema7"]=c.ewm(span=7,adjust=False,min_periods=7).mean()
    d["macd"]=c.ewm(span=12,adjust=False,min_periods=12).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
    d["macd_pct"]=100*d.macd/c
    delta=c.diff()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["rsi"]=100-100/(1+up/down)
    d.loc[(up==0)&(down==0),"rsi"]=50
    d["obv"]=(np.sign(delta).fillna(0)*v).cumsum()
    d["obv30"]=d.obv.rolling(30).mean()
    d["slope30"]=d.sma30.pct_change(5)
    for n in (20,25,30):
        d[f"range{n}"]=100*(d.high.rolling(n).max()/d.low.rolling(n).min()-1)
    return d

def alerts(data,near=3,band=10,hot=8):
    d=features(data)
    c=d.close
    near20=(100*(c/d.sma20-1)).abs()<=near
    near30=(100*(c/d.sma30-1)).abs()<=near
    cross=(d.macd>0)&(d.macd.shift(1)<=0)
    down=(d.ema7<d.sma30)&(d.ema7.shift(1)>=d.sma30.shift(1))
    # Pivot confirmed two sessions later; no retrospective alert.
    peaks=[]; rows=[]; start=None; waiting=False; active=False
    for i in range(len(d)):
        r=d.iloc[i]
        if i<30: continue
        pk=[]
        j=i-2
        if d.high.iloc[j]>=d.high.iloc[j-2:j+3].max():
            for k in peaks:
                gap=j-k
                if 3<=gap<=60 and d.high.iloc[j]>d.high.iloc[k]:
                    if d.rsi.iloc[j]<d.rsi.iloc[k]: pk.append("RSI_short" if gap<10 else "RSI_long" if gap>10 else "RSI_10")
                    if d.macd.iloc[j]<d.macd.iloc[k]: pk.append("MACD")
            peaks.append(j); peaks=[k for k in peaks if j-k<=60]
        obv_ok=r.obv>=r.obv30
        stable=40<=r.rsi<=60
        dist=100*(r.close/r.sma30-1)
        flags=[]
        if cross.iloc[i]:
            if dist>hot:
                waiting=True; flags.append("tang_nong")
            elif stable and (near20.iloc[i] or near30.iloc[i]):
                start=i; flags.append("bat_dau_gom")
        if waiting:
            if near30.iloc[i]:
                waiting=False
                if stable and obv_ok and r.slope30>=0:
                    start=i; flags.append("tai_gom")
                else: flags.append("gom_that_bai")
            else: flags.append("cho_sma30")
        if start is not None:
            age=i-start+1
            if r.macd<0 and r.close<r.sma30 and not obv_ok:
                flags.append("gom_that_bai"); start=None
            elif age>=20 and age<=30 and stable and r.range20<=band:
                flags.append("gom_tich_luy")
                if age>=25 and r.range25<=band: flags.append("canh_bao_song_lon")
            elif age>30: start=None
        if r.macd>0 and obv_ok and r.close>r.sma30:
            active=True; flags.append("song_duy_tri")
        if active and down.iloc[i]:
            flags.append("song_dieu_chinh" if r.macd>0 or obv_ok else "vung_ban_xac_nhan")
        if active and any(p.startswith("RSI") for p in pk):
            flags.append("canh_bao_vung_ban")
        if active and ("MACD" in pk and any(p.startswith("RSI") for p in pk) or (r.macd<0 and not obv_ok and r.ema7<r.sma30)):
            flags.append("vung_ban_xac_nhan")
        if "vung_ban_xac_nhan" in flags: active=False
        for f in dict.fromkeys(flags):
            rows.append(dict(date=d.index[i],state=f,close=r.close,macd=r.macd,macd_pct=r.macd_pct,rsi=r.rsi,obv=r.obv,obv30=r.obv30,sma20=r.sma20,sma30=r.sma30,ema7=r.ema7,slope30=r.slope30,distance30=dist,range20=r.range20,range25=r.range25,divergence=";".join(sorted(set(pk))),age=i-start+1 if start is not None else None))
    return pd.DataFrame(rows)

def evaluate(data,events,horizons=(1,3,5,7,10,20,25,30,60)):
    """Descriptive forward returns, not classification accuracy or trade returns."""
    c=data.sort_index().close
    rows=[]
    if events.empty: return pd.DataFrame()
    for state,g in events.groupby("state"):
        for h in horizons:
            # Nonoverlapping windows per state; no future data in event generation.
            last=-h
            for _,e in g.sort_values("date").iterrows():
                i=c.index.get_loc(e.date)
                if i-last<h: continue
                last=i
                ret=100*(c.iloc[i+h]/c.iloc[i]-1) if i+h<len(c) else np.nan
                rows.append(dict(state=state,date=e.date,horizon=h,return_pct=ret))
    return pd.DataFrame(rows)
