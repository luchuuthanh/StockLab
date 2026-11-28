import numpy as np
import pandas as pd
from cv02_phase2 import pk
from cophieu68_import import fetch_cophieu68_history

def run(symbol):
    end=pd.Timestamp.today().normalize()
    d=fetch_cophieu68_history(symbol,max_pages=200,start_date=end-pd.DateOffset(years=7)).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    c=pd.to_numeric(d.close); dates=pd.to_datetime(d.date)
    volume=pd.to_numeric(d.volume)
    delta=c.diff()
    r=100-100/(1+delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()/(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean())
    short,es=pk(r,45); long,el=pk(r,252)
    e={n:c.ewm(span=n,adjust=False,min_periods=n).mean() for n in (20,50,100,200)}
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    obv=(np.sign(delta).fillna(0)*volume).cumsum()
    split=end-pd.DateOffset(years=2)
    rows=[]; events=[]
    # Independent outcome: price breaks prior 20-session extreme and holds
    # beyond it for 3 closes. No EMA/RSI in outcome definition.
    for sign in (1,-1):
        prior=(long==-sign)
        near=sign*(c/e[100]-1)>=-.02
        cross=sign*(c-e[100])>0
        slope=sign*e[100].diff(5)>0
        gap=((e[20]-e[100]).abs()+(e[50]-e[100]).abs())/e[100]
        narrow=gap<gap.shift(5)
        wave=sign*(e[20]-e[50])>0
        wave_cross=wave & ~wave.shift(1,fill_value=False)
        recent=lambda x:pd.Series(x,index=d.index).eq(sign).rolling(5,min_periods=1).max().astype(bool)
        flags={'near':near,'cross':cross,'slope':slope,'narrow':narrow,'wave_cross':wave_cross.rolling(5,min_periods=1).max().astype(bool),'pk_short':recent(es),'pk_long':recent(el),'macd':sign*hist.diff(3)>0,'obv':sign*obv.diff(5)>0,'ema200':sign*(c-e[200])>0}
        combos={'near':['near'],'cross':['cross']}
        for k in flags:
            if k not in ('near','cross'):combos['cross+'+k]=['cross',k];combos['near+'+k]=['near',k]
        combos.update({'cross+momentum':['cross','macd','obv'],'near+wave+pk':['near','narrow','pk_short'],'cross+pk+ema200':['cross','pk_long','ema200']})
        extreme=(c.rolling(20).max() if sign==1 else c.rolling(20).min()).shift(1)
        for name,keys in combos.items():
            signal=prior & e[200].notna()
            # Long PK break may itself flip prior state; use previous confirmed trend.
            if 'pk_long' in keys: signal=(long.shift(5)==-sign)&e[200].notna()
            for k in keys:signal &= flags[k]
            for persistence in (1,2,3,4,5):
                active=signal.rolling(persistence,min_periods=persistence).sum().eq(persistence)
                starts=active & ~active.shift(1,fill_value=False)
                last=-100
                for i in np.flatnonzero(starts):
                    if i-last<10 or i+10>=len(c) or dates.iloc[i]<end-pd.DateOffset(years=5):continue
                    period='development' if dates.iloc[i]<split else 'holdout'
                    if period=='development' and dates.iloc[i+10]>=split:continue
                    last=i; onset=i-persistence+1; level=extreme.iloc[onset]; delay=0
                    for j in range(onset+1,i+9):
                        if (sign*(c.iloc[j:j+3]-level)>0).all():
                            delay=j+2-i;break
                    events.append(dict(symbol=symbol,side=sign,combo=name,persistence=persistence,period=period,date=dates.iloc[i],onset=dates.iloc[onset],delay=delay,already_confirmed=bool(delay<=0 and any((sign*(c.iloc[j:j+3]-level)>0).all() for j in range(onset+1,i-1))),total_delay=delay+persistence-1 if delay>0 else np.nan,ret10=sign*(c.iloc[i+10]/c.iloc[i]-1)))
    ev=pd.DataFrame(events)
    for keys,g in ev.groupby(['symbol','side','combo','persistence','period']):
        for h in range(3,11):
            hit=g.delay.between(1,h); n=len(g); p=hit.mean(); z=1.96
            lower=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
            rows.append(dict(zip(['symbol','side','combo','persistence','period'],keys),h=h,n=n,success=p,false=1-p,lower95=lower,delay=g.loc[hit,'delay'].mean(),total_delay=g.loc[hit,'total_delay'].mean(),already_confirmed=int(g.already_confirmed.sum()),success_from_onset=(hit & g.total_delay.le(h)).mean(),ret10=g.ret10.mean()))
    return ev,pd.DataFrame(rows)

if __name__=='__main__':
    a,b=zip(*(run(s) for s in ('GAS','PVD')))
    pd.concat(a).to_csv('cv02_early5_events.csv',index=False)
    pd.concat(b).to_csv('cv02_early5_summary.csv',index=False)
