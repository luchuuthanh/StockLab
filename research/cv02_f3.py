import numpy as np
import pandas as pd
from cv02_phase2 import pk
from cophieu68_import import fetch_cophieu68_history

def run(symbol):
    end=pd.Timestamp.today().normalize()
    d=fetch_cophieu68_history(symbol,max_pages=200,start_date=end-pd.DateOffset(years=7)).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    c=pd.to_numeric(d.close); v=pd.to_numeric(d.volume); dates=pd.to_datetime(d.date)
    delta=c.diff()
    r=100-100/(1+delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()/(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean())
    e={n:c.ewm(span=n,adjust=False,min_periods=n).mean() for n in (20,50,100,200)}
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    obv=(np.sign(delta).fillna(0)*v).cumsum()
    split=end-pd.DateOffset(years=2); start=end-pd.DateOffset(years=5)
    events=[]; states={}; trades=[]
    for label,w,limit,ema,horizon in [('short',32,3,20,10),('medium',90,7,100,30),('long',252,9,200,60)]:
        trend,_=pk(r,w)
        width=c.rolling(10).max()/c.rolling(10).min()-1
        flat=(width<=.05)&((e[20]/e[50]-1).abs()<=.03)
        context=trend.where(~flat).ffill().fillna(0)
        score=np.sign(c-e[ema])+np.sign(hist.diff(3))+np.sign(obv.diff(5))+np.sign(r.diff(3))
        for n in range(1,limit+1):
            state=pd.Series(0,index=c.index)
            for direction in (-1,1):
                ok=(direction*score>=2)&(direction*trend>0)
                state.loc[ok.rolling(n,min_periods=n).sum().eq(n)]=direction
            states[label,n]=state.replace(0,np.nan).ffill().fillna(0)
            for variant,condition in [('flat',flat),('flat_volume',flat&(v.rolling(5).mean()<v.rolling(20).mean())),('flat_momentum',flat&(score.abs()>=2))]:
                active=condition.rolling(n,min_periods=n).sum().eq(n)
                last=-100
                for i in np.flatnonzero(active&~active.shift(1,fill_value=False)):
                    if i-last<horizon or i+horizon>=len(c) or dates.iloc[i]<start or not np.isfinite(e[200].iloc[i]):continue
                    onset=i-n+1; prior=int(context.iloc[max(0,onset-1)])
                    if prior==0:continue
                    period='development' if dates.iloc[i]<split else 'holdout'
                    if period=='development' and dates.iloc[i+horizon]>=split:continue
                    last=i; hi=c.iloc[onset-9:i+1].max(); lo=c.iloc[onset-9:i+1].min()
                    direction=0; delay=0
                    for j in range(i+1,i+horizon-1):
                        q=c.iloc[j:j+3]
                        if (q>hi).all():direction=1
                        elif (q<lo).all():direction=-1
                        if direction:delay=j+2-i;break
                    events.append(dict(symbol=symbol,scale=label,n=n,variant=variant,date=dates.iloc[i],period=period,prior=prior,outcome='reversal' if direction==-prior else 'continuation' if direction==prior else 'flat',kind=('bottom' if prior<0 else 'top') if direction==-prior else ('up_pause' if prior>0 else 'down_pause') if direction else 'unresolved',delay=delay,total_delay=delay+n-1 if delay else np.nan,prediction=int(np.sign(score.iloc[i]))))
    for ns in range(1,4):
        for nm in range(1,8):
            for nl in range(1,10):
                s,m,l=[states[k,n] for k,n in [('short',ns),('medium',nm),('long',nl)]]
                target=np.select([(m<0)&(l<0),(s>0)&(m>0)&(l>0),(s>0)&(m>0),(m>0)&(l>0)],[0,1,.5,.5],default=np.nan)
                pos=pd.Series(target,index=c.index).ffill().fillna(0).shift(1).fillna(0)
                # close-to-close proxy with one-bar lag, not executable T+2 simulation
                turnover=pos.diff().abs().fillna(0)
                ret=pos.shift(1).fillna(0)*c.pct_change().fillna(0)-.002*turnover
                for period,mask in [('development',(dates>=start)&(dates<split)),('holdout',dates>=split)]:
                    rr=ret[mask]; equity=(1+rr).cumprod()
                    change=pos.diff()[mask]
                    trades.append(dict(symbol=symbol,period=period,ns=ns,nm=nm,nl=nl,return_net=equity.iloc[-1]-1,drawdown=(equity/equity.cummax()-1).min(),positive_days=(rr[rr!=0]>0).mean(),entries=int(((change>0)&(pos.shift(1)[mask]==0)).sum()),adds=int(((change>0)&(pos.shift(1)[mask]>0)).sum()),exits=int(((change<0)&(pos[mask]==0)).sum())))
    return pd.DataFrame(events),pd.DataFrame(trades)
if __name__=='__main__':
    a,b=zip(*(run(s) for s in ('GAS','PVD')))
    ev=pd.concat(a); ev.to_csv('cv02_f3_events.csv',index=False)
    pd.concat(b).to_csv('cv02_f3_positions.csv',index=False)
    rows=[]
    for keys,g in ev.groupby(['symbol','scale','n','variant','period','prior']):
        for h in range(1,10):
            rev=(g.outcome=='reversal')&g.delay.between(1,h); cont=(g.outcome=='continuation')&g.delay.between(1,h)
            p=rev.mean(); n=len(g); z=1.96
            rows.append(dict(zip(['symbol','scale','n','variant','period','prior'],keys),h=h,samples=n,reversal=p,continuation=cont.mean(),unresolved=1-p-cont.mean(),wilson=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)))
    pd.DataFrame(rows).to_csv('cv02_f3_summary.csv',index=False)
