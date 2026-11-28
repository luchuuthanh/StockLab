import numpy as np
import pandas as pd
from cv02_phase2 import pk
from lightweight_chart import calculate_stochastic_trial
from cophieu68_import import fetch_cophieu68_history

def run(symbol):
    end=pd.Timestamp.today().normalize()
    d=fetch_cophieu68_history(symbol,max_pages=200,start_date=end-pd.DateOffset(years=7)).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    c=pd.to_numeric(d.close); v=pd.to_numeric(d.volume); dates=pd.to_datetime(d.date)
    osc=calculate_stochastic_trial(d); k=osc['EMA 3']; red=osc['EMA 9']
    delta=c.diff()
    r=100-100/(1+delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()/(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean())
    e={n:c.ewm(span=n,adjust=False,min_periods=n).mean() for n in (20,50,100,200)}
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    obv=(np.sign(delta).fillna(0)*v).cumsum()
    split=end-pd.DateOffset(years=2); start=end-pd.DateOffset(years=5)
    states={}; events=[]; results=[]
    for label,w,limit,ema in [('short',32,3,20),('medium',90,7,100),('long',252,9,200)]:
        trend,_=pk(r,w)
        score=np.sign(c-e[ema])+np.sign(hist.diff(3))+np.sign(obv.diff(5))+np.sign(r.diff(3))
        for n in range(1,limit+1):
            state=pd.Series(0,index=c.index)
            for direction in (-1,1):
                ok=(direction*score>=2)&(direction*trend>0)
                state.loc[ok.rolling(n,min_periods=n).sum().eq(n)]=direction
            states[label,n]=state.replace(0,np.nan).ffill().fillna(0)
    masks={'development':(dates>=start)&(dates<split),'holdout':dates>=split}
    flat=(c.rolling(10).max()/c.rolling(10).min()-1<=.05)&((e[20]/e[50]-1).abs()<=.03)
    context=states['long',1].where(~flat).ffill().fillna(0)
    for side,levels in [('low',(20,25,32)),('high',(70,75,80))]:
        for level in levels:
            inside=k.lt(level) if side=='low' else k.gt(level)
            entry=inside&~inside.shift(1,fill_value=False); exit=~inside&inside.shift(1,fill_value=False)
            duration=inside.astype(int).groupby((~inside).cumsum()).cumsum()
            for event,flag in [('enter',entry),('exit',exit),('stay',inside)]:
                for i in np.flatnonzero(flag):
                    if dates.iloc[i]<start or i+10>=len(c) or not np.isfinite(e[200].iloc[i]):continue
                    period='development' if dates.iloc[i]<split else 'holdout'
                    if period=='development' and dates.iloc[i+10]>=split:continue
                    prior=int(context.iloc[i]); direction=1 if side=='low' else -1
                    events.append(dict(symbol=symbol,period=period,date=dates.iloc[i],side=side,level=level,event=event,duration=int(duration.iloc[i-1] if event=='exit' else duration.iloc[i]),trend=prior,flat=bool(flat.iloc[i]),context=('accumulation_candidate' if prior<0 else 'distribution_candidate') if flat.iloc[i] else ('correction_candidate' if prior>0 and side=='low' else 'recovery_candidate' if prior<0 and side=='high' else 'other'),ret3=direction*(c.iloc[i+3]/c.iloc[i]-1),ret7=direction*(c.iloc[i+7]/c.iloc[i]-1),ret10=direction*(c.iloc[i+10]/c.iloc[i]-1)))
    for low in (20,25,32):
      for high in (70,75,80):
        rebound=k.ge(low)&k.shift().lt(low)
        fade=k.le(high)&k.shift().gt(high)
        recent_up=rebound.rolling(3,min_periods=1).max().astype(bool)
        recent_down=fade.rolling(3,min_periods=1).max().astype(bool)
        for ns in range(1,4):
          for nm in range(1,8):
            for nl in range(1,10):
                s,m,l=[states[a,n] for a,n in [('short',ns),('medium',nm),('long',nl)]]
                buy=recent_up&(k>red)&(s>0)&(m>0)&(c>e[20])&(hist.diff(3)>0)&(obv.diff(5)>0)
                add=buy&(l>0)&(c>e[50])&(c>e[100])
                sell=(m<0)&(l<0) | (recent_down&(k<red)&(c<e[100])&(hist.diff(3)<0)&(obv.diff(5)<0)&(m<0))
                reduce=recent_down&(s<0)&(c<e[20])&(c<e[50])
                target=np.select([sell,add,buy,reduce],[0,1,.5,.5],default=np.nan)
                # Reduction must not open a position; additions only after an initial entry.
                p=0.; path=[]
                for t,a,b,z in zip(target,add,buy,reduce):
                    if t==0:p=0.
                    elif a:p=1. if p>0 else .5
                    elif b:p=max(p,.5)
                    elif z:p=min(p,.5)
                    path.append(p)
                pos=pd.Series(path,index=c.index).shift(1).fillna(0)
                change=pos.diff().fillna(0)
                ret=pos.shift(1).fillna(0)*c.pct_change().fillna(0)-.002*change.abs()
                for period,mask in masks.items():
                    rr=ret[mask]; eq=(1+rr).cumprod(); ch=change[mask]; prev=pos.shift(1)[mask]
                    # Closed position episodes, excluding carry-in and unfinished episodes.
                    wins=[]; active=False; wealth=1.
                    for a,b,x in zip(prev,pos[mask],rr):
                        if a==0 and b>0:active=True;wealth=1.
                        if active:wealth*=1+x
                        if active and b==0:wins.append(wealth-1);active=False
                    results.append(dict(symbol=symbol,period=period,low=low,high=high,ns=ns,nm=nm,nl=nl,return_net=eq.iloc[-1]-1,drawdown=(eq/eq.cummax()-1).min(),positive_days=(rr[rr!=0]>0).mean(),closed_trades=len(wins),trade_win=np.mean(np.array(wins)>0) if wins else np.nan,entries=int(((ch>0)&(prev==0)).sum()),adds=int(((ch>0)&(prev>0)).sum()),exits=int(((ch<0)&(pos[mask]==0)).sum())))
    return pd.DataFrame(events),pd.DataFrame(results)
if __name__=='__main__':
    a,b=zip(*(run(s) for s in ('GAS','PVD')))
    ev=pd.concat(a); ev.to_csv('cv02_f31_events.csv',index=False)
    pos=pd.concat(b); pos.to_csv('cv02_f31_positions.csv',index=False)
    keys=['symbol','period','side','level','event','trend','flat','context']
    summary=ev.groupby(keys).agg(samples=('ret10','size'),win3=('ret3',lambda x:(x>0).mean()),win7=('ret7',lambda x:(x>0).mean()),win10=('ret10',lambda x:(x>0).mean()),mean10=('ret10','mean'),duration=('duration','mean')).reset_index()
    summary.to_csv('cv02_f31_summary.csv',index=False)
    params=['symbol','low','high','ns','nm','nl']
    dev=pos[pos.period=='development'].copy()
    dev['score']=dev.return_net+dev.drawdown
    chosen=dev.sort_values('score',ascending=False).groupby('symbol').head(1)
    compare=pos.merge(chosen[params],on=params)
    from pathlib import Path
    base=Path('research/results/cv02_f3_positions.csv')
    if base.exists():
        old=pd.read_csv(base)
        compare=compare.merge(old,on=['symbol','period','ns','nm','nl'],how='left',suffixes=('','_f3'))
    compare.to_csv('cv02_f31_comparison.csv',index=False)
