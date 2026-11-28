import numpy as np
import pandas as pd
from cophieu68_import import fetch_cophieu68_history

def pk(r,window,tol=4):
    a=r.to_numpy(); n=len(a); hi=[]; lo=[]; out=np.zeros(n); events=np.zeros(n); pending=None; state=0
    for i in range(n):
        j=i-3
        if j>=3 and np.isfinite(a[j-3:i+1]).all():
            v=a[j-3:i+1]
            if a[j]==max(v) and np.argmax(v)==3: hi.append(j)
            if a[j]==min(v) and np.argmin(v)==3: lo.append(j)
        def line(p,sign):
            q=[k for k in p if k>=i-window]
            if len(q)<2:return np.nan
            x,y=q[-2:]
            m=(a[y]-a[x])/(y-x)
            return a[y]+m*(i-y) if m*sign>0 else np.nan
        upper=line(hi,-1); lower=line(lo,1)
        if pending:
            start,direction,level,slope=pending
            boundary=level+slope*(i-start)
            if not np.isfinite(a[i]) or direction*(a[i]-boundary)<-tol: pending=None
            elif i-start==3:
                if direction*(a[i]-boundary)>tol: state=direction; events[i]=direction
                pending=None
        if pending is None and i>0:
            for direction,b,p in [(1,upper,hi),(-1,lower,lo)]:
                if np.isfinite(b) and direction*(a[i]-b)>tol and direction!=state:
                    x,y=p[-2:]; m=(a[y]-a[x])/(y-x)
                    if direction*(a[i-1]-(b-m))<=tol:
                        pending=(i,direction,b,m); break
        out[i]=state
    return pd.Series(out,index=r.index),events

def run(symbol):
    end=pd.Timestamp.today().normalize()
    d=fetch_cophieu68_history(symbol,max_pages=200,start_date=end-pd.DateOffset(years=7)).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    c=pd.to_numeric(d.close); delta=c.diff()
    g=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    l=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    r=100-100/(1+g/l)
    e={n:c.ewm(span=n,adjust=False,min_periods=n).mean() for n in (20,50,100,200)}
    short,es=pk(r,45); long,el=pk(r,252)
    votes={n:np.sign(c-e[n]) for n in e}
    base=np.sign(r.rolling(120).apply(lambda x:np.polyfit(np.arange(len(x)),x,1)[0],raw=True).where(lambda x:x.abs()>.03,0))
    results=[]
    for family,rs in [('baseline',base),('PK',long)]:
        scores={'A':rs,'B':rs+votes[20]+votes[50],'C':rs+votes[100]+votes[200],'D':rs+sum(votes.values()),'E':rs+sum(votes.values())+sum(np.sign(e[n].diff(10)) for n in e)+np.sign(e[20]-e[50])+np.sign(e[50]-e[100])+np.sign(e[100]-e[200])}
        for name,score in scores.items():
            t={'A':1,'B':2,'C':2,'D':3,'E':7}[name]
            raw=pd.Series(np.select([score>=t,score<=-t],[1,-1],default=0),index=d.index)
            state=raw.copy()
            if family=='PK':
                for i in range(3,len(raw)):
                    state.iloc[i]=raw.iloc[i] if raw.iloc[i-3:i+1].eq(raw.iloc[i]).all() else state.iloc[i-1]
            wave=np.select([(state==1)&(short==-1),(state==-1)&(short==1),state==1,state==-1],['T2','G2','T1','G1'],default='L')
            dates=pd.to_datetime(d.date)
            valid=e[200].notna() & (dates>=end-pd.DateOffset(years=5))
            pd.DataFrame(dict(date=dates,close=c,rsi=r,short=short,long=long,trend=state,wave=wave,break_short=es,break_long=el))[valid].to_csv(f'cv02_{symbol}_{family}_{name}.csv',index=False)
            split=end-pd.DateOffset(years=2)
            for period,m in [('development',dates<split),('holdout',dates>=split)]:
                for h in (5,10,20,30):
                    ret=c.shift(-h)/c-1; future=dates.shift(-h)
                    for k in (-1,0,1):
                        mask=valid&m&(state==k)&ret.notna()
                        if period=='development':mask &= future<split
                        v=ret[mask]
                        results.append(dict(symbol=symbol,family=family,method=name,period=period,h=h,trend=k,n=len(v),mean=v.mean(),agreement=((v*k)>0).mean() if k else (v.abs()<=.05).mean(),switches=int((valid&m&state.ne(state.shift())).sum())))
    return results
if __name__=='__main__':
    rows=[]
    for symbol in ('GAS','PVD'):rows.extend(run(symbol))
    df=pd.DataFrame(rows);df.to_csv('cv02_phase2_summary.csv',index=False)
    print(df.to_string(index=False))
