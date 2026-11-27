import pandas as pd
import numpy as np
from cophieu68_import import fetch_cophieu68_history

def evaluate(symbol):
    end=pd.Timestamp.today().normalize()
    start=end-pd.DateOffset(years=7)
    d=fetch_cophieu68_history(symbol,max_pages=200,start_date=start).sort_values('date').drop_duplicates('date')
    c=pd.to_numeric(d.close)
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    r=100-100/(1+gain/loss)
    e={n:c.ewm(span=n,adjust=False,min_periods=n).mean() for n in (20,50,100,200)}
    # Preliminary causal RSI slope baseline, not the final pivot-line algorithm.
    slope=r.rolling(30).apply(lambda x: np.polyfit(np.arange(len(x)),x,1)[0],raw=True)
    long=r.rolling(120).apply(lambda x: np.polyfit(np.arange(len(x)),x,1)[0],raw=True)
    rs=np.sign(long.where(long.abs()>.03,0))
    votes={n:np.sign(c-e[n]) for n in e}
    scores={'A':rs,'B':rs+votes[20]+votes[50],'C':rs+votes[100]+votes[200],'D':rs+sum(votes.values()),'E':rs+sum(votes.values())+sum(np.sign(e[n].diff(10)) for n in e)+np.sign(e[20]-e[50])+np.sign(e[50]-e[100])+np.sign(e[100]-e[200])}
    out=[]
    for name,score in scores.items():
        threshold={'A':1,'B':2,'C':2,'D':3,'E':7}[name]
        state=pd.Series(np.select([score>=threshold,score<=-threshold],[1,-1],default=0),index=d.index).where(e[200].notna() & long.notna())
        mask=(pd.to_datetime(d.date)>=end-pd.DateOffset(years=5)) & state.notna()
        wave=np.where((state==1)&(slope<0),'T2',np.where((state==-1)&(slope>0),'G2',np.where(state==1,'T1',np.where(state==-1,'G1','L'))))
        for h in (5,10,20,30):
            ret=c.shift(-h)/c-1
            for k in (-1,0,1):
                m=mask & (state==k) & ret.notna()
                v=ret[m]
                out.append(dict(symbol=symbol,method=name,horizon=h,trend=k,n=len(v),mean_return=v.mean(),direction_agreement=((v*k)>0).mean() if k else (v.abs()<=.05).mean(),switches=int((state.diff().ne(0)&mask).sum())))
        pd.DataFrame({'date':d.date,'close':c,'rsi':r,'state':state,'wave':wave,**{'ema'+str(n):v for n,v in e.items()}})[mask].to_csv(f'cv02_{symbol}_{name}.csv',index=False)
    return out

if __name__=='__main__':
    rows=[]
    for symbol in ('GAS','PVD'):
        rows.extend(evaluate(symbol))
    pd.DataFrame(rows).to_csv('cv02_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
