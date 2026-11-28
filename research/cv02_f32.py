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
    flat=(c.rolling(10).max()/c.rolling(10).min()-1<=.05)&((e[20]/e[50]-1).abs()<=.03)
    main=states['long',1]
    prior=main.where(~flat).ffill().shift(1).fillna(0)
    rows=[]
    for scale,limit,h in [('short',3,10),('medium',7,30),('long',9,60)]:
      for n in range(1,limit+1):
       for side,levels in [('low',(20,25,32)),('high',(70,75,80))]:
        for level in levels:
         inside=k.lt(level) if side=='low' else k.gt(level)
         duration=inside.astype(int).groupby((~inside).cumsum()).cumsum()
         for phase,flag in [('enter',inside&~inside.shift(1,fill_value=False)),('stay',duration.eq(n)),('exit',~inside&inside.shift(1,fill_value=False))]:
          for variant in ('zone','zone_pk','zone_momentum','zone_ema','all'):
           last=-h
           for i in np.flatnonzero(flag):
            if i<n or i+h>=len(c) or i-last<h or dates.iloc[i]<start or not np.isfinite(e[200].iloc[i]):continue
            p=int(prior.iloc[i])
            if p==0:continue
            period='development' if dates.iloc[i]<split else 'holdout'
            if period=='development' and dates.iloc[i+h]>=split:continue
            # Candidate reversal direction is opposite the preceding main trend.
            direction=-p
            pkok=states[scale,n].iloc[i]==direction
            momentum=(direction*hist.diff(3).iloc[i]>0 and direction*obv.diff(5).iloc[i]>0)
            emaok=direction*(c.iloc[i]-e[100].iloc[i])>0 and direction*(e[20].iloc[i]-e[50].iloc[i])>0
            checks={'zone':True,'zone_pk':pkok,'zone_momentum':momentum,'zone_ema':emaok,'all':pkok and momentum and emaok}
            if not checks[variant]:continue
            # Fixed past 20-session price range, independent of oscillator/EMA labels.
            past=c.iloc[max(0,i-19):i+1]; upper=past.max(); lower=past.min()
            future=c.iloc[i+1:i+h+1]
            up=(future>upper).rolling(3).sum().eq(3)
            down=(future<lower).rolling(3).sum().eq(3)
            # Require persistence at horizon end; otherwise unresolved.
            outcome=1 if up.any() and (future.iloc[-3:]>upper).all() else -1 if down.any() and (future.iloc[-3:]<lower).all() else 0
            context=('flat_after_down' if p<0 else 'flat_after_up') if flat.iloc[i] else ('downtrend' if p<0 else 'uptrend')
            # Prediction fixed at observation; no future input.
            prediction=direction if (pkok or momentum) and emaok else p
            rows.append(dict(symbol=symbol,period=period,date=dates.iloc[i],scale=scale,n=n,horizon=h,side=side,level=level,phase=phase,variant=variant,context=context,prior=p,prediction=prediction,outcome=outcome,reversal=int(outcome==direction),continuation=int(outcome==p),unresolved=int(outcome==0),correct=int(outcome==prediction),wrong=int(outcome==-prediction),ema200_confirm=int(direction*(c.iloc[i]-e[200].iloc[i])>0)))
            last=i
    return pd.DataFrame(rows)

if __name__=='__main__':
    ev=pd.concat([run(s) for s in ('GAS','PVD')],ignore_index=True)
    ev.to_csv('cv02_f32_events.csv',index=False)
    keys=['symbol','period','scale','n','horizon','side','level','phase','variant','context']
    out=ev.groupby(keys).agg(samples=('correct','size'),correct=('correct','sum'),wrong=('wrong','sum'),unresolved=('unresolved','sum'),reversal=('reversal','sum'),continuation=('continuation','sum')).reset_index()
    for col in ('correct','wrong','unresolved','reversal','continuation'):
        out[col+'_rate']=out[col]/out.samples
    p=out.correct_rate; n=out.samples; z=1.96
    out['wilson_lower']=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    out.to_csv('cv02_f32_summary.csv',index=False)
    params=[x for x in keys if x!='period']
    dev=out[(out.period=='development')&(out.samples>=5)].sort_values(['wilson_lower','n'],ascending=[False,True])
    chosen=dev.groupby(['symbol','scale','context']).head(1)
    out.merge(chosen[params],on=params).to_csv('cv02_f32_holdout.csv',index=False)
