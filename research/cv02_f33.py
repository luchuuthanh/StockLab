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
    main=states['long',1]; prior=main.where(~flat).ffill().shift(1).fillna(0)
    rows=[]
    for side,levels in [('low',(20,25,32)),('high',(70,75,80))]:
      for level in levels:
        inside=k.lt(level) if side=='low' else k.gt(level)
        entry=inside&~inside.shift(1,fill_value=False)
        exit=~inside&inside.shift(1,fill_value=False)
        for onset in np.flatnonzero(entry):
          if onset<20 or dates.iloc[onset]<start or not np.isfinite(e[200].iloc[onset]):continue
          p=int(prior.iloc[onset])
          if not p:continue
          direction=-p
          context=('flat_after_down' if p<0 else 'flat_after_up') if flat.iloc[onset] else ('downtrend' if p<0 else 'uptrend')
          exits=np.flatnonzero(exit.iloc[onset:].to_numpy())
          endzone=onset+int(exits[0]) if len(exits) else len(c)
          upper=c.iloc[onset-20:onset].max(); lower=c.iloc[onset-20:onset].min()
          anchor=c.iloc[onset]
          for scale,limit in [('short',3),('medium',7),('long',9)]:
           for variant in ('zone','pk','momentum','ema100','ema200','pk_momentum','all100','all200','all'):
            for phase in ('enter','stay','exit'):
             for n in range(1,limit+1):
              i=onset+n-1 if phase!='exit' else endzone+n-1
              if i>=len(c) or (phase=='stay' and i>=endzone) or (phase=='enter' and n!=1):continue
              pkok=states[scale,n].iloc[i]==direction
              mom=direction*hist.diff(3).iloc[i]>0 and direction*obv.diff(5).iloc[i]>0
              a=direction*(c.iloc[i]-e[100].iloc[i])>0
              b=direction*(c.iloc[i]-e[200].iloc[i])>0
              wave=direction*(e[20].iloc[i]-e[50].iloc[i])>0
              checks={'zone':False,'pk':pkok,'momentum':mom,'ema100':a,'ema200':b,'pk_momentum':pkok and mom,'all100':pkok and mom and a and wave,'all200':pkok and mom and b and wave,'all':pkok and mom and a and b and wave}
              pred='reversal' if checks[variant] else 'counterwave' if (pkok or mom or wave) and not context.startswith('flat') else 'continuation'
              period='development' if dates.iloc[onset]<split else 'holdout'
              if period=='development' and dates.iloc[i]>=split:continue
              for h in (5,7,10,20,30,60):
               if i+h>=len(c) or (period=='development' and dates.iloc[i+h]>=split):continue
               future=c.iloc[i+1:i+h+1]
               # Independent price structure proxy: two successive blocks of higher highs/lows or lower highs/lows.
               q=max(2,h//3); first=future.iloc[:q]; last=future.iloc[-q:]
               structural=(last.max()>first.max() and last.min()>first.min()) if direction>0 else (last.max()<first.max() and last.min()<first.min())
               beyond=(future>upper) if direction>0 else (future<lower)
               persistent=bool(beyond.rolling(3).sum().eq(3).any() and beyond.iloc[-3:].all())
               excursion=float((direction*(future/anchor-1)).max())
               continuation=bool((p*(future.iloc[-1]/anchor-1)>=.03) and ((future.iloc[-3:]<lower).all() if p<0 else (future.iloc[-3:]>upper).all()))
               if persistent and structural:actual='reversal'
               elif context.startswith('flat'):actual='continuation' if continuation else 'unresolved'
               elif excursion>=.03 and continuation:actual='counterwave'
               elif excursion<.03 and continuation:actual='continuation'
               else:actual='unresolved'
               rows.append(dict(symbol=symbol,period=period,onset=dates.iloc[onset],date=dates.iloc[i],delay=i-onset,scale=scale,n=n,phase=phase,side=side,level=level,variant=variant,context=context,horizon=h,prediction=pred,actual=actual,correct=int(pred==actual),wrong=int(actual!='unresolved' and pred!=actual),unresolved=int(actual=='unresolved')))
    return pd.DataFrame(rows)

if __name__=='__main__':
    ev=pd.concat([run(s) for s in ('GAS','PVD')],ignore_index=True)
    ev.to_csv('cv02_f33_events.csv',index=False)
    keys=['symbol','period','scale','n','phase','side','level','variant','context','horizon']
    out=ev.groupby(keys).agg(samples=('correct','size'),correct=('correct','sum'),wrong=('wrong','sum'),unresolved=('unresolved','sum'),delay=('delay','mean')).reset_index()
    for col in ('correct','wrong','unresolved'):out[col+'_rate']=out[col]/out.samples
    p=out.correct_rate; n=out.samples; z=1.96
    out['wilson_lower']=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    out.to_csv('cv02_f33_summary.csv',index=False)
    ev.groupby(keys+['prediction','actual']).size().rename('samples').reset_index().to_csv('cv02_f33_confusion.csv',index=False)
    params=[x for x in keys if x!='period']
    chosen=out[(out.period=='development')&(out.samples>=10)].sort_values(['wilson_lower','delay'],ascending=[False,True]).groupby(['symbol','scale','context','horizon']).head(1)
    out.merge(chosen[params],on=params).to_csv('cv02_f33_holdout.csv',index=False)
