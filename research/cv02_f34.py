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
    prior=states['long',1].where(~flat).ffill().shift(1).fillna(0)
    high=pd.to_numeric(d.high); low=pd.to_numeric(d.low)
    # Pivots become observable only three sessions after the extremum.
    ph=high.shift(3).eq(high.rolling(7).max()); pl=low.shift(3).eq(low.rolling(7).min())
    peaks=high.shift(3).where(ph); troughs=low.shift(3).where(pl)
    rows=[]; last=-100
    for onset in range(200,len(c)):
        if dates.iloc[onset]<start or onset-last<10:continue
        p=int(prior.iloc[onset])
        if not p:continue
        zone=(k.iloc[onset]<=32 and k.iloc[onset-1]>32) or (k.iloc[onset]>=70 and k.iloc[onset-1]<70)
        if not zone:continue
        last=onset; direction=-p
        context=('flat_after_down' if p<0 else 'flat_after_up') if flat.iloc[onset] else ('downtrend' if p<0 else 'uptrend')
        prevhi=peaks.iloc[:onset+1].dropna().tail(2); prevlo=troughs.iloc[:onset+1].dropna().tail(2)
        if len(prevhi)<2 or len(prevlo)<2:continue
        upper=prevhi.iloc[-1]; lower=prevlo.iloc[-1]; anchor=c.iloc[onset]
        period='development' if dates.iloc[onset]<split else 'holdout'
        for scale,limit in [('short',3),('medium',7),('long',9)]:
          # Same event and decision time for all EMA ablations.
          for n in range(1,limit+1):
            i=onset+n-1
            if i>=len(c) or (period=='development' and dates.iloc[i]>=split):continue
            pkok=states[scale,n].iloc[i]==direction
            mom=direction*hist.diff(3).iloc[i]>0 and direction*obv.diff(5).iloc[i]>0
            wave=direction*(e[20].iloc[i]-e[50].iloc[i])>0
            a=direction*(c.iloc[i]-e[100].iloc[i])>0; b=direction*(c.iloc[i]-e[200].iloc[i])>0
            base=pkok and mom
            variants={'base':base,'ema100':base and a,'ema200':base and b,'both':base and a and b}
            for h in (5,7,10,20,30,60):
              j=i+h
              if j>=len(c) or (period=='development' and dates.iloc[j]>=split):continue
              future=c.iloc[i+1:j+1]; hp=peaks.iloc[i+1:j+1].dropna(); lp=troughs.iloc[i+1:j+1].dropna()
              hh=pd.concat([prevhi.tail(1),hp]); ll=pd.concat([prevlo.tail(1),lp])
              structural=len(hh)>=2 and len(ll)>=2 and direction*(hh.iloc[-1]-hh.iloc[-2])>0 and direction*(ll.iloc[-1]-ll.iloc[-2])>0
              beyond=(future>upper) if direction>0 else (future<lower)
              rev=structural and beyond.iloc[-3:].all()
              cont=(future.iloc[-3:]<lower).all() if p<0 else (future.iloc[-3:]>upper).all()
              excursion=float((direction*(future/anchor-1)).max())
              for threshold in (1,2,3,4,5):
                if rev: actual='accumulation' if context=='flat_after_down' else 'distribution' if context=='flat_after_up' else 'reversal_up' if p<0 else 'reversal_down'
                elif cont:actual=('recovery' if p<0 else 'correction') if excursion>=threshold/100 and not flat.iloc[onset] else ('continue_down' if p<0 else 'continue_up')
                else:actual='unresolved'
                for variant,ok in variants.items():
                  pred=('accumulation' if context=='flat_after_down' else 'distribution' if context=='flat_after_up' else 'reversal_up' if p<0 else 'reversal_down') if ok else ('recovery' if p<0 else 'correction') if (pkok or mom or wave) and not flat.iloc[onset] else ('continue_down' if p<0 else 'continue_up')
                  rows.append((symbol,period,str(dates.iloc[onset].date()),scale,n,n-1,h,threshold,variant,context,pred,actual))
    return pd.DataFrame(rows,columns=['symbol','period','event','scale','n','delay','horizon','threshold','variant','context','prediction','actual'])

if __name__=='__main__':
    symbols='GAS PVD VCB BID CTG TCB MBB ACB VPB HDB STB TPB VIB SHB FPT HPG HSG NKG MSN VNM SAB MWG PNJ REE GEX POW NT2 VSH PC1 PVT PVS BSR DPM DCM GMD VHC ANV DGC KDH NLG'.split()
    import concurrent.futures
    def safe(s):
      try:
        a=run(s)
        return s,a,'ok' if len(a) else 'empty'
      except Exception as ex:return s,pd.DataFrame(),repr(ex)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: data=list(pool.map(safe,symbols))
    pd.DataFrame([(s,len(a),err) for s,a,err in data],columns=['symbol','rows','status']).to_csv('cv02_f34_coverage.csv',index=False)
    ev=pd.concat([a for s,a,err in data if len(a)],ignore_index=True)
    keys=['period','scale','n','horizon','threshold','variant','context']
    ev['group']=np.where(ev.symbol.isin(['GAS','PVD']),ev.symbol,'other38')
    keys=['group']+keys
    cm=ev.groupby(keys+['prediction','actual']).size().rename('samples').reset_index()
    cm.to_csv('cv02_f34_confusion.csv',index=False)
    ev['correct']=ev.prediction.eq(ev.actual);ev['unresolved']=ev.actual.eq('unresolved');ev['wrong']=~ev.correct&~ev.unresolved
    out=ev.groupby(keys).agg(samples=('correct','size'),correct=('correct','sum'),wrong=('wrong','sum'),unresolved=('unresolved','sum'),delay=('delay','mean')).reset_index()
    p=out.correct/out.samples;n=out.samples;z=1.96
    out['correct_rate']=p;out['wilson_lower']=((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)).clip(lower=0)
    out.to_csv('cv02_f34_summary.csv',index=False)
    freq=ev.groupby(keys+['actual']).size().rename('count').reset_index()
    freq['probability']=freq['count']/freq.groupby(keys)['count'].transform('sum')
    freq.to_csv('cv02_f34_probabilities.csv',index=False)
    chosen=out[(out.period=='development')&(out.samples>=10)].sort_values(['wilson_lower','delay'],ascending=[False,True]).groupby(['group','scale','context','horizon']).head(1)
    out.merge(chosen[[k for k in keys if k!='period']],on=[k for k in keys if k!='period']).to_csv('cv02_f34_holdout.csv',index=False)
    ev.to_csv('cv02_f34_events.csv.gz',index=False,compression='gzip')
