import pandas as pd, numpy as np
from pathlib import Path
from multiprocessing import Pool
sw = pd.read_csv('outputs/sweep/results.csv'); sw=sw[sw.tag.str.contains('__')]
tags = list(sw.sort_values('sens_at_1ph',ascending=False).tag[:6]) + ['spectral-gru__large__h45__sop10']
def run(tag):
    d = pd.read_csv(Path('outputs/cv')/tag/'out_of_fold_predictions.csv',
                    usecols=['recording_id','label','probability','target_seizure_id','decision_time_seconds'])
    d = d.sort_values(['recording_id','decision_time_seconds']).reset_index(drop=True)
    neg=np.sort(d.loc[d.label==0,'probability'].to_numpy())[::-1]
    thr=float(neg[min(int((d.label==0).sum()/60),len(neg)-1)])
    lab=d.label.to_numpy(); prob=d.probability.to_numpy(); sid=d.target_seizure_id.to_numpy(object)
    recs=d.recording_id.to_numpy()
    b=np.flatnonzero(np.r_[True,recs[1:]!=recs[:-1],True]); sl=[(b[i],b[i+1]) for i in range(len(b)-1)]
    pm=lab==1; uq=pd.unique(sid[pm]); ix={s:i for i,s in enumerate(uq)}
    pi=np.array([ix[s] for s in sid[pm]])
    def sens(pr):
        out=np.zeros(len(uq),bool); np.logical_or.at(out,pi,pr[pm]>=thr); return out.mean()
    obs=sens(prob); rng=np.random.default_rng(7); null=np.empty(400)
    for k in range(400):
        pr=prob.copy()
        for a,z in sl:
            n=z-a
            if n>1: pr[a:z]=np.roll(prob[a:z],rng.integers(n))
        null[k]=sens(pr)
    return dict(tag=tag,obs=float(obs),null_mean=float(null.mean()),null_sd=float(null.std()),
                null_p95=float(np.percentile(null,95)),p=float((null>=obs).mean()),n=int(len(uq)))
if __name__=='__main__':
    with Pool(7) as p: r=pd.DataFrame(p.map(run,tags))
    pd.set_option('display.width',200); print(r.to_string(index=False))
    r.to_csv('outputs/analysis/surrogate_small.csv',index=False)
