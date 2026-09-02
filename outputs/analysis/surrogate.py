import pandas as pd, numpy as np, json
from pathlib import Path
from multiprocessing import Pool

sweep = pd.read_csv('outputs/sweep/results.csv')
tags = [t for t in sweep.tag if '__' in t]

def run(tag):
    p = Path('outputs/cv')/tag/'out_of_fold_predictions.csv'
    d = pd.read_csv(p, usecols=['recording_id','label','probability','target_seizure_id',
                                'decision_time_seconds'])
    d = d.sort_values(['recording_id','decision_time_seconds']).reset_index(drop=True)
    neg = np.sort(d.loc[d.label==0,'probability'].to_numpy())[::-1]
    thr = float(neg[min(int((d.label==0).sum()/60), len(neg)-1)])
    lab = d.label.to_numpy(); prob = d.probability.to_numpy()
    sid = d.target_seizure_id.to_numpy(dtype=object)
    recs = d.recording_id.to_numpy()
    # group slices per recording
    bounds = np.flatnonzero(np.r_[True, recs[1:]!=recs[:-1], True])
    slices = [(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)]
    posmask = lab==1
    uniq = pd.unique(sid[posmask])
    sidx = {s:i for i,s in enumerate(uniq)}
    pidx = np.array([sidx[s] for s in sid[posmask]])
    def sens(pr):
        al = pr[posmask]>=thr
        out = np.zeros(len(uniq), bool)
        np.logical_or.at(out, pidx, al)
        return out.mean()
    obs = sens(prob)
    rng = np.random.default_rng(7)
    null = np.empty(500)
    for b in range(500):
        pr = prob.copy()
        for a,z in slices:
            n=z-a
            if n>1: pr[a:z]=np.roll(prob[a:z], rng.integers(n))
        null[b]=sens(pr)
    pv = (null>=obs).mean()
    return dict(tag=tag, thr=thr, obs=float(obs), null_mean=float(null.mean()),
                null_sd=float(null.std()), null_p95=float(np.percentile(null,95)),
                p=float(pv), n_seiz=int(len(uniq)))

if __name__=='__main__':
    with Pool(14) as pool:
        res = pool.map(run, tags)
    r = pd.DataFrame(res).sort_values('p')
    r.to_csv('outputs/analysis/surrogate_results.csv', index=False)
    pd.set_option('display.width',200)
    print(r.to_string(index=False))
    print()
    print('configs with surrogate p<0.05: %d/42 (expected by chance ~2.1)'%(r.p<0.05).sum())
    print('mean observed sens %.4f   mean surrogate-null sens %.4f'%(r.obs.mean(), r.null_mean.mean()))
    print('min p %.4f ; Bonferroni-corrected min p %.3f'%(r.p.min(), min(1,r.p.min()*42)))
