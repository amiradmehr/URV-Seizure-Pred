import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
FD='data/interim/chunk_features'
NAMES=json.load(open(f'{FD}/feature_names.json'))
CH=5.0; GUARD=1800.0; RATIO=5
s=pd.read_csv('outputs/analysis/_sz_with_rid.csv'); s=s[s.has_feat]
th_idx=[NAMES.index(f'{c}::log_power_theta') for c in ['BTE_LEFT','BTE_RIGHT','CROSS_HEAD']]
tot_idx=[NAMES.index(f'{c}::log_total_power') for c in ['BTE_LEFT','BTE_RIGHT','CROSS_HEAD']]
rng=np.random.default_rng(0)   # same seed/order as positive_control.py for the control draw
out=[]
for rid,g in s.groupby('recording_id'):
    F=np.load(f'{FD}/{rid}_features.npy'); A=np.load(f'{FD}/{rid}_availability.npy')
    n=len(F)
    if n<400: continue
    iv=[(float(r.onset_seconds),float(r.onset_seconds)+float(r.duration_seconds)) for r in g.itertuples()]
    ict=set()
    for (o,e) in iv:
        a=int(np.ceil(o/CH)); b=int(np.floor(e/CH)); cc=list(range(a,b)) or [int(o//CH)]
        ict.update(c for c in cc if 0<=c<n)
    ok=np.ones(n,bool); t=np.arange(n)*CH+CH/2
    for (o,e) in iv: ok &= ~((t>o-GUARD)&(t<e+GUARD))
    cand=np.flatnonzero(ok)
    if len(ict)==0 or len(cand)<10: continue
    k=min(len(cand),RATIO*len(ict)); ctl=rng.choice(cand,size=k,replace=False)   # keeps RNG stream identical
    ni=len(ict)
    if ni<2 or len(ctl)<5: continue
    av=np.array([A[0],A[9],A[18]])
    def feat(idx,rows):
        M=F[rows][:,idx]; return (M*av[None,:]).sum(1)/av.sum()
    ic=np.array(sorted(ict))
    real=roc_auc_score(np.r_[np.ones(ni),np.zeros(len(ctl))], np.r_[feat(th_idx,ic),feat(th_idx,ctl)])
    # SHAM: random CONTIGUOUS block of ni chunks entirely inside the interictal-ok region,
    # scored against the same scattered controls (block chunks removed from controls).
    okset=set(cand.tolist()); shams=[]
    starts=[c for c in cand[:-ni] if all((c+j) in okset for j in range(ni))] if ni<=len(cand) else []
    if starts:
        for _ in range(20):
            st=int(rng.choice(starts)); blk=np.arange(st,st+ni)
            ctl2=np.setdiff1d(ctl,blk)
            if len(ctl2)<5: continue
            shams.append(roc_auc_score(np.r_[np.ones(ni),np.zeros(len(ctl2))],
                                       np.r_[feat(th_idx,blk),feat(th_idx,ctl2)]))
    out.append(dict(rid=rid,n_ict=ni,n_ctl=len(ctl),real_auc=real,
                    sham_med=float(np.median(shams)) if shams else np.nan,
                    sham_p90=float(np.quantile(shams,0.9)) if shams else np.nan,
                    sham_frac_gt70=float(np.mean(np.array(shams)>0.7)) if shams else np.nan))
R=pd.DataFrame(out); R.to_csv('outputs/analysis/refute3/sham_vs_real.csv',index=False)
print('n recordings',len(R))
print('REAL   median %.4f mean %.4f frac>0.7 %.4f frac<0.5 %.4f'%(R.real_auc.median(),R.real_auc.mean(),(R.real_auc>0.7).mean(),(R.real_auc<0.5).mean()))
q=R.dropna(subset=['sham_med'])
print('SHAM(contiguous interictal block, same n) median-of-medians %.4f mean %.4f frac(sham_med>0.7) %.4f'%(q.sham_med.median(),q.sham_med.mean(),(q.sham_med>0.7).mean()))
print('SHAM overall frac of draws >0.7: %.4f'%q.sham_frac_gt70.mean())
print('paired real-vs-shammed: real>sham_med in %.4f of recordings'%(q.real_auc>q.sham_med).mean())
from scipy import stats
print('wilcoxon real vs sham_med:',stats.wilcoxon(q.real_auc,q.sham_med))
