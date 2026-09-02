import pandas as pd, numpy as np, glob, os
from scipy import stats
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
d=pd.read_csv('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/elig_reasons.csv'); m=m.merge(d[['seizure_id','gap_prev_end_min','n_seiz_in_rec']],on='seizure_id')
m=m[m.eligible_for_prediction]
# per-subject total annotated seizure burden
burden=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv').groupby('subject').size().rename('n_annotated')
files=sorted(glob.glob(f'{R}/outputs/cv/*/out_of_fold_predictions.csv'))
print("configs:",len(files))
res=[]
for f in files:
    p=pd.read_csv(f,usecols=['label','probability','target_seizure_id'])
    pos=p[p.label==1]
    if pos.empty: continue
    # per-seizure max probability, converted to within-config percentile among negatives
    neg=p.loc[p.label==0,'probability'].values
    neg=np.sort(neg)
    sm=pos.groupby('target_seizure_id')['probability'].max().rename('maxp').reset_index()
    sm['pct']=np.searchsorted(neg,sm.maxp.values)/len(neg)
    sm=sm.merge(m[['seizure_id','gap_prev_end_min','vigilance','subject','n_seiz_in_rec']],left_on='target_seizure_id',right_on='seizure_id')
    sm=sm.merge(burden,on='subject')
    sm['has_prior']=sm.gap_prev_end_min.notna()
    sm['cfg']=os.path.basename(os.path.dirname(f))
    res.append(sm)
a=pd.concat(res)
print("rows:",len(a),"configs:",a.cfg.nunique(),"seizures:",a.target_seizure_id.nunique())
def paired(col,labA,labB,maskA,maskB,name):
    g=a.groupby('cfg').apply(lambda s:pd.Series({'A':s.loc[maskA(s),'pct'].mean(),'B':s.loc[maskB(s),'pct'].mean()}),include_groups=False).dropna()
    t=stats.ttest_rel(g.A,g.B)
    print(f"{name}: {labA}={g.A.mean():.4f} {labB}={g.B.mean():.4f} diff={g.A.mean()-g.B.mean():+.4f} paired t p={t.pvalue:.4g} (n_cfg={len(g)})")
paired('pct','has_prior_seizure','isolated',lambda s:s.has_prior,lambda s:~s.has_prior,'prior-seizure-in-recording')
paired('pct','asleep','awake',lambda s:s.vigilance=='asleep',lambda s:s.vigilance=='awake','vigilance(control)')
paired('pct','high-burden(>=8 annot)','low-burden(<8)',lambda s:s.n_annotated>=8,lambda s:s.n_annotated<8,'patient seizure burden')
# correlation of per-seizure mean percentile with gap
per=a.groupby('target_seizure_id').agg(pct=('pct','mean'),gap=('gap_prev_end_min','first'),nb=('n_annotated','first')).reset_index()
sub=per.dropna(subset=['gap'])
rho,p=stats.spearmanr(sub.gap,sub.pct); print(f"\nSpearman(gap_to_prev_seizure_min, mean score percentile) n={len(sub)} rho={rho:.3f} p={p:.3g}")
rho,p=stats.spearmanr(per.nb,per.pct); print(f"Spearman(patient annotated burden, mean score percentile) n={len(per)} rho={rho:.3f} p={p:.3g}")
