import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
res=pd.read_csv(ROOT/'outputs/sweep/results.csv'); tags=res.dropna(subset=['sens_asleep']).tag.tolist()
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
vig=dict(zip(sz.seizure_id.astype(str),sz.vigilance.astype(str)))
rows=[]
for tag in tags:
    d=pd.read_csv(ROOT/f'outputs/cv/{tag}/out_of_fold_predictions.csv',
                  usecols=['recording_id','subject','label','target_seizure_id','probability'])
    d['subject']=d['subject'].astype(str).str.zfill(3)
    neg=d[d.label==0]; pos=d[d.label==1]
    # per-patient negative distribution (sorted)
    psort={s:np.sort(v.to_numpy()) for s,v in neg.groupby('subject')['probability']}
    pk=pos.groupby(['target_seizure_id','subject','recording_id']).probability.max().reset_index()
    def pct(r):
        a=psort.get(r.subject,np.array([]))
        return np.searchsorted(a,r.probability)/len(a) if len(a)>=60 else np.nan
    pk['pct_pat']=pk.apply(pct,axis=1)
    pk['tag']=tag; pk['vigilance']=pk.target_seizure_id.map(lambda s: vig.get(str(s),'un'))
    rows.append(pk[['tag','target_seizure_id','subject','recording_id','probability','pct_pat','vigilance']])
P=pd.concat(rows,ignore_index=True); P.to_csv(OUT/'per_seizure_patient_pct.csv',index=False)

S=pd.read_csv(OUT/'per_seizure_by_config.csv',dtype={'subject':str})
S=S.merge(P[['tag','target_seizure_id','pct_pat']].rename(columns={'target_seizure_id':'seizure_id'}),on=['tag','seizure_id'],how='left')
S=S[S.vigilance.isin(['asleep','awake'])]
FA=1-1/60.0
S['caught_rec']=(S.pct_rec>=FA).astype(float); S.loc[S.pct_rec.isna(),'caught_rec']=np.nan
S['caught_pat']=(S.pct_pat>=FA).astype(float); S.loc[S.pct_pat.isna(),'caught_pat']=np.nan
S['caught_loc']=(S.pct_loc>=FA).astype(float); S.loc[S.pct_loc.isna(),'caught_loc']=np.nan
S.to_csv(OUT/'per_seizure_full.csv',index=False)

def paired(col,label,frame=None):
    f=S if frame is None else frame
    p=f.pivot_table(index='tag',columns='vigilance',values=col,aggfunc='mean').dropna()
    d=p['asleep']-p['awake']; t=stats.ttest_rel(p['asleep'],p['awake'])
    print(f'{label:44s} asleep={p["asleep"].mean():.4f} awake={p["awake"].mean():.4f} diff={d.mean():+.4f} '
          f't={t.statistic:+.2f} p={t.pvalue:.4g} ({(d>0).sum()}/{len(d)} configs +)')
    return d.mean()

print('=== 5. SENSITIVITY @ 1 FA/h with different threshold calibrations ===')
g0=paired('caught','GLOBAL threshold (published result)')
g1=paired('caught_pat','PER-PATIENT calibrated threshold')
g2=paired('caught_rec','PER-RECORDING calibrated threshold')
g3=paired('caught_loc','PER-RECORDING +-3h calibrated threshold')
print(f'\ngap explained by patient-level base rate : {100*(1-g1/g0):.0f}%')
print(f'gap explained by recording-level base rate: {100*(1-g2/g0):.0f}%')

print('\n=== 6. PATIENT AS UNIT ===')
sub=S.dropna(subset=['caught'])
per_pat=sub.groupby(['tag','subject','vigilance']).caught.mean().reset_index()
w=per_pat.pivot_table(index=['tag','subject'],columns='vigilance',values='caught').dropna()
print('patients with BOTH asleep and awake eligible seizures:',w.reset_index().subject.nunique())
d=(w['asleep']-w['awake']).groupby('tag').mean()
t=stats.ttest_rel(w.groupby('tag')['asleep'].mean(),w.groupby('tag')['awake'].mean())
print(f'WITHIN-PATIENT paired (patients w/ both): asleep={w["asleep"].mean():.4f} awake={w["awake"].mean():.4f} '
      f'diff={d.mean():+.4f} t={t.statistic:+.2f} p={t.pvalue:.4g} ({(d>0).sum()}/{len(d)} configs +)')
# patient-averaged unpaired (all patients)
pp=sub.groupby(['tag','subject','vigilance']).caught.mean().reset_index()
q=pp.pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
t2=stats.ttest_rel(q['asleep'],q['awake'])
print(f'PATIENT-AVERAGED (all patients, unpaired): asleep={q["asleep"].mean():.4f} awake={q["awake"].mean():.4f} '
      f'diff={(q["asleep"]-q["awake"]).mean():+.4f} t={t2.statistic:+.2f} p={t2.pvalue:.4g}')

print('\n=== 7. LEAVE-ONE-PATIENT-OUT on the global-threshold effect ===')
pv=[]
for s in sorted(sub.subject.unique()):
    f=sub[sub.subject!=s]
    p=f.pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
    if p.shape[1]<2: continue
    t=stats.ttest_rel(p['asleep'],p['awake'])
    pv.append((s,(p['asleep']-p['awake']).mean(),t.pvalue,int((sub.subject==s).sum()/42)))
L=pd.DataFrame(pv,columns=['subject','diff','p','n_sz']).sort_values('p',ascending=False)
print(L.head(8).to_string(index=False))
print('worst-case p when dropping one patient: %.4g'%L.p.max(),' n patients with p>0.05:',(L.p>0.05).sum())
