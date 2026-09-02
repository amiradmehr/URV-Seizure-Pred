import pandas as pd, numpy as np
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
m['rec']=m.apply(lambda r: f"sub-{r['subject']:03d}_ses-{int(r['session']):02d}_task-{r['task']}_run-{int(r['run']):02d}",axis=1)
d=pd.read_csv(f'{R}/outputs/analysis/elig_reasons.csv')
m=m.merge(d[['seizure_id','nonbg']],on='seizure_id')
def ov(a1,a2,b1,b2): return a1<b2 and a2>b1
print("HIST_min POST_min  n_eligible  n_patients  pct_of_883")
for HIST in [60,55,30,20,15]:
    for POST in [60,30,10,0]:
        keep=[]
        for rec,g in m.groupby('rec'):
            seiz=g[['onset_seconds','duration_seconds']].values
            for _,s in g.iterrows():
                on=float(s['onset_seconds']); st=on-HIST*60
                if st<0: continue
                if s['nonbg']: continue   # approx: nonbg computed at 60-min window (upper bound on exclusion)
                if any(ov(st,on,float(a),float(a)+float(b)+POST*60) for a,b in seiz): continue
                keep.append((s['seizure_id'],s['subject']))
        k=pd.DataFrame(keep,columns=['sid','subj'])
        print(f"  {HIST:3d}      {POST:3d}     {len(k):4d}        {k.subj.nunique():3d}       {100*len(k)/883:.1f}%")
