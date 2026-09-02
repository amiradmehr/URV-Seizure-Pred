import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
pc=A.groupby('subject').vigilance.agg(lambda s:'asleep-only' if (s=='asleep').all() else ('awake-only' if (s=='awake').all() else 'mixed'))
A['pgroup']=A.subject.map(pc)
print('seizure-weighted (analyst a6):',A.groupby(['tag','pgroup']).caught.mean().unstack().mean().round(4).to_dict())
per=A.groupby(['seizure_id','subject','pgroup']).caught.mean().reset_index()
print('seizure-as-unit           :',per.groupby('pgroup').caught.mean().round(4).to_dict())
print('patient-as-unit           :',per.groupby('subject').caught.mean().groupby(pc).mean().round(4).to_dict())
print('n seizures per group      :',per.pgroup.value_counts().to_dict())
print('n patients total          :',per.subject.nunique())
# claim 6 tertile check
sub=pd.read_csv(OUT/'negatives_with_proxy.csv')
for tag in ['logistic-mean__small__h5__sop30','spectral-gru__large__h15__sop30']:
    dd=pd.read_csv(ROOT/f'outputs/cv/{tag}/out_of_fold_predictions.csv',usecols=['recording_id','decision_time_seconds','label','probability'])
    m=sub.merge(dd[dd.label==0][['recording_id','decision_time_seconds','probability']],on=['recording_id','decision_time_seconds'],how='inner',suffixes=('','_t')).dropna(subset=['probability_t'])
    m['q']=m.groupby('recording_id').proxy.rank(pct=True)
    hi=m[m.q>=2/3].probability_t; lo=m[m.q<=1/3].probability_t
    print('%-34s within-rec tertile hi-lo = %+.4f (hi=%.4f lo=%.4f, n=%d/%d)'%(tag,hi.mean()-lo.mean(),hi.mean(),lo.mean(),len(hi),len(lo)))
