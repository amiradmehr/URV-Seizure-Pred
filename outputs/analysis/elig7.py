import pandas as pd, numpy as np
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
dm=pd.read_csv(f'{R}/data/interim/manifests/decision_manifest.csv',usecols=['label','subject','split','recording_id','target_seizure_id'])
g=m.groupby('subject').agg(n=('eligible_for_prediction','size'),e=('eligible_for_prediction','sum'))
lost=set(g.index[g.e==0])
dm['subject']=dm['subject'].astype(int)
dm['lost_all']=dm.subject.isin(lost)
print("=== decisions contributed by the 24 patients who lose all seizures ===")
print(dm.groupby('lost_all').agg(decisions=('label','size'),pos=('label','sum')))
print("\nsplit x decisions:")
print(dm.groupby('split').agg(decisions=('label','size'),pos=('label','sum'),subj=('subject','nunique')))
print("\nseizures per split (annotated vs eligible):")
cfg_tr=set(range(1,101)); cfg_va=set(range(101,113)); cfg_te=set(range(113,126))
m['split']=m.subject.map(lambda s:'train' if s in cfg_tr else ('validation' if s in cfg_va else 'test'))
print(m.groupby('split').agg(annotated=('eligible_for_prediction','size'),eligible=('eligible_for_prediction','sum'),subj=('subject','nunique')))
print("\ntotal interictal hours represented by negatives (1 min each): %.1f h"%((dm.label==0).sum()/60))
print("positive decisions: %d over %d seizures"%(int(dm.label.sum()), m.eligible_for_prediction.sum()))
# how many patients contribute ONLY negatives to training
tr=dm[dm.split=='train']
ps=tr.groupby('subject')['label'].sum()
print("\ntrain subjects: %d, of which with 0 positive decisions: %d"%(len(ps),(ps==0).sum()))
