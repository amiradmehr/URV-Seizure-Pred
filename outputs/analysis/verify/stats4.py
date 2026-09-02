import pandas as pd, numpy as np, os, glob
d=pd.read_csv('outputs/analysis/verify/recon.csv'); d['subject']=d.subject.astype(str)
dec=pd.read_csv('data/interim/manifests/decision_manifest.csv',dtype={'subject':str})
g=d.groupby('subject').agg(n=('seizure_id','count'),k=('actual','sum'))
zero=set(g[g.k==0].index)
tr=dec[dec.split=='train']; pos=tr.groupby('subject').label.sum(); zsub=set(pos[pos==0].index)
print('16 zero-positive TRAIN subjects contribute decisions:',len(tr[tr.subject.isin(zsub)]))
print('24 zero-eligible patients contribute decisions (all splits):',len(dec[dec.subject.isin(zero)]))
print('  of which in train:',len(tr[tr.subject.isin(zero)]))
print('overlap zero-eligible & train-zero-positive:',len(zero&zsub))

# claim6: dropped recordings
sh=pd.read_csv('data/interim/manifests/processed_shard_manifest.csv'); kept=set(sh.recording_id)
d['dropped']=~d.recording_id.isin(kept)
mis=d[d.dropped]
print('\ndropped seizure-recordings:',mis.recording_id.nunique(),'seizures',len(mis),'eligible',mis.actual.sum())
print('reason among the 170: short',mis.short.sum(),'clust',mis.clust.sum(),'nonbg',mis.nonbg.sum())
# EDF duration from header
def edf_dur(p):
    with open(p,'rb') as f:
        f.seek(236); nrec=int(f.read(8)); dur=float(f.read(8))
    return nrec*dur
allrec={}
for p in glob.glob('data/raw/sub-*/ses-*/eeg/*_eeg.edf'):
    allrec[os.path.basename(p).replace('_eeg.edf','')]=p
drop_all=set(allrec)-kept
dd=[edf_dur(allrec[r]) for r in drop_all]
kk=[edf_dur(allrec[r]) for r in list(kept)[:400]]
dd=np.array(dd); kk=np.array(kk)
print('\nALL 356 dropped recordings: median dur %.1f min, frac < 55 min = %.3f, max %.1f min'%(np.median(dd)/60,(dd<55*60).mean(),dd.max()/60))
print('sample of 400 kept recordings: median dur %.1f min, frac < 55 min = %.3f'%(np.median(kk)/60,(kk<55*60).mean()))
dropsz=set(mis.recording_id)
ds=np.array([edf_dur(allrec[r]) for r in dropsz])
print('110 dropped SEIZURE recordings: median %.1f min, frac<55min %.3f, max %.1f min'%(np.median(ds)/60,(ds<55*60).mean(),ds.max()/60))
print('total EEG hours in the 110 dropped seizure recordings: %.1f h (vs kept total est)'%(ds.sum()/3600))
allk=np.array([edf_dur(allrec[r]) for r in kept])
print('total EEG hours kept: %.1f h ; dropped(all 356): %.1f h'%(allk.sum()/3600, dd.sum()/3600))
