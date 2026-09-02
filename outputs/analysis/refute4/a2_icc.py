import numpy as np, pandas as pd, pickle, os, json
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if pd.notna(res.loc[res.tag==t,'sop_min'].iloc[0])]
# need seizure -> subject, recording, fold : read one file
d=pd.read_csv(f'{ROOT}/outputs/cv/{tags[0]}/out_of_fold_predictions.csv',
              usecols=['subject','recording_id','label','target_seizure_id','fold','split'])
print('subjects in OOF file:',d.subject.nunique(),' rows:',len(d))
pos=d[d.label==1]
meta=pos.groupby('target_seizure_id').agg(subject=('subject','first'),rec=('recording_id','first'),fold=('fold','first'))
print('seizures:',len(meta),'patients:',meta.subject.nunique(),'recordings:',meta.rec.nunique(),'folds:',meta.fold.nunique())
neg_only=set(d[d.label==0].subject.unique())-set(meta.subject.unique())
print('subjects contributing only negatives:',len(neg_only))
sizes=meta.subject.value_counts().to_numpy().astype(float)
N=sizes.sum(); k=len(sizes)
print('m_bar=%.4f  m_A(Kish)=%.4f  max=%d'%(N/k,(sizes**2).sum()/N,sizes.max()))
rsz=meta.rec.value_counts().to_numpy().astype(float)
print('per-recording: k_rec=%d m_bar=%.3f m_A=%.3f'%(len(rsz),rsz.sum()/len(rsz),(rsz**2).sum()/rsz.sum()))

det=pickle.load(open(f'{ROOT}/outputs/analysis/refute4/det.pkl','rb'))
D=pd.DataFrame({t:det[t]['obs'] for t in tags}).reindex(meta.index)
subj=meta.subject; rec=meta.rec; fold=meta.fold

def icc_anova(y,g):
    df=pd.DataFrame({'y':np.asarray(y,float),'g':np.asarray(g)})
    gb=df.groupby('g').y; ni=gb.size().to_numpy().astype(float); mi=gb.mean().to_numpy()
    NN=ni.sum(); kk=len(ni); gm=df.y.mean()
    MSB=(ni*(mi-gm)**2).sum()/(kk-1)
    MSW=((df.y-df.g.map(gb.mean()))**2).sum()/(NN-kk)
    m0=(NN-(ni**2).sum()/NN)/(kk-1)
    den=MSB+(m0-1)*MSW
    return np.nan if den==0 else (MSB-MSW)/den

icc_s=np.array([icc_anova(D[t].astype(float),subj) for t in tags])
icc_r=np.array([icc_anova(D[t].astype(float),rec) for t in tags])
icc_f=np.array([icc_anova(D[t].astype(float),fold) for t in tags])
print('\nICC(detect) patient : mean %.4f median %.4f'%(np.nanmean(icc_s),np.nanmedian(icc_s)))
print('ICC(detect) recording: mean %.4f median %.4f'%(np.nanmean(icc_r),np.nanmedian(icc_r)))
print('ICC(detect) fold     : mean %.4f median %.4f'%(np.nanmean(icc_f),np.nanmedian(icc_f)))

# permutation null for the ANOVA ICC on the SAME binary vectors (destroy patient structure)
rng=np.random.default_rng(3)
nulls=[]
for t in tags[:12]:
    v=D[t].to_numpy().astype(float)
    nulls += [icc_anova(rng.permutation(v),subj) for _ in range(60)]
nulls=np.array(nulls,float)
print('\npermutation-null ICC(patient): mean %.4f  p95 %.4f  p99 %.4f'%(np.nanmean(nulls),np.nanpercentile(nulls,95),np.nanpercentile(nulls,99)))
print('observed ICC exceeds null p95 in %d/%d configs'%(int((icc_s>np.nanpercentile(nulls,95)).sum()),len(icc_s)))

# ---- paired contrast ICC: does clustering hurt MODEL-vs-MODEL comparison? ----
s10=[t for t in tags if t.endswith('sop10')]
pair_icc=[]; pair_de=[]
subs=subj.unique(); pos_by_sub=[np.where(subj.values==s)[0] for s in subs]
for i in range(len(s10)):
    for j in range(i+1,len(s10)):
        diff=D[s10[i]].to_numpy().astype(float)-D[s10[j]].to_numpy().astype(float)
        if diff.std()==0: continue
        pair_icc.append(icc_anova(diff,subj))
pair_icc=np.array(pair_icc,float)
print('\nICC of PAIRED difference (detect_i - detect_j), %d SOP10 pairs: mean %.4f median %.4f p90 %.4f'%(
      len(pair_icc),np.nanmean(pair_icc),np.nanmedian(pair_icc),np.nanpercentile(pair_icc,90)))
m_A=(sizes**2).sum()/N
print('  -> DE for the PAIRED contrast at m_A=%.2f : mean %.3f  (vs marginal DE 1.715)'%(m_A,1+(m_A-1)*np.nanmean(pair_icc)))

# empirical clustered bootstrap of the paired difference for a few pairs
rng=np.random.default_rng(5); B=2000
out=[]
for (i,j) in [(0,1),(0,5),(3,9),(2,7),(6,11)]:
    diff=D[s10[i]].to_numpy().astype(float)-D[s10[j]].to_numpy().astype(float)
    bb=np.empty(B)
    for b in range(B):
        sel=np.concatenate([pos_by_sub[p] for p in rng.integers(0,len(subs),len(subs))])
        bb[b]=diff[sel].mean()
    se_c=bb.std(ddof=1); se_i=diff.std(ddof=1)/np.sqrt(len(diff))
    out.append((se_c/se_i)**2)
print('  empirical DE of the paired difference (5 pairs): %s  mean %.3f'%(np.round(out,3),np.mean(out)))
json.dump(dict(icc_patient=float(np.nanmean(icc_s)),icc_rec=float(np.nanmean(icc_r)),
               icc_paired=float(np.nanmean(pair_icc)),de_paired_emp=float(np.mean(out)),
               m_A=float(m_A),N=int(N),k=int(k)),open(f'{ROOT}/outputs/analysis/refute4/icc.json','w'),indent=2)
