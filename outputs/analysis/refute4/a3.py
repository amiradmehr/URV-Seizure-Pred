import numpy as np, pandas as pd, pickle, json
from scipy import stats
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if pd.notna(res.loc[res.tag==t,'sop_min'].iloc[0])]
d0=pd.read_csv(f'{ROOT}/outputs/cv/{tags[0]}/out_of_fold_predictions.csv',
               usecols=['subject','recording_id','label','target_seizure_id','fold'])
meta=d0[d0.label==1].groupby('target_seizure_id').agg(subject=('subject','first'),fold=('fold','first'))
det=pickle.load(open(f'{ROOT}/outputs/analysis/refute4/det.pkl','rb'))
D=pd.DataFrame({t:det[t]['obs'] for t in tags}).reindex(meta.index)
subj=meta.subject; fold=meta.fold; N=len(D)
def icc_anova(y,g):
    df=pd.DataFrame({'y':np.asarray(y,float),'g':np.asarray(g)})
    gb=df.groupby('g').y; ni=gb.size().to_numpy().astype(float); mi=gb.mean().to_numpy()
    NN=ni.sum(); kk=len(ni); gm=df.y.mean()
    MSB=(ni*(mi-gm)**2).sum()/(kk-1); MSW=((df.y-df.g.map(gb.mean()))**2).sum()/(NN-kk)
    m0=(NN-(ni**2).sum()/NN)/(kk-1); den=MSB+(m0-1)*MSW
    return np.nan if den==0 else (MSB-MSW)/den
# (a) fold-residualised patient ICC
raw=[];resid=[]
for t in tags:
    v=D[t].to_numpy().astype(float)
    raw.append(icc_anova(v,subj))
    fm=pd.Series(v).groupby(fold.values).transform('mean').to_numpy()
    resid.append(icc_anova(v-fm,subj))
print('ICC patient  raw mean %.4f | fold-residualised mean %.4f  (%.0f%% of the clustering is fold-level)'%(
      np.nanmean(raw),np.nanmean(resid),100*(1-np.nanmean(resid)/np.nanmean(raw))))
# (b) validate the clustered bootstrap: DE under permuted (structure-free) outcomes
rng=np.random.default_rng(2); subs=subj.unique(); pos_by=[np.where(subj.values==s)[0] for s in subs]
def de_boot(v,B=1500):
    p=v.mean()
    if p<=0 or p>=1: return np.nan
    bb=np.empty(B)
    for b in range(B):
        sel=np.concatenate([pos_by[q] for q in rng.integers(0,len(subs),len(subs))])
        bb[b]=v[sel].mean()
    return bb.std(ddof=1)**2/(p*(1-p)/N)
real=[de_boot(D[t].to_numpy().astype(float)) for t in tags]
perm=[de_boot(rng.permutation(D[t].to_numpy().astype(float))) for t in tags]
real=np.array(real,float);perm=np.array(perm,float)
print('DE_emp real  mean %.3f median %.3f  -> n_eff %.0f'%(np.nanmean(real),np.nanmedian(real),N/np.nanmean(real)))
print('DE_emp perm  mean %.3f median %.3f  (should be ~1.0 if the bootstrap is calibrated)'%(np.nanmean(perm),np.nanmedian(perm)))
print('paired t real vs perm: t=%.2f p=%.2g'%stats.ttest_rel(real,perm)[:2])
# (c) are ANY config pairs actually distinguishable? clustered paired test on all SOP10 pairs
s10=[t for t in tags if t.endswith('sop10')]
M=D[s10].to_numpy().astype(float); B=1200
ps=[];ds=[]
for i in range(len(s10)):
    for j in range(i+1,len(s10)):
        diff=M[:,i]-M[:,j]
        bb=np.empty(B)
        for b in range(B):
            sel=np.concatenate([pos_by[q] for q in rng.integers(0,len(subs),len(subs))])
            bb[b]=diff[sel].mean()
        se=bb.std(ddof=1); dd=diff.mean()
        ds.append(dd); ps.append(2*stats.norm.sf(abs(dd)/se) if se>0 else 1.0)
ds=np.array(ds);ps=np.array(ps)
print('\nSOP10 pairwise sensitivity differences (%d pairs): |d| median %.4f p90 %.4f max %.4f'%(len(ds),np.median(abs(ds)),np.percentile(abs(ds),90),abs(ds).max()))
print('  pairs with |d| >= 0.069 (analyst MDD): %d/%d'%(int((abs(ds)>=0.069).sum()),len(ds)))
print('  clustered-paired p<0.05: %d/%d ; p<0.05/210 (Bonferroni): %d'%(int((ps<0.05).sum()),len(ps),int((ps<0.05/len(ps)).sum())))
# (d) AP/chance NULL by circular shift of scores within recording (preserves autocorr + patient offsets)
print('\n--- AP/chance null by within-recording circular shift ---')
from sklearn.metrics import average_precision_score
for t in ['spectral-gru__large__h45__sop10','spectral-attention__large__h5__sop30','spectral-meanpool__small__h5__sop10']:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv',
                  usecols=['recording_id','label','probability','decision_time_seconds']).sort_values(['recording_id','decision_time_seconds'])
    y=d.label.to_numpy(); p=d.probability.to_numpy(); rid=d.recording_id.to_numpy()
    bnd=np.flatnonzero(np.r_[True,rid[1:]!=rid[:-1]]); bnd=np.r_[bnd,len(rid)]
    obs=average_precision_score(y,p)/y.mean()
    nulls=np.empty(200)
    for b in range(200):
        q=p.copy()
        for a,c in zip(bnd[:-1],bnd[1:]):
            if c-a>1: q[a:c]=np.roll(p[a:c],rng.integers(0,c-a))
        nulls[b]=average_precision_score(y,q)/y.mean()
    print('%-42s obs=%.4f  null mean=%.4f sd=%.4f p95=%.4f  perm-p=%.3f'%(
        t,obs,nulls.mean(),nulls.std(ddof=1),np.percentile(nulls,95),(1+(nulls>=obs).sum())/(1+len(nulls))))
