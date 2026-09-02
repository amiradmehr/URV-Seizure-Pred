import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from pathlib import Path

tags = ['spectral-gru__large__h45__sop10','logistic-mean__small__h5__sop10',
        'spectral-attention__large__h5__sop30','spectral-meanpool__large__h45__sop30']
sz = pd.read_csv('data/interim/manifests/seizure_manifest.csv', dtype={'subject':str}).set_index('seizure_id')

for t in tags:
    d = pd.read_csv(Path('outputs/cv')/t/'out_of_fold_predictions.csv',
                    usecols=['subject','recording_id','label','probability','target_seizure_id',
                             'decision_time_seconds'], dtype={'subject':str})
    pooled_auc = roc_auc_score(d.label, d.probability)
    # within-patient AUC (patient-specific comparable), weighted by n positives
    aucs=[]; ws=[]
    for s,g in d.groupby('subject'):
        if g.label.nunique()<2: continue
        aucs.append(roc_auc_score(g.label,g.probability)); ws.append(g.label.sum())
    aucs=np.array(aucs); ws=np.array(ws,float)
    # within-recording AUC
    raucs=[]
    for s,g in d.groupby('recording_id'):
        if g.label.nunique()<2: continue
        raucs.append(roc_auc_score(g.label,g.probability))
    raucs=np.array(raucs)
    print('\n===',t)
    print(' pooled AUC %.4f | within-patient AUC: mean %.4f (unweighted, n=%d), pos-weighted %.4f, median %.4f'%(
        pooled_auc, aucs.mean(), len(aucs), (aucs*ws).sum()/ws.sum(), np.median(aucs)))
    print(' within-recording AUC: mean %.4f (n=%d), frac>0.5 %.3f'%(raucs.mean(), len(raucs), (raucs>0.5).mean()))
    t_,p_=stats.ttest_1samp(aucs,0.5); print('  within-patient AUC vs 0.5: t=%.2f p=%.3f'%(t_,p_))

    # patient-level score offsets: how much of score variance is between-patient?
    neg = d[d.label==0]
    mu = neg.groupby('subject').probability.mean()
    tot = neg.probability.var()
    betw = neg.groupby('subject').probability.transform('mean').var()
    print('  between-patient share of NEGATIVE score variance: %.3f  (patient mean range %.3f-%.3f)'%(
        betw/tot, mu.min(), mu.max()))

    # is 'caught at 1/h' basically determined by the patient's baseline score level?
    negs=np.sort(neg.probability.to_numpy())[::-1]; thr=negs[int(len(negs)/60)]
    pos=d[d.label==1]
    c=pos.assign(a=pos.probability>=thr).groupby('target_seizure_id')['a'].any()
    base = pos.groupby('target_seizure_id')['subject'].first().map(mu)
    print('  corr(caught, patient baseline negative score) r=%.3f'%(np.corrcoef(c.values.astype(float),base.values)[0,1]))
    vg = sz.loc[c.index,'vigilance'].astype(str).values
    print('  patient baseline score: asleep-seizure %.4f vs awake-seizure %.4f'%(
        base.values[vg=='asleep'].mean(), base.values[vg=='awake'].mean()))
