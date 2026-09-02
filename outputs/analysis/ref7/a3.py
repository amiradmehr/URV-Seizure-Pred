import numpy as np, pandas as pd, pickle, sys, json
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from sklearn.metrics import roc_auc_score
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
C=pickle.load(open(ROOT+"/outputs/analysis/ref7/cache.pkl","rb"))
pre,i300,ioff,inter=C["pre"],C["i300"],C["ioff"],C["inter"]
meta,imeta=C["meta"],C["imeta"]; pair_idx=C["pair_idx"]; pair_win=C["pair_win"]
FN=json.load(open(ROOT+"/data/interim/chunk_features/feature_names.json"))
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
dur=np.array([m["dur"] for m in meta],float)
def norm(W,AV): return np.stack([normalize_window(W[i],AV[i]) for i in range(len(W))])
Np,Ni=norm(i300,AVp),norm(inter,AVi)
LL=[i for i,n in enumerate(FN) if n.endswith("log_line_length")]
print("=== B. MECHANISM TEST for claim 1's '~9x dilution'")
for ci_,nm in [(LL[0],FN[LL[0]])]:
    a=Np[:,:,ci_]; b=Ni[:,:,ci_]
    okp=AVp[:,ci_]; okn=AVi[:,ci_]
    print(" feature:",nm," (present in %d/%d ictal, %d/%d inter)"%(okp.sum(),len(okp),okn.sum(),len(okn)))
    # event-triggered profile in the ictal window: last 60 chunks = onset..onset+300
    prof=a[okp][:,-72:].mean(0)
    print("  ictal-window profile (IQR units), chunk offsets -6..-1min rel to onset then 0..+5min:")
    print("   t=-1min %.3f | t=0..+1min %.3f | +1..+2 %.3f | +4..+5 %.3f | rest-of-window %.3f"%(
       a[okp][:,-72:-60].mean(), a[okp][:,-60:-48].mean(), a[okp][:,-48:-36].mean(),
       a[okp][:,-12:].mean(), a[okp][:,:-60].mean()))
    mp_p=a[okp].mean(1); mp_n=b[okn].mean(1)
    l60_p=a[okp][:,-60:].mean(1); l60_n=b[okn][:,-60:].mean(1)
    def d(x,y): return (x.mean()-y.mean())/np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    print("  mean-pool(540): ictal %.4f vs inter %.4f  sd_inter=%.4f  d=%.3f AUC=%.3f"%(
       mp_p.mean(),mp_n.mean(),mp_n.std(),d(mp_p,mp_n),roc_auc_score(np.r_[np.ones(len(mp_p)),np.zeros(len(mp_n))],np.r_[mp_p,mp_n])))
    print("  last60-pool  : ictal %.4f vs inter %.4f  sd_inter=%.4f  d=%.3f AUC=%.3f"%(
       l60_p.mean(),l60_n.mean(),l60_n.std(),d(l60_p,l60_n),roc_auc_score(np.r_[np.ones(len(l60_p)),np.zeros(len(l60_n))],np.r_[l60_p,l60_n])))
    print("  IF pure 9x dilution: shift would be %.4f/9=%.4f ; sd of mean-pool ratio sd(last60)/sd(mean)=%.2f"%(
       l60_p.mean()-l60_n.mean(),(l60_p.mean()-l60_n.mean())/9., l60_n.std()/mp_n.std()))
print()
print("=== C. what mean-pooling a median-centred window actually is")
allw=np.concatenate([Np,Ni]); av=np.concatenate([AVp,AVi])
m=allw.mean(1)
print("  |mean over 540 of normalized window| : median=%.4f p95=%.4f  (0 iff mean==median)"%(
   np.median(np.abs(m[av])),np.percentile(np.abs(m[av]),95)))
print("  sd across windows of that statistic  : %.4f"%m[av].std())
print()
print("=== D. paired within-recording test, pre-ictal vs same-recording far-interictal")
res=[]
for mode in ["raw","normalized"]:
    diffs=[];subs=[]
    for k,i in enumerate(pair_idx):
        wp=pre[i]; wn=pair_win[k]; a=AVp[i]
        if mode=="normalized":
            wp=normalize_window(wp,a); wn=np.stack([normalize_window(w,a) for w in wn])
        diffs.append(wp.mean(0)-wn.mean(1).mean(0)); subs.append(meta[i]["subject"])
    D=np.stack(diffs); subs=np.array(subs)
    # subject-level averaging to avoid pseudo-replication
    us=np.unique(subs); Ds=np.stack([D[subs==s].mean(0) for s in us])
    print(" mode=%s  n_seizures=%d n_subjects=%d"%(mode,len(D),len(us)))
    for lab,M in [("per-seizure",D),("per-subject",Ds)]:
        t,p=stats.ttest_1samp(M,0.0,axis=0)
        ok=np.isfinite(p)
        order=np.argsort(np.where(ok,p,9))
        best=[(FN[j],float(t[j]),float(p[j])) for j in order[:4]]
        nsig=int((p[ok]<0.05/27).sum())
        print("   %-12s n=%d  #p<0.05/27 = %d ; top: %s"%(lab,M.shape[0],nsig,
            "; ".join("%s t=%.2f p=%.4f"%b for b in best)))
    # power
    print("   detectable |d| at alpha=0.05/27, 80%% power, n=%d : %.3f ; at alpha=.05: %.3f"%(
        len(D),(3.10+0.84)/np.sqrt(len(D)),(1.96+0.84)/np.sqrt(len(D))))
