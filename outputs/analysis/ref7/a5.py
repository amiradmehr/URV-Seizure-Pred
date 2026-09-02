import numpy as np, pandas as pd, sys, json
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/ref7")
from pin import normalize_window_orig as NW
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
FD=ROOT+"/data/interim/chunk_features/"
FN=json.load(open(FD+"feature_names.json")); FN=np.array(FN)
dec=pd.read_csv(ROOT+"/data/interim/manifests/decision_manifest.csv",
    usecols=["recording_id","subject","decision_end_sample","label"])
rng=np.random.default_rng(7)
pos=dec[dec.label==1].sample(n=2000,random_state=11)
neg=dec[dec.label==0].sample(n=4000,random_state=12)
S=pd.concat([pos,neg])
CH=1280;NC=540
banks={}
def bank(rid):
    if rid not in banks:
        try: banks[rid]=(np.load(FD+rid+"_features.npy",mmap_mode="r"),np.load(FD+rid+"_availability.npy"))
        except Exception: banks[rid]=(None,None)
    return banks[rid]
TP=[5,14,23]  # log_total_power columns
recs=[]
for rid,g in S.groupby("recording_id"):
    b,av=bank(rid)
    if b is None: continue
    for _,r in g.iterrows():
        stop=int(r.decision_end_sample)//CH; start=stop-NC
        if start<0 or stop>b.shape[0]: continue
        w=np.asarray(b[start:stop],dtype=np.float32)
        q=np.percentile(w,[25,50,75],axis=0); iqr=q[2]-q[0]
        pres=av.astype(bool)
        n=NW(w,pres)
        mv=float(np.abs(n).max())
        miniqr=float(iqr[pres].min()) if pres.any() else np.nan
        argc=int(np.argmax(np.abs(n).max(axis=0)))
        # deadness
        dead_frac=0.0; ndead_ch=0
        for c,tp in enumerate(TP):
            if not pres[tp]: continue
            f=float((w[:,tp]<-20).mean()); dead_frac=max(dead_frac,f)
            if f>=0.99: ndead_ch+=1
        recs.append(dict(rid=rid,label=int(r.label),maxval=mv,miniqr=miniqr,argcol=argc,
                         argcol_iqr=float(iqr[argc]),dead_frac=dead_frac,ndead=ndead_ch))
D=pd.DataFrame(recs); D.to_csv(ROOT+"/outputs/analysis/ref7/claim45.csv",index=False)
print("windows scanned: %d (pos=%d neg=%d)"%(len(D),(D.label==1).sum(),(D.label==0).sum()))
print("\nCLAIM 4")
print("  max|normalized|: median=%.2f p99=%.2f max=%.3e"%(D.maxval.median(),D.maxval.quantile(.99),D.maxval.max()))
print("  present-channel IQR<1e-6: %d/%d (%.3f%%)   >100: %d (%.3f%%)   >1000: %d"%(
  (D.miniqr<1e-6).sum(),len(D),100*(D.miniqr<1e-6).mean(),(D.maxval>100).sum(),100*(D.maxval>100).mean(),(D.maxval>1000).sum()))
top=D.nlargest(8,"maxval")[["label","maxval","miniqr","argcol","argcol_iqr","dead_frac"]]
top["feature"]=[FN[i] for i in top.argcol]
print("  the 8 largest windows -- is the 1e-6 floor really the cause?")
print(top.to_string(index=False))
print("  correlation: among windows with max|val|>100, frac whose driving column had IQR<1e-6 = %.3f"%(
  (D[D.maxval>100].argcol_iqr<1e-6).mean() if (D.maxval>100).any() else np.nan))
print("  counterfactual max|val| if the floor were 1e-3 (bounding, from argcol_iqr): %.3e"%(
  (D.maxval*np.minimum(1.0,D.argcol_iqr.clip(lower=1e-12)/1e-3)).replace([np.inf],np.nan).max()))
# which columns have tiny IQR
tiny=D[D.miniqr<1e-6]
print("  windows with tiny present-channel IQR by label: pos=%.3f%% neg=%.3f%%"%(
  100*(D[D.label==1].miniqr<1e-6).mean(),100*(D[D.label==0].miniqr<1e-6).mean()))
print("  of those %d windows, %d (%.0f%%) also have a >=99%% dead present channel"%(
  len(tiny),int((tiny.ndead>0).sum()),100*(tiny.ndead>0).mean() if len(tiny) else 0))
print("\nCLAIM 5")
for th,nm in [(0.0,">0% dead"),(0.99,">=99% dead")]:
    a=(D[D.label==1].dead_frac>th).sum(); b=(D[D.label==0].dead_frac>th).sum()
    na=(D.label==1).sum(); nb=(D.label==0).sum()
    orr,p=stats.fisher_exact([[a,na-a],[b,nb-b]])
    print("  %-12s pos=%.3f%% (%d/%d)  neg=%.3f%% (%d/%d)  fisher p=%.4g"%(nm,100*a/na,a,na,100*b/nb,b,nb,p))
# max AUC gain from a binary cue present in f_neg of negs and 0 of pos
f=(D[D.label==0].dead_frac>0.99).mean()
print("  a binary cue present in %.4f%% of negatives and 0%% of positives shifts AUC by at most %.5f"%(100*f,f/2))
print("  those windows come from %d recordings / %d subjects"%(
  D[(D.dead_frac>0.99)].rid.nunique(), D[(D.dead_frac>0.99)].rid.str[:7].nunique()))
