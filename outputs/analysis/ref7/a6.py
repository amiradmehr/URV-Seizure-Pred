import numpy as np, pandas as pd, json, glob, os
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
FD=ROOT+"/data/interim/chunk_features/"
rng=np.random.default_rng(3)
files=sorted(glob.glob(FD+"*_features.npy"))
sel=[files[i] for i in rng.choice(len(files),400,replace=False)]
TP=[5,14,23]
tot=0; dead=0; recs_any=0; samp=[]
for f in sel:
    a=np.load(f.replace("_features.npy","_availability.npy"))
    b=np.load(f,mmap_mode="r")
    if b.shape[0]==0: continue
    cols=[c for c in TP if a[c]]
    if not cols: continue
    x=np.asarray(b[:,cols],dtype=np.float32)
    tot+=x.size; d=int((x<-20).sum()); dead+=d
    if d>0: recs_any+=1
    idx=rng.choice(x.shape[0],min(200,x.shape[0]),replace=False)
    samp.append(x[idx].ravel())
S=np.concatenate(samp)
print("CLAIM 5a: present-channel chunks scanned=%d  frac log_total_power<-20 = %.4f%%  recordings with any = %d/%d (%.1f%%)"%(
  tot,100*dead/tot,recs_any,len(sel),100*recs_any/len(sel)))
print("  log_total_power percentiles (n=%d): p1=%.2f p50=%.2f p99=%.2f min=%.2f ; log(1e-12)=%.4f"%(
  len(S),np.percentile(S,1),np.percentile(S,50),np.percentile(S,99),S.min(),np.log(1e-12)))
print("  frac of present-channel chunks in [-27.7,-20): %.4f%% ; exactly at floor (<-27.6): %.4f%%"%(
  100*np.mean((S<-20)&(S>=-27.7)),100*np.mean(S<-27.6)))
# do 'dead' chunks correspond to exact zeros in the stored recording?
D=pd.read_csv(ROOT+"/outputs/analysis/ref7/claim45.csv")
bad=D[D.dead_frac>0.99]
print("\nCLAIM 5b: recording-level confound check")
sz=pd.read_csv(ROOT+"/data/interim/manifests/seizure_manifest.csv")
sz["recording_id"]=("sub-"+sz.subject.astype(str).str.zfill(3)+"_ses-"+sz.session.astype(str).str.zfill(2)
                    +"_task-"+sz.task.astype(str)+"_run-"+sz.run.astype(str).str.zfill(2))
elig_recs=set(sz[sz.eligible_for_prediction].recording_id)
dec=pd.read_csv(ROOT+"/data/interim/manifests/decision_manifest.csv",usecols=["recording_id","label"])
posrecs=set(dec[dec.label==1].recording_id)
br=sorted(set(bad.rid))
print("  %d recordings supplied all the >=99%%-dead windows: %s"%(len(br),br[:12]))
print("  how many of them contain ANY positive decision at all? %d/%d"%(len(set(br)&posrecs),len(br)))
print("  recordings with >=1 positive decision: %d of %d recordings in the manifest"%(len(posrecs),dec.recording_id.nunique()))
# verify exact zeros in the stored continuous recording for one dead window
UR=ROOT+"/data/interim/unscaled_recordings/"
for rid in br[:3]:
    p=UR+rid+".npy"
    if not os.path.exists(p): print("   %s: unscaled npy missing"%rid); continue
    x=np.load(p,mmap_mode="r")
    b=np.load(FD+rid+"_features.npy",mmap_mode="r"); a=np.load(FD+rid+"_availability.npy")
    ch=[c for c,t in enumerate(TP) if a[t]]
    for c in ch:
        col=np.asarray(b[:,TP[c]]); k=np.where(col<-20)[0]
        if len(k)==0: continue
        s=int(k[len(k)//2])*1280
        seg=np.asarray(x[c,s:s+1280])
        print("   %s ch%d: %d/%d chunks dead; sample chunk exact-zero frac=%.3f max|v|=%.3e"%(
            rid,c,len(k),len(col),float((seg==0).mean()),float(np.abs(seg).max())))
        break
