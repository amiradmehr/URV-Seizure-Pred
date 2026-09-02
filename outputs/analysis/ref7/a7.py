import numpy as np, pandas as pd, mne, warnings
from scipy import signal as sg
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
sz=pd.read_csv(ROOT+"/data/interim/manifests/seizure_manifest.csv")
sz["recording_id"]=("sub-"+sz.subject.astype(str).str.zfill(3)+"_ses-"+sz.session.astype(str).str.zfill(2)
                    +"_task-"+sz.task.astype(str)+"_run-"+sz.run.astype(str).str.zfill(2))
e=sz[sz.eligible_for_prediction].copy()
e=e.sample(frac=1.0,random_state=5).groupby("subject").head(2)
e=e.sample(n=min(130,len(e)),random_state=6)
BANDS=dict(in_0p5_40=(0.5,40),hi_40_48=(40,48),line_48_52=(48,52),hi_52_70=(52,70),
           hi_70_90=(70,90),line_98_102=(98,102),hi_90_125=(90,125))
def psd(x,fs=256.0):
    f,P=sg.welch(x,fs=fs,nperseg=1024,noverlap=512,axis=-1)
    return f,P
def bp(f,P,lo,hi): 
    m=(f>=lo)&(f<hi); return np.trapezoid(P[...,m],f[m],axis=-1)
rows=[]
for _,r in e.iterrows():
    sub=str(r.subject).zfill(3); ses=str(r.session).zfill(2)
    p=f"{ROOT}/data/raw/sub-{sub}/ses-{ses}/eeg/{r.recording_id}_eeg.edf"
    try: raw=mne.io.read_raw_edf(p,preload=False,verbose="ERROR")
    except Exception: continue
    fs=raw.info["sfreq"]; on=float(r.onset_seconds)
    segs={"last10":(on-600,on),"far":(on-2700,on-2100)}
    ok=True; out={}
    for nm,(a,b) in segs.items():
        s0,s1=int(a*fs),int(b*fs)
        if s0<0 or s1>raw.n_times: ok=False; break
        try: x=raw.get_data(start=s0,stop=s1)
        except Exception: ok=False; break
        x=x-x.mean(axis=-1,keepdims=True)
        f,P=psd(x,fs)
        out[nm]={k:float(np.mean(bp(f,P,*v))) for k,v in BANDS.items()}
    if not ok: continue
    d=dict(seizure_id=r.seizure_id,subject=r.subject,vigilance=str(r.vigilance),chs=len(raw.ch_names))
    for nm in segs:
        for k in BANDS: d[f"{nm}__{k}"]=out[nm][k]
    rows.append(d)
D=pd.DataFrame(rows); D.to_csv(ROOT+"/outputs/analysis/ref7/claim6.csv",index=False)
print("n seizures read=%d subjects=%d"%(len(D),D.subject.nunique()))
hi=["hi_40_48","hi_52_70","hi_70_90","hi_90_125"]
tot_all=D[["last10__"+k for k in BANDS]].sum(axis=1)
above40_incl_line=tot_all-D["last10__in_0p5_40"]
above40_excl_line=above40_incl_line-D["last10__line_48_52"]-D["last10__line_98_102"]
print("\nCLAIM 6a: fraction of 0.5-125 Hz power above 40 Hz (last 10 min before onset)")
print("  including 48-52 and 98-102 Hz line bands: median=%.4f  mean=%.4f  p90=%.4f"%(
   (above40_incl_line/tot_all).median(),(above40_incl_line/tot_all).mean(),(above40_incl_line/tot_all).quantile(.9)))
print("  EXCLUDING the two line bands           : median=%.4f  mean=%.4f"%(
   (above40_excl_line/tot_all).median(),(above40_excl_line/tot_all).mean()))
print("  share of the >40 Hz power that is 48-52 + 98-102 Hz line noise: median=%.4f"%(
   ((D["last10__line_48_52"]+D["last10__line_98_102"])/above40_incl_line).median()))
print("\nCLAIM 6b: paired log(last10 / 45-35min) per band")
for k in ["in_0p5_40"]+hi:
    lr=np.log(D["last10__"+k]/D["far__"+k]); lr=lr[np.isfinite(lr)]
    t,p=stats.ttest_1samp(lr,0.0)
    print("  %-12s mean=%+.4f  t=%.2f p=%.3f  n=%d  [95%% CI %+.3f,%+.3f]"%(
      k,lr.mean(),t,p,len(lr),lr.mean()-1.96*lr.sem(),lr.mean()+1.96*lr.sem()))
print("\nCLAIM 6c: asleep vs awake single-band AUC (last 10 min), and is >40 Hz really better?")
m=D.vigilance.isin(["asleep","awake"]); Dv=D[m]; y=(Dv.vigilance=="asleep").astype(int).values
print("  n=%d (asleep=%d awake=%d) subjects=%d"%(len(Dv),y.sum(),(1-y).sum(),Dv.subject.nunique()))
auc={}; sc={}
for k in ["in_0p5_40"]+hi:
    s=np.log(Dv["last10__"+k].values); sc[k]=s; auc[k]=roc_auc_score(y,s)
    print("  %-12s AUC=%.3f"%(k,auc[k]))
rng=np.random.default_rng(0); idx=np.arange(len(y))
for k in hi:
    dif=[]
    for _ in range(2000):
        b=rng.choice(idx,len(idx),True)
        if len(np.unique(y[b]))<2: continue
        dif.append(roc_auc_score(y[b],sc[k][b])-roc_auc_score(y[b],sc["in_0p5_40"][b]))
    dif=np.array(dif)
    print("   paired bootstrap AUC(%s) - AUC(in_0.5_40) = %+.3f [%+.3f,%+.3f] p=%.3f"%(
      k,dif.mean(),np.percentile(dif,2.5),np.percentile(dif,97.5),2*min((dif<0).mean(),(dif>0).mean())))
# within-subject (subjects with both vigilance states)
print("  within-subject AUC (subjects with both states):")
for k in ["in_0p5_40"]+hi:
    num=den=0;u=0
    for s_ in Dv.subject.unique():
        mm=(Dv.subject==s_).values
        if len(np.unique(y[mm]))<2: continue
        u+=1; a=(y[mm]==1).sum(); b=(y[mm]==0).sum()
        num+=roc_auc_score(y[mm],sc[k][mm])*a*b; den+=a*b
    print("    %-12s within-subject AUC=%s (n_subj=%d)"%(k,("%.3f"%(num/den)) if den else "n/a",u))
