import pandas as pd, numpy as np, os
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
sz=pd.read_csv(f"{ROOT}/data/interim/manifests/seizure_manifest.csv")
print("rows",len(sz),"elig",sz.eligible_for_prediction.sum(),"evaluated",sz.eligibility_evaluated.sum())
sz["recording_id"]=sz.seizure_id.str.replace(r"_seizure-\d+$","",regex=True)
print("n recordings with seizures",sz.recording_id.nunique(),"n subjects",sz.subject.nunique())
POST=60*60.; HIST=60*60.
def ov(a,b,c,d): return a<d and b>c
rows=[]
for path,grp in sz.groupby("events_path"):
    ev=pd.read_csv(path,sep="\t")
    ev["onset"]=pd.to_numeric(ev["onset"],errors="coerce")
    ev["duration"]=pd.to_numeric(ev["duration"],errors="coerce")
    ev["eventType"]=ev["eventType"].fillna("").astype(str)
    rec_ev=ev[ev.onset.notna()&ev.duration.notna()&ev.duration.ge(0)]
    szev=grp[["onset_seconds","duration_seconds"]].values
    for _,s in grp.iterrows():
        on=float(s.onset_seconds); st=on-HIST
        short = st<0
        clust=any(ov(st,on,float(a),float(a)+float(d)+POST) for a,d in szev)
        nonbg=False
        for _,e in rec_ev.iterrows():
            et=str(e.eventType).strip().lower()
            if et=="bckg" or et.startswith("sz_"): continue
            if ov(st,on,float(e.onset),float(e.onset)+float(e.duration)): nonbg=True;break
        rows.append(dict(seizure_id=s.seizure_id,subject=s.subject,recording_id=s.recording_id,
            onset=on,dur=s.duration_seconds,event_type=s.event_type,vigilance=s.vigilance,
            lateralization=s.lateralization,localization=s.localization,
            elig_actual=bool(s.eligible_for_prediction),short=short,clust=clust,nonbg=nonbg))
r=pd.DataFrame(rows)
r["pred_elig"]=~(r.short|r.clust|r.nonbg)
print("\nreconstructed elig",r.pred_elig.sum(),"actual",r.elig_actual.sum())
print("mismatch",(r.pred_elig!=r.elig_actual).sum())
print(pd.crosstab(r.pred_elig,r.elig_actual))
inel=r[~r.elig_actual]
print("\nn ineligible",len(inel))
print("non-exclusive: short",inel.short.sum(),"clust",inel.clust.sum(),"nonbg",inel.nonbg.sum())
# code order cascade: short(start<0) first, then... actual code order: start<0, stop>n_times, nonfinite, bad, nonbg, clust
c=inel.copy(); rem=np.ones(len(c),bool)
for name,mask in [("short",c.short.values),("nonbg",c.nonbg.values),("clust",c.clust.values)]:
    take=rem&mask; print("cascade(code order)",name,take.sum()); rem=rem&~mask
print("residual",rem.sum())
print("\nanalyst cascade order (clust first): clust",inel.clust.sum())
rem2=~inel.clust.values
print(" then short",(rem2&inel.short.values).sum()," then nonbg",(rem2&~inel.short.values&inel.nonbg.values).sum())
print("\nnonbg total in all 883:",r.nonbg.sum())
r.to_csv(f"{ROOT}/outputs/analysis/refute/reasons.csv",index=False)
