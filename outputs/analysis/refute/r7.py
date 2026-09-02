import pandas as pd, numpy as np, glob, os
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
sz=pd.read_csv(f'{ROOT}/data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['rid']=sz.seizure_id.str.replace(r'_seizure-\d+$','',regex=True)
feats={os.path.basename(p).replace('_features.npy','') for p in glob.glob(f'{ROOT}/data/interim/chunk_features/*_features.npy')}
# preload events
evs={}
for path in sz.events_path.unique():
    ev=pd.read_csv(path,sep="\t")
    ev["onset"]=pd.to_numeric(ev["onset"],errors="coerce"); ev["duration"]=pd.to_numeric(ev["duration"],errors="coerce")
    ev["eventType"]=ev["eventType"].fillna("").astype(str)
    evs[path]=ev[ev.onset.notna()&ev.duration.notna()&ev.duration.ge(0)]
def ov(a,b,c,d): return a<d and b>c
def yields(H,P):
    H*=60.; P*=60.
    out=[]
    for path,grp in sz.groupby('events_path'):
        rec_ev=evs[path]; szev=grp[['onset_seconds','duration_seconds']].values
        nb=[(float(e.onset),float(e.onset)+float(e.duration)) for _,e in rec_ev.iterrows()
            if not(str(e.eventType).strip().lower()=="bckg" or str(e.eventType).strip().lower().startswith("sz_"))]
        for _,s in grp.iterrows():
            on=float(s.onset_seconds); st=on-H
            e = (st>=0) and not any(ov(st,on,float(a),float(a)+float(d)+P) for a,d in szev) \
                and not any(ov(st,on,x,y) for x,y in nb)
            out.append((s.seizure_id,s.subject,s.rid,e,s.vigilance))
    return pd.DataFrame(out,columns=['sid','subj','rid','elig','vig'])
base=yields(60,60)
print("sanity: H60/P60 elig=%d patients=%d (expect 317/101)"%(base.elig.sum(),base[base.elig].subj.nunique()))
print(f"{'H':>4}{'P':>5}{'elig':>7}{'pats':>6}{'elig&feats':>12}{'newly_elig':>12}{'new&nofeat':>12}")
for H,P in [(60,60),(60,30),(60,10),(55,60),(30,10),(15,60),(15,10),(55,30),(55,10)]:
    y=yields(H,P); e=y[y.elig]
    hasf=e.rid.isin(feats)
    new=e[~e.sid.isin(set(base.loc[base.elig,'sid']))]
    print(f"{H:>4}{P:>5}{len(e):>7}{e.subj.nunique():>6}{int(hasf.sum()):>12}{len(new):>12}{int((~new.rid.isin(feats)).sum()):>12}")
# how many recovered at 60/10 have a prior seizure ending 10-60 min before onset (postictal contamination)
y=yields(60,10); rec=y[y.elig & ~y.sid.isin(set(base.loc[base.elig,'sid']))]
print("\nrecovered by relaxing postictal 60->10:",len(rec))
cnt=0
for _,s in sz[sz.seizure_id.isin(set(rec.sid))].iterrows():
    g=sz[sz.rid==s.rid]
    prior=[float(a)+float(d) for a,d in zip(g.onset_seconds,g.duration_seconds) if float(a)+float(d)<=s.onset_seconds]
    if prior and (s.onset_seconds-max(prior))<=3600: cnt+=1
print("  of which a prior seizure ENDS within 60 min of onset (postictal EEG inside history):",cnt)
