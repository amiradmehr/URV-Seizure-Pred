import pandas as pd, numpy as np
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
r=pd.read_csv(f"{ROOT}/outputs/analysis/refute/reasons.csv")
print("patients with >=1 eligible:",r[r.elig_actual].subject.nunique())
sz=pd.read_csv(f"{ROOT}/data/interim/manifests/seizure_manifest.csv")
sz["elig"]=sz.eligible_for_prediction
def cramv(ct):
    chi2,p,dof,_=stats.chi2_contingency(ct)
    n=ct.values.sum(); k=min(ct.shape)
    return chi2,dof,p,np.sqrt(chi2/(n*(k-1)))
for col in ["event_type","localization","vigilance","lateralization"]:
    ct=pd.crosstab(sz[col].fillna("NA"),sz["elig"])
    print(col,"chi2=%.1f dof=%d p=%.3g V=%.3f"%cramv(ct),"ncat",ct.shape[0])
# MWU
a=sz.loc[sz.elig,"onset_seconds"]; b=sz.loc[~sz.elig,"onset_seconds"]
u,p=stats.mannwhitneyu(a,b); print("onset MWU p=%.3g A=%.3f"%(p,u/(len(a)*len(b))))
a=sz.loc[sz.elig,"duration_seconds"]; b=sz.loc[~sz.elig,"duration_seconds"]
u,p=stats.mannwhitneyu(a,b); print("dur MWU p=%.3g A=%.3f n_el=%d"%(p,u/(len(a)*len(b)),len(a)))
# vigilance 2x2
ct=pd.crosstab(sz.vigilance.fillna("NA"),sz.elig); print("\nvigilance table\n",ct)
sub=sz[sz.vigilance.isin(["awake","asleep"])]
t=pd.crosstab(sub.vigilance,sub.elig)
orr,p=stats.fisher_exact(t.values); print("fisher awake/asleep",t.values.tolist(),"OR=%.3f p=%.3g"%(orr,p))
# clustered vs vigilance
m=r.merge(sz[["seizure_id","vigilance"]],on="seizure_id",how="left",suffixes=("","_m"))
sub2=m[m.vigilance.isin(["awake","asleep"])]
print("\nclustered rate by vigilance\n",sub2.groupby("vigilance")[["clust","short","nonbg"]].mean())
t2=pd.crosstab(sub2.vigilance,sub2.clust); orr,p=stats.fisher_exact(t2.values); print("clust OR",orr,"p=%.3g"%p, t2.values.tolist())
t3=pd.crosstab(sub2.vigilance,sub2.short); orr,p=stats.fisher_exact(t3.values); print("short OR",orr,"p=%.3g"%p, t3.values.tolist())
