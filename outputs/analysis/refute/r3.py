import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(0)
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv')
sz['elig']=sz.eligible_for_prediction.astype(bool)
sz['subject']=sz.subject.astype(str)
print("== within-patient homogeneity of attributes ==")
for col in ["event_type","localization","lateralization","vigilance"]:
    v=sz.fillna({col:"NA"}).groupby("subject")[col].nunique()
    print(f"{col}: patients with a single value {int((v==1).sum())}/{v.size}; "
          f"share of seizures in single-value patients "
          f"{sz.fillna({col:'NA'}).groupby('subject')[col].transform('nunique').eq(1).mean():.3f}")
def chi2stat(labels, cats):
    ct=pd.crosstab(cats,labels)
    if ct.shape[1]<2 or ct.shape[0]<2: return 0.0
    return stats.chi2_contingency(ct)[0]
print("\n== within-patient permutation test (permute eligibility inside each patient) ==")
subj=sz.subject.values; elig=sz.elig.values
idx_by_s=[np.where(subj==s)[0] for s in np.unique(subj)]
for col in ["event_type","localization","lateralization","vigilance"]:
    cats=sz[col].fillna("NA").values
    obs=chi2stat(elig,cats)
    null=[]
    for _ in range(2000):
        perm=elig.copy()
        for ii in idx_by_s: perm[ii]=rng.permutation(perm[ii])
        null.append(chi2stat(perm,cats))
    null=np.array(null)
    p=(1+(null>=obs).sum())/(1+len(null))
    print(f"{col}: obs chi2={obs:.1f}  perm-null mean={null.mean():.1f} 95th={np.percentile(null,95):.1f}  p_within-patient={p:.4f}")
# onset / duration within patient
print("\n== onset & duration, within-patient permutation on MWU-U ==")
for col in ["onset_seconds","duration_seconds"]:
    x=sz[col].values
    def stat(lab):
        a=x[lab]; b=x[~lab]
        return stats.mannwhitneyu(a,b)[0]/(len(a)*len(b))
    obs=stat(elig); null=[]
    for _ in range(2000):
        perm=elig.copy()
        for ii in idx_by_s: perm[ii]=rng.permutation(perm[ii])
        null.append(stat(perm))
    null=np.array(null); p=(1+(np.abs(null-0.5)>=abs(obs-0.5)).sum())/(1+len(null))
    print(f"{col}: obs A={obs:.3f} null A mean={null.mean():.3f} sd={null.std():.3f} p_within={p:.4f}")
