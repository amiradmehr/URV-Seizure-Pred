import numpy as np, pandas as pd, os, json
from scipy import stats, optimize
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
za=stats.norm.ppf(0.975); zb=stats.norm.ppf(0.80)
N=286.0; k=93; m_A=5.48; ICC=0.160; DE=1.687; n_eff=N/DE
print(f"n_seizures={N:.0f}  n_patients={k}  DE={DE:.2f}  n_eff={n_eff:.1f}\n")

def n_two_prop(p1,p2):
    pb=(p1+p2)/2
    return (za*np.sqrt(2*pb*(1-pb))+zb*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p2-p1)**2
def n_one_samp(p0,p1):
    return (za*np.sqrt(p0*(1-p0))+zb*np.sqrt(p1*(1-p1)))**2/(p1-p0)**2
def solve(f,p0,n_target):
    g=lambda p2: f(p0,p2)-n_target
    lo,hi=p0+1e-6,0.999
    if g(hi)>0: return np.nan
    return optimize.brentq(g,lo+1e-9,hi)

print("=== MDD in seizure-level sensitivity, 80%% power, alpha=0.05 two-sided ===")
print(f"{'baseline':>9} | {'two-indep-cohorts':>28} | {'one-sample vs fixed ref':>26}")
print(f"{'p0':>9} | {'naive n=286':>13} {'n_eff=169':>13} | {'naive':>12} {'n_eff':>12}")
for p0 in (0.05,0.10,0.20):
    a=solve(n_two_prop,p0,N); b=solve(n_two_prop,p0,n_eff)
    c=solve(n_one_samp,p0,N); d=solve(n_one_samp,p0,n_eff)
    f=lambda x: "n/a" if not np.isfinite(x) else f"{x:.3f}"
    print(f"{p0:>9.2f} | {f(a):>13} {f(b):>13} | {f(c):>12} {f(d):>12}   (abs diff eff: "
          f"+{(b-p0) if np.isfinite(b) else float('nan'):.3f} / +{(d-p0):.3f})")

# ---- paired model-vs-model (McNemar) with real discordance ----
det={}
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
for tag in res.tag:
    p=f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv'
    if not os.path.exists(p): continue
    d=pd.read_csv(p,usecols=['subject','label','target_seizure_id','probability'])
    neg=d.loc[d.label==0,'probability'].to_numpy()
    thr=np.sort(neg)[::-1][int(np.floor(len(neg)/60.0))]
    pos=d[d.label==1]
    det[tag]=pos.assign(a=pos.probability>=thr).groupby('target_seizure_id').a.any()
    if 'subj' not in dir(): subj=pos.groupby('target_seizure_id').subject.first()
D=pd.DataFrame(det); subj=subj.reindex(D.index)
sop10=[t for t in D.columns if t.endswith('sop10')]
cols=D[sop10].to_numpy()
disc=[]
for i in range(len(sop10)):
    for j in range(i+1,len(sop10)):
        disc.append((cols[:,i]!=cols[:,j]).mean())
disc=np.array(disc)
pd_med=float(np.median(disc))
print(f"\nPaired McNemar: observed discordance between SOP=10 config pairs "
      f"median={pd_med:.3f} (range {disc.min():.3f}-{disc.max():.3f}, {len(disc)} pairs)")
def mcnemar_mdd(pi_d,n):
    # solve delta = pi10-pi01 for 80% power
    g=lambda dl: ((za*np.sqrt(pi_d)+zb*np.sqrt(pi_d-dl**2))**2/dl**2)-n
    return optimize.brentq(g,1e-5,pi_d-1e-6)
for n_,lab in ((N,'naive n=286'),(n_eff,'n_eff=169')):
    print(f"   MDD paired ({lab}): delta = {mcnemar_mdd(pd_med,n_):.3f} seizures-sensitivity points")

# ---- required n for 0.30 vs 0.10 ----
print("\n=== SAMPLE SIZE TO DETECT sens 0.30 vs 0.10 baseline (80%%, alpha .05) ===")
for lab,f_,p0,p1 in (('two independent cohorts (per arm)',n_two_prop,0.10,0.30),
                     ('one-sample vs fixed 0.10 reference',n_one_samp,0.10,0.30)):
    ne=f_(p0,p1); nr=ne*DE
    print(f"{lab}: n_eff={ne:.1f} -> RAW eligible seizures needed = {nr:.0f}"
          f"  (have {N:.0f}, shortfall {max(0,nr-N):.0f})")
    per_pat=N/k
    print(f"    at {per_pat:.2f} eligible seizures/seizure-bearing patient -> {nr/per_pat:.0f} patients")
    print(f"    at 317 eligible / 125 recruited = {317/125:.2f} per recruited patient -> {nr/(317/125):.0f} patients to recruit")
    print(f"    at 317/883 = {317/883:.3f} eligibility rate -> {nr/(317/883):.0f} annotated seizures to be monitored")

# ---- ceiling: more seizures from the SAME 93 patients ----
print(f"\nCeiling if you only add seizures to the existing {k} patients:")
for m in (3.08,6,10,20,1e6):
    print(f"   m={m if m<1e5 else float('inf'):>8.2f} seizures/patient -> N={k*m:>9.0f}  n_eff={k*m/(1+(m-1)*ICC):>7.1f}")
print(f"   asymptotic ceiling k/ICC = {k/ICC:.0f} effective seizures, ever.")

# ---- AP/chance bars ----
sd_cfg=res.ap_over_chance.std(); mu=res.ap_over_chance.mean()
print(f"\n=== AP/chance decision bars ===")
print(f"sweep: mean={mu:.4f} sd={sd_cfg:.4f} n={len(res)}  observed max={res.ap_over_chance.max():.4f}")
for K,lab in ((1,'uncorrected'),(42,'Bonferroni over the 42 already tried'),(43,'family of 43')):
    a=0.05/K; z=stats.norm.ppf(1-a)
    print(f"  {lab:>36}: z={z:.3f} -> bar vs empirical null N({mu:.3f},{sd_cfg:.3f}) = {mu+z*sd_cfg:.3f}"
          f"   | vs theoretical null N(1.000,{sd_cfg:.3f}) = {1+z*sd_cfg:.3f}")
# max-of-K exact
for K in (42,43):
    z=stats.norm.ppf(0.95**(1/K))
    print(f"  max-of-{K} exact 95th pct: z={z:.3f} -> {mu+z*sd_cfg:.3f}")
