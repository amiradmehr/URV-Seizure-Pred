import pandas as pd, numpy as np
from scipy import stats
s=pd.read_csv('outputs/analysis/sep.csv')
print(len(s),"rows; groups:", s.groupby(['elig','clust','short']).size().to_dict())
el=s[s.elig]; cl=s[(~s.elig)&s.clust]
print("eligible n=%d sep=%.4f  excl-clust n=%d sep=%.4f  MWU p=%.4g"%(
 len(el),el.sep.mean(),len(cl),cl.sep.mean(),stats.mannwhitneyu(cl.sep,el.sep)[1]))
# paired within recording
recs=set(el.rec)&set(cl.rec); recs=sorted(recs)
a=np.array([cl[cl.rec==r].sep.mean() for r in recs]); b=np.array([el[el.rec==r].sep.mean() for r in recs])
d=a-b
t,p=stats.ttest_rel(a,b); w=stats.wilcoxon(a,b)[1]
ci=stats.t.interval(0.95,len(d)-1,loc=d.mean(),scale=stats.sem(d))
print("\nwithin-recording paired: n=%d clust=%.4f elig=%.4f diff=%+.4f t p=%.3f wilcoxon p=%.3f higher in %d/%d"%(
  len(recs),a.mean(),b.mean(),d.mean(),p,w,(d>0).sum(),len(d)))
print("  95%% CI of diff = [%+.4f, %+.4f]   (marginal effect was +0.0189 -> inside CI? %s)"%(
  ci[0],ci[1], ci[0]<=0.0189<=ci[1]))
print("  sd(diff)=%.4f ; power to detect +0.0189 at alpha .05, n=%d: %.2f"%(
  d.std(ddof=1),len(d),
  1-stats.nct.cdf(stats.t.ppf(0.975,len(d)-1), len(d)-1, 0.0189/(d.std(ddof=1)/np.sqrt(len(d))))))
# same but on sep - null (baseline-corrected)
a2=np.array([ (cl[cl.rec==r].sep-cl[cl.rec==r].null).mean() for r in recs])
b2=np.array([ (el[el.rec==r].sep-el[el.rec==r].null).mean() for r in recs])
print("\nbaseline-corrected (sep-null) within recording: clust=%+.4f elig=%+.4f diff=%+.4f p=%.3f"%(
  a2.mean(),b2.mean(),(a2-b2).mean(),stats.ttest_rel(a2,b2)[1]))
# how selected are those 55 recordings?
allrec=s.groupby('rec').size()
print("\nrecordings analysed: %d ; with BOTH kinds: %d"%(s.rec.nunique(),len(recs)))
sub=s[s.subject.notna()]
print("seizures/recording: both-kind recs %.2f vs other recs %.2f"%(
  s[s.rec.isin(recs)].groupby('rec').size().mean(), s[~s.rec.isin(recs)].groupby('rec').size().mean()))
# per-subject
subs=sorted(set(el.subject)&set(cl.subject))
a3=np.array([cl[cl.subject==x].sep.mean() for x in subs]); b3=np.array([el[el.subject==x].sep.mean() for x in subs])
print("within-subject paired: n=%d diff=%+.4f p=%.3f"%(len(subs),(a3-b3).mean(),stats.ttest_rel(a3,b3)[1]))
