import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'src')
from seizure_prediction.features import chunk_features
M=pd.read_csv('data/interim/manifests/processed_shard_manifest.csv',dtype={'subject':str})
sz=pd.read_csv('outputs/analysis/_sz_with_rid.csv')
FD='data/interim/chunk_features'
rng=np.random.default_rng(7)
# pick 4 recordings that contain seizures, from different subjects
cand=sz[sz.has_feat].drop_duplicates('recording_id')
cand=cand.groupby('subject').head(1).sample(4,random_state=3)
for r in cand.itertuples():
    rid=r.recording_id
    row=M[M.recording_id==rid].iloc[0]
    X=np.load(row.X_path,mmap_mode='r')
    U=np.load(f'data/interim/unscaled_recordings/{rid}.npy',mmap_mode='r')
    F=np.load(f'{FD}/{rid}_features.npy')
    A=np.asarray(json.load(open(row.channel_availability_path)),bool)
    n=X.shape[1]
    ok_len = (n//1280 == len(F))
    # recompute a random chunk
    i=int(rng.integers(100,len(F)-100))
    f2=chunk_features(np.asarray(X[:,i*1280:(i+1)*1280],dtype=np.float32),256.0,1280,A)[0]
    dproc=float(np.abs(f2-F[i]).max())
    f3=chunk_features(np.asarray(U[:,i*1280:(i+1)*1280],dtype=np.float32),256.0,1280,A)[0]
    dunsc=float(np.abs(f3-F[i]).max())
    print(f"{rid}  procshape={X.shape} unscshape={U.shape} nfeat={len(F)} n//1280={n//1280} lenOK={ok_len} maxabsdiff_vs_PROCESSED={dproc:.3g} maxabsdiff_vs_UNSCALED={dunsc:.3g}")
