#!/usr/bin/env python3
"""Compare MicroLIA (Godines+2019) with BinML on real OGLE-IV events.

HEALTH WARNING: MicroLIA is bit-rotted as distributed. PyPI 2.8.1 imports but its
training_set.create calls simulate.simulate_mira_lightcurve, which the package does not ship;
GitHub main fails at import (simulate.py fetches an RRLyrae template tarball from a URL that now
404s). This script therefore MONKEYPATCHES the two broken simulators (a multi-sine Mira and a
local RRLyrae stub) so a model can be trained at all. The resulting numbers are INDICATIVE, not a
pristine MicroLIA benchmark, and the variable-class training data is our approximation.

Finding: MicroLIA detects microlensing (~6/8 real events) but has no binary/anomaly class -- it
cannot distinguish a planetary/binary event from a single lens, which is exactly what BinML adds.
"""

import numpy as np, warnings, os
warnings.simplefilter("ignore"); np.random.seed(0)
from MicroLIA import simulate, training_set, ensemble_model
from binml.legacy.surveys import fetch_ogle_ews

def _s(x): return float(np.ravel(x)[0])
def _mira(time, baseline, p1,a1,p2,a2,p3,a3):
    t=np.asarray(time,float); mag=np.full_like(t,_s(baseline))
    for p,a in [(p1,a1),(p2,a2),(p3,a3)]:
        p=_s(p); a=_s(a)
        if p>1e-3: mag=mag+a*np.sin(2*np.pi*t/p)
    return mag
def _rrlyr(timestamps, baseline, bailey=None):
    t=np.asarray(timestamps,float); per=float(np.random.uniform(0.3,0.8)); amp=float(np.random.uniform(0.3,0.9))
    ph=((t-t.min())/per)%1.0; mag=_s(baseline)+amp*(2*np.abs(ph-0.5)-0.5)
    return mag, per, amp
simulate.simulate_mira_lightcurve=_mira
if hasattr(simulate,'rrlyr_variable'): simulate.rrlyr_variable=_rrlyr

t,_,_=fetch_ogle_ews(2013,341, cache_dir="validation/ogle_cache")
t=np.asarray(t); t=t[(t>=t.min())&(t<t.min()+400)]
os.chdir("/tmp")
print(f"template {len(t)} pts; building training set (n_class=80)...", flush=True)
dx,dy=training_set.create(timestamps=[t], min_mag=15, max_mag=20, n_class=80, save_file=False)
print("dx", np.asarray(dx).shape, "classes:", sorted(set(dy)), flush=True)
model=ensemble_model.Classifier(dx,dy,clf='rf',optimize=False,impute=True); model.create()
print("trained OK\n", flush=True)

EV=[("2014-BLG-0289",2014,289,"strong bin"),("2013-BLG-0578",2013,578,"strong bin"),
    ("2016-BLG-1195",2016,1195,"Earth planet"),("2017-BLG-0482",2017,482,"planet"),
    ("2012-BLG-0026",2012,26,"2-planet"),("2013-BLG-0341",2013,341,"planet"),
    ("2015-BLG-0966",2015,966,"planet"),("2018-BLG-0677",2018,677,"planet")]
print(f"{'event':16s} {'note':13s} {'MicroLIA top':14s} {'P(ML)':>6s}")
print("-"*54)
CACHE="/Users/kunalbhatia/Desktop/Research/microlensing/BinML-repo/validation/ogle_cache"
nml=0
for name,yr,ev,note in EV:
    tt,mm,ee=fetch_ogle_ews(yr,ev,cache_dir=CACHE)
    pred=model.predict(tt,mm,ee)
    arr=np.asarray(pred,dtype=object); cls=arr[:,0].astype(str); pr=arr[:,1].astype(float)
    top=cls[np.argmax(pr)]
    mlp=next((pr[i] for i,c in enumerate(cls) if c.upper() in ('ML','MICROLENSING')), float('nan'))
    nml += ('ML' in top.upper() or 'LENS' in top.upper())
    print(f"{name:16s} {note:13s} {top:14s} {mlp:6.2f}")
print(f"\nMicroLIA: {nml}/8 real events top-classified as microlensing.")
