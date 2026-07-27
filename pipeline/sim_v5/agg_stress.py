"""Aggregate the in-region stress-test predictions into per-class / per-regime metrics."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from .classes import CLASS_NAMES
I_NON = CLASS_NAMES.index("NonPSPL")

def load_dir(d):
    parts = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        for k in z.files:
            if k == "param_fields": continue
            parts.setdefault(k, []).append(z[k])
    if not parts: return None
    out = {k: np.concatenate(v) for k, v in parts.items()}
    return out

def prf(y, pred, w, c):
    tp = (w*((pred==c)&(y==c))).sum(); fp=(w*((pred==c)&(y!=c))).sum(); fn=(w*((pred!=c)&(y==c))).sum()
    r = tp/max(tp+fn,1e-9); p = tp/max(tp+fp,1e-9)
    return r, p, (2*r*p/max(r+p,1e-9))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)   # local dir with regime subdirs of npz
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    regimes = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    report = {"regimes": {}, "class_names": CLASS_NAMES}
    allY=[]; allP=[]; allW=[]
    for rg in regimes:
        D = load_dir(os.path.join(a.root, rg))
        if D is None: continue
        y=D["label"].astype(int); pred=D["logits"].argmax(1)
        w=1.0/np.clip(D["keep_prob"].astype(np.float64),1e-3,1.0)
        n=len(y)
        cls={}
        for c,name in enumerate(CLASS_NAMES):
            if (y==c).sum()==0: continue
            r,p,f=prf(y,pred,w,c)
            cls[name]={"recall":round(r,4),"precision":round(p,4),"f1":round(f,4),"n":int((y==c).sum())}
        report["regimes"][rg]={"n":int(n),"per_class":cls}
        allY.append(y);allP.append(pred);allW.append(w)
        print(f"{rg:14s} n={n:>8,}  NonPSPL r/p={cls.get('NonPSPL',{}).get('recall','-')}/{cls.get('NonPSPL',{}).get('precision','-')}")
    # global (natural population only, for honest population metrics)
    Dn = load_dir(os.path.join(a.root,"natural"))
    if Dn is not None:
        y=Dn["label"].astype(int); pred=Dn["logits"].argmax(1)
        w=1.0/np.clip(Dn["keep_prob"].astype(np.float64),1e-3,1.0)
        glob_cls={}
        for c,name in enumerate(CLASS_NAMES):
            r,p,f=prf(y,pred,w,c); glob_cls[name]={"recall":round(r,4),"precision":round(p,4),"f1":round(f,4),"n":int((y==c).sum())}
        macro=np.mean([glob_cls[n]["f1"] for n in CLASS_NAMES])
        report["natural_population"]={"n":int(len(y)),"per_class":glob_cls,"macro_f1":round(macro,4)}
    json.dump(report, open(a.out,"w"), indent=2)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
