"""Paired comparison of clean-only vs corruption-mixture candidate training."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--robust',required=True); ap.add_argument('--clean',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args()
    robust={}; clean={}
    for dst,rows in ((robust,read(a.robust)),(clean,read(a.clean))):
        for r in rows:
            k=(r['dataset'],r['family'],int(r['severity']),r['id'])
            dst[k]=float(r['candidate_mae'])
    keys=sorted(set(robust)&set(clean)); missing=(len(robust),len(clean),len(keys))
    groups=defaultdict(list)
    for ds,fam,sev,i in keys:
        groups[(ds,fam,sev)].append((robust[(ds,fam,sev,i)],clean[(ds,fam,sev,i)]))
    rng=np.random.default_rng(a.seed); result=[]
    for (ds,fam,sev),pairs in sorted(groups.items()):
        x=np.asarray(pairs); delta=x[:,1]-x[:,0] # positive means corruption training wins
        n=len(delta); ix=rng.integers(0,n,(a.bootstrap,n)); boot=delta[ix].mean(axis=1)
        result.append({'dataset':ds,'family':fam,'severity':sev,'n_images':n,
            'robust_train_candidate_mae':x[:,0].mean(),'clean_train_candidate_mae':x[:,1].mean(),
            'delta_clean_minus_corruption':delta.mean(),'ci_low':np.quantile(boot,.025),
            'ci_high':np.quantile(boot,.975),'fraction_corruption_better':np.mean(delta>0)})
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    # Bootstrap images after averaging their repeated conditions, preserving pairing.
    summaries=[]
    for ds in sorted({k[0] for k in keys}):
        for label, keep in [('clean',lambda fam,sev:fam=='clean'),
                            ('all_corruptions',lambda fam,sev:fam!='clean'),
                            ('all_conditions',lambda fam,sev:True)]:
            per_image=defaultdict(list)
            for (d,fam,sev,i) in keys:
                if d==ds and keep(fam,sev):
                    per_image[i].append((robust[(d,fam,sev,i)],clean[(d,fam,sev,i)]))
            pairs=np.asarray([np.mean(v,axis=0) for _,v in sorted(per_image.items())])
            delta=pairs[:,1]-pairs[:,0]; n=len(delta)
            ix=rng.integers(0,n,(a.bootstrap,n)); boot=delta[ix].mean(axis=1)
            summaries.append({'dataset':ds,'condition_set':label,'n_images':n,
                'robust_train_candidate_mae':pairs[:,0].mean(),'clean_train_candidate_mae':pairs[:,1].mean(),
                'delta_clean_minus_corruption':delta.mean(),'ci_low':np.quantile(boot,.025),
                'ci_high':np.quantile(boot,.975),'fraction_corruption_better':np.mean(delta>0)})
    sp=p.with_name('paired_candidate_summary.csv')
    with sp.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(f'paired rows: robust={missing[0]}, clean={missing[1]}, matched={missing[2]}; groups={len(result)}; saved={p} and {sp}')

if __name__=='__main__': main()
