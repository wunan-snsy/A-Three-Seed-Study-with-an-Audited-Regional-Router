"""Paired image bootstrap summaries for per-image locked-test evaluation files."""
from __future__ import annotations
import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs-root',default=r'D:\PhysCausalDiff\output\runs')
    ap.add_argument('--seeds',default='17,29');ap.add_argument('--bootstrap',type=int,default=3000)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--out',default='D:/PhysCausalDiff/output/runs/test_bootstrap_summary.csv')
    a=ap.parse_args();root=Path(a.runs_root);groups=defaultdict(list)
    for seed in map(int,a.seeds.split(',')):
        p=root/f'formal_seed{seed}'/f'test_seed{seed}_full'/'per_image_metrics.csv'
        with p.open(encoding='utf8',newline='') as f:
            for r in csv.DictReader(f):
                key=(seed,r['dataset'],r['family'],int(r['severity']),int(r['replicate']))
                groups[key].append(r)
    rng=np.random.default_rng(a.seed);out=[]
    for (seed,ds,fam,sev,rep),rows in sorted(groups.items()):
        benefit=np.array([float(x['benefit']) for x in rows]); harm=np.array([float(x['positive_harm']) for x in rows])
        frac=np.array([float(x['selected_region_fraction']) for x in rows]);cand=np.array([float(x['candidate_mae']) for x in rows])
        n=len(rows);idx=rng.integers(0,n,size=(a.bootstrap,n))
        bmean=benefit[idx].mean(axis=1);hmean=harm[idx].mean(axis=1)
        out.append({'seed':seed,'dataset':ds,'family':fam,'severity':sev,'replicate':rep,'n_images':n,
                    'benefit_mean':benefit.mean(),'benefit_ci_low':np.quantile(bmean,.025),'benefit_ci_high':np.quantile(bmean,.975),
                    'positive_harm_mean':harm.mean(),'harm_ci_low':np.quantile(hmean,.025),'harm_ci_high':np.quantile(hmean,.975),
                    'selected_region_fraction':frac.mean(),'candidate_mae_mean':cand.mean()})
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(path,'groups',len(out),'rows',sum(x['n_images'] for x in out))

if __name__=='__main__':main()
