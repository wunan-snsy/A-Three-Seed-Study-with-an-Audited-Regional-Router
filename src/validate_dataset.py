"""Run data manifest, loader, shape and corruption checks on audited real files."""
import argparse, json
from collections import Counter
from pathlib import Path
import torch
from data_utils import read_manifest, RGBDSaliencyDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--project-root',default='.');p.add_argument('--limit',type=int,default=24);p.add_argument('--size',type=int,default=320);a=p.parse_args()
 rows=read_manifest(a.manifest,a.project_root); rows=rows[:a.limit]
 ds=RGBDSaliencyDataset(rows,size=a.size,training=False)
 shapes=Counter();bad=[]
 for i in range(len(ds)):
  try:
   s=ds[i]
   shapes['x'.join(map(str,s['rgb'].shape))]+=1
   assert s['depth'].shape==(1,a.size,a.size) and s['valid'].shape==s['depth'].shape and s['gt'].shape==s['depth'].shape
   assert torch.isfinite(s['rgb']).all() and torch.isfinite(s['depth']).all() and torch.isfinite(s['gt']).all()
   assert set(torch.unique(s['gt']).tolist()).issubset({0.,1.})
  except Exception as e: bad.append({'index':i,'id':rows[i].get('id'),'error':repr(e)})
 print(json.dumps({'checked':len(rows),'bad':bad,'tensor_shapes':dict(shapes),'pass':not bad},indent=2));
 if bad: raise SystemExit(1)
if __name__=='__main__':main()
