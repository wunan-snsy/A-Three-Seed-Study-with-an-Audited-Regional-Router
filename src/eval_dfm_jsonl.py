import sys,json,argparse,torch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import train_dfm_jsonl as tr

def main(a):
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ds=tr.JsonlSet(a.manifest,a.root,a.size,a.test_mode,a.seed,a.max_items); dl=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False)
 m=tr.load_dfmnet(a.dfm_root)().to(dev); ck=torch.load(a.ckpt,map_location=dev); m.load_state_dict(ck['state_dict']); m.eval(); vals=[]
 with torch.no_grad():
  for rgb,d,gt,_ in dl:
   out=m(rgb.to(dev),d.to(dev))[0]; pred=torch.sigmoid(out).cpu(); vals.append(float(torch.mean(torch.abs(pred-gt))))
 res={'checkpoint':a.ckpt,'manifest':a.manifest,'condition':a.test_mode,'n':len(vals),'mae_mean':float(np.mean(vals)),'mae_std':float(np.std(vals)),'per_image_mae':vals}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(res,open(a.out,'w'),indent=2); print(res['mae_mean'],res['n'])
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--ckpt',required=True); p.add_argument('--out',required=True); p.add_argument('--root',default='.'); p.add_argument('--dfm-root',required=True,help='Path to the upstream DFM-Net repository'); p.add_argument('--size',type=int,default=256); p.add_argument('--max-items',type=int,default=0); p.add_argument('--test-mode',default='clean'); p.add_argument('--seed',type=int,default=2026); main(p.parse_args())
