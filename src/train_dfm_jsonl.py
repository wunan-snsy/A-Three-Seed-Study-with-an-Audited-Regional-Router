import os,sys,json,argparse,random,time,csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
def load_dfmnet(dfm_root):
    """Load DFM-Net from a separately cloned upstream repository."""
    sys.path.insert(0, str(Path(dfm_root).resolve()))
    import net
    net.initialize_weights = lambda model: None
    return net.DFMNet

class JsonlSet(Dataset):
    def __init__(self, manifest, root, size=128, mode='clean', seed=2026, max_items=0, partition=None):
        self.root=Path(root); self.size=size; self.mode=mode; self.seed=seed; self.epoch=0
        self.rows=[json.loads(x) for x in open(manifest,encoding='utf-8') if x.strip()]
        if partition is not None: self.rows=[r for r in self.rows if r.get('partition')==partition]
        if max_items: self.rows=self.rows[:max_items]
    def set_epoch(self, epoch): self.epoch=epoch
    def __len__(self): return len(self.rows)
    def _p(self, s):
        p=Path(s)
        return p if p.is_absolute() else self.root/p
    def _corrupt_depth(self, a, idx):
        if self.mode=='clean': return a
        rng=np.random.default_rng(self.seed + idx + self.epoch * 1000003)
        x=a.astype(np.float32)/255.
        parts=self.mode.split('_'); fam=parts[0]; sev=int(parts[1]) if len(parts)>1 and parts[1].isdigit() else 2
        if self.mode in ('corruption','mixture'):
            fam=('clean','noise','holes','blur','shift','removal')[int(rng.integers(0,6))]
            sev=int(rng.integers(1,4))
        if fam=='clean': return a
        if fam=='removal': return np.zeros_like(a)
        if fam=='noise': x=np.clip(x+rng.normal(0,[0.03,0.08,0.15][max(1,min(3,sev))-1],x.shape),0,1)
        elif fam=='holes':
            m=rng.random(x.shape)<[0.05,0.15,0.30][max(1,min(3,sev))-1]; x[m]=0
        elif fam=='blur': x=np.asarray(Image.fromarray((x*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur([0.5,1.0,2.0][max(1,min(3,sev))-1])),dtype=np.float32)/255.
        elif fam=='shift': x=np.roll(x,shift=[1,2,4][max(1,min(3,sev))-1],axis=1)
        return (x*255).astype(np.uint8)
    def __getitem__(self,i):
        r=self.rows[i]; rgb=np.asarray(Image.open(self._p(r['paths']['rgb'])).convert('RGB').resize((self.size,self.size),Image.BILINEAR),dtype=np.float32)/255.
        d=np.asarray(Image.open(self._p(r['paths']['depth'])).convert('L').resize((self.size,self.size),Image.BILINEAR))
        d=self._corrupt_depth(d,i); d=torch.from_numpy(d.astype(np.float32)/255.).unsqueeze(0)
        gt=np.asarray(Image.open(self._p(r['paths']['gt'])).convert('L').resize((self.size,self.size),Image.NEAREST),dtype=np.float32)/255.
        return torch.from_numpy(rgb.transpose(2,0,1)),d,torch.from_numpy(gt[None]),r['id']

def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev)
    ds=JsonlSet(args.manifest,args.root,args.size,args.mode,args.seed,args.max_items,args.partition)
    val_ds=JsonlSet(args.manifest,args.root,args.size,'clean',args.seed,0,'validation') if args.validation else None
    # DFM-Net contains BatchNorm layers after global pooling; a final batch of
    # one image is invalid in training mode, so discard only that incomplete batch.
    dl=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=0,
                  pin_memory=(dev.type=='cuda'),drop_last=True)
    val_dl=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=(dev.type=='cuda')) if val_ds else None
    model=load_dfmnet(args.dfm_root)().to(dev); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    history=[]; t0=time.time(); start=1; best=float('inf')
    if args.resume and Path(args.resume).exists():
        state=torch.load(args.resume,map_location=dev); model.load_state_dict(state['state_dict']); opt.load_state_dict(state['optimizer'])
        start=int(state['epoch'])+1; history=state.get('history',[]); best=float(state.get('best_val_mae',best))
    for ep in range(start,args.epochs+1):
        ds.set_epoch(ep)
        model.train(); total=0.; n=0
        for rgb,d,gt,_ in dl:
            rgb,d,gt=rgb.to(dev),d.to(dev),gt.to(dev)
            opt.zero_grad(set_to_none=True); outs=model(rgb,d)
            aux=outs[1].mean(1,keepdim=True) if outs[1].shape[1] != 1 else outs[1]; loss=F.binary_cross_entropy_with_logits(outs[0],gt)+F.binary_cross_entropy_with_logits(aux,gt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            total+=float(loss.detach())*rgb.size(0); n+=rgb.size(0)
        train_loss=total/max(n,1); val_mae=None
        if val_dl is not None:
            model.eval(); vals=[]
            with torch.no_grad():
                for rgb,d,gt,_ in val_dl:
                    pred=torch.sigmoid(model(rgb.to(dev),d.to(dev))[0]).cpu(); vals.append(float((pred-gt).abs().mean()))
            val_mae=float(np.mean(vals))
        history.append({'epoch':ep,'loss':train_loss,'validation_clean_mae':val_mae}); print(args.mode,ep,train_loss,val_mae,flush=True)
        state={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'history':history,'best_val_mae':min(best,val_mae if val_mae is not None else best),
               'mode':args.mode,'size':args.size,'seed':args.seed,'manifest':str(args.manifest),'partition':args.partition,
               'protocol_version':'dfm_full_v2_actual_corruption_mixture'}
        torch.save(state,out/'last.pth')
        if val_mae is not None and val_mae < best: best=val_mae; state['best_val_mae']=best; torch.save(state,out/'best.pth')
        if ep==args.epochs: torch.save(state,out/f'epoch_{ep}.pth')
        json.dump({'mode':args.mode,'epochs_completed':ep,'target_epochs':args.epochs,'items':len(ds),'validation_items':len(val_ds) if val_ds else 0,
                   'history':history,'seconds':time.time()-t0,'device':str(dev),'protocol_version':'dfm_full_v2_actual_corruption_mixture'},open(out/'train_summary.json','w'),indent=2)

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--root',default='.'); p.add_argument('--dfm-root',required=True,help='Path to the upstream DFM-Net repository'); p.add_argument('--out',required=True); p.add_argument('--mode',choices=['clean','corruption'],default='clean'); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=2); p.add_argument('--size',type=int,default=128); p.add_argument('--max-items',type=int,default=0); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=2026); p.add_argument('--partition',default='fit'); p.add_argument('--validation',action='store_true'); p.add_argument('--resume'); run(p.parse_args())
