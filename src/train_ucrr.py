"""Train the three stages of UCRR and save auditable checkpoints/logs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import RGBDSaliencyDataset, read_manifest
from models import RGBReference, RGBDCandidate, UtilityRouter, utility_target, segmentation_loss


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def batches(rows, partition):
    return [r for r in rows if r.get("partition") == partition]


def make_loader(rows, training, size, seed, family="clean", batch_size=2):
    ds = RGBDSaliencyDataset(rows, size=size, training=training, corruption=family, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=0,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def validate_seg(model, loader, device, mode):
    model.eval(); total = 0.; n = 0
    for b in loader:
        rgb, depth, valid, gt = (b[k].to(device, non_blocking=True) for k in ("rgb", "depth", "valid", "gt"))
        pred = model(rgb) if mode == "rgb" else model(rgb, depth, valid)
        total += float((pred - gt).abs().mean(dim=(1, 2, 3)).sum()); n += len(rgb)
    return total / max(n, 1)


def train_seg(model, train_loader, val_loader, device, epochs, lr, out, mode, seed, resume=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = float("inf"); start_epoch=1
    last_path=out/f"{mode}_last.pt"
    if resume and last_path.exists():
        ck=torch.load(last_path,map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start_epoch=int(ck['epoch'])+1;best=float(ck.get('best_val_mae',float('inf')))
        if ck.get('scheduler'): sched.load_state_dict(ck['scheduler'])
        print(json.dumps({'stage':mode,'resume_epoch':start_epoch}),flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train(); loss_sum=0.; n=0; start=time.time()
        for b in train_loader:
            rgb, depth, valid, gt = (b[k].to(device, non_blocking=True) for k in ("rgb", "depth", "valid", "gt"))
            opt.zero_grad(set_to_none=True)
            pred = model(rgb) if mode == "rgb" else model(rgb, depth, valid)
            loss = segmentation_loss(pred, gt); loss.backward(); opt.step()
            loss_sum += float(loss.detach()) * len(rgb); n += len(rgb)
        sched.step(); val = validate_seg(model, val_loader, device, mode)
        rec={"epoch":epoch,"train_loss":loss_sum/max(n,1),"val_mae":val,"seconds":time.time()-start,"stage":mode}
        print(json.dumps(rec), flush=True)
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"best_val_mae":best,"seed":seed},last_path)
        if val < best:
            best=val; torch.save({"epoch":epoch,"model":model.state_dict(),"val_mae":val},out/f"{mode}_best.pt")
    return best


@torch.no_grad()
def cache_router_data(router, reference, candidate, loader, device):
    """Cache only region-grid router features/targets (small, reproducible tensors)."""
    feats=[];targets=[];router.eval();reference.eval();candidate.eval()
    for b in loader:
        rgb,depth,valid,gt=(b[k].to(device,non_blocking=True) for k in ("rgb","depth","valid","gt"))
        ref=reference(rgb);cand=candidate(rgb,depth,valid)
        feats.append(router.features(ref,cand,depth,valid).cpu())
        targets.append(utility_target(ref,cand,gt,router.region_size).cpu())
    return torch.cat(feats),torch.cat(targets)


def fit_router(router, train_cache, val_cache, device, epochs, lr, out, resume=False):
    opt=torch.optim.AdamW(router.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');start_epoch=1;last_path=out/'router_last.pt'
    if resume and last_path.exists():
        ck=torch.load(last_path,map_location=device,weights_only=False)
        router.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start_epoch=int(ck['epoch'])+1;best=float(ck.get('best_val_loss',float('inf')))
        print(json.dumps({'stage':'router','resume_epoch':start_epoch}),flush=True)
    for epoch in range(start_epoch,epochs+1):
        router.train(); total=0.; n=0
        features,targets=train_cache;order=torch.randperm(len(features))
        for ids in order.split(32):
            x=features[ids].to(device,non_blocking=True);target=targets[ids].to(device,non_blocking=True)
            score=router.predict_features(x)
            loss=torch.nn.functional.mse_loss(score,target)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            total+=float(loss.detach())*len(x);n+=len(x)
        val_loss=None
        if val_cache is not None:
            router.eval();vtotal=0.;vn=0
            with torch.no_grad():
                features,targets=val_cache
                for ids in torch.arange(len(features)).split(64):
                    x=features[ids].to(device,non_blocking=True);target=targets[ids].to(device,non_blocking=True)
                    pred=router.predict_features(x)
                    vtotal+=float(torch.nn.functional.mse_loss(pred,target,reduction='sum'));vn+=target.numel()
            val_loss=vtotal/max(vn,1)
        print(json.dumps({"stage":"router","epoch":epoch,"utility_mse":total/max(n,1),"val_utility_mse":val_loss}),flush=True)
        if val_loss is None or val_loss < best:
            best=val_loss if val_loss is not None else best
            torch.save({"epoch":epoch,"model":router.state_dict(),"best_val_loss":best},out/"router_best.pt")
        torch.save({"epoch":epoch,"model":router.state_dict(),"optimizer":opt.state_dict(),"best_val_loss":best},last_path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--project-root',default='.')
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--size',type=int,default=320)
    ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--no-pretrained',action='store_true')
    ap.add_argument('--router-corruption',choices=['clean','noise','holes','blur','shift','removal','mixture'],default='mixture')
    ap.add_argument('--out',default='output/runs/smoke_seed17')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();seed_all(args.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest_paths=[x for x in args.manifest.split(',') if x]
    rows=[]
    for manifest_path in manifest_paths: rows.extend(read_manifest(manifest_path,args.project_root))
    fit, val, cal=(batches(rows,k) for k in ('fit','validation','calibration'))
    if args.limit:
        fit=fit[:args.limit];val=val[:max(2,args.limit//4)];cal=cal[:max(2,args.limit//4)]
    if not fit or not val or not cal: raise ValueError(f"empty partition: fit={len(fit)} val={len(val)} cal={len(cal)}")
    ref_train_loader=make_loader(fit,True,args.size,args.seed,'clean',args.batch_size)
    cand_train_loader=make_loader(fit,True,args.size,args.seed,'mixture',args.batch_size)
    val_loader=make_loader(val,False,args.size,args.seed,'clean',args.batch_size)
    meta={"args":vars(args),"device":str(dev),"torch":torch.__version__,"cuda":torch.version.cuda,
          "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "pretrained_requested":not args.no_pretrained,
          "pretrained_resnet18":"https://download.pytorch.org/models/resnet18-f37072fd.pth" if not args.no_pretrained else None,
          "partition_counts":{"fit":len(fit),"validation":len(val),"calibration":len(cal)},
          "manifest_paths":[str(Path(p).resolve()) for p in manifest_paths],
          "status":"training pipeline run; benchmark claims require locked calibration and test evaluation"}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    pretrained=not args.no_pretrained
    reference=RGBReference(pretrained).to(dev)
    train_seg(reference,ref_train_loader,val_loader,dev,args.epochs,1e-4,out,'rgb',args.seed,args.resume)
    ck=torch.load(out/'rgb_best.pt',map_location=dev,weights_only=False);reference.load_state_dict(ck['model'])
    candidate=RGBDCandidate(pretrained).to(dev)
    train_seg(candidate,cand_train_loader,val_loader,dev,args.epochs,1e-4,out,'rgbd',args.seed,args.resume)
    ck=torch.load(out/'rgbd_best.pt',map_location=dev,weights_only=False);candidate.load_state_dict(ck['model'])
    for p in reference.parameters():p.requires_grad_(False)
    for p in candidate.parameters():p.requires_grad_(False)
    fit_router_loader=make_loader(fit,True,args.size,args.seed,args.router_corruption,args.batch_size)
    # Keep router early-stopping input fixed; corruption is varied in fitting, not validation.
    val_router_loader=make_loader(val,False,args.size,args.seed,'clean',args.batch_size)
    router=UtilityRouter(16).to(dev)
    print('Caching compact router features...',flush=True)
    train_cache=cache_router_data(router,reference,candidate,fit_router_loader,dev)
    val_cache=cache_router_data(router,reference,candidate,val_router_loader,dev)
    fit_router(router,train_cache,val_cache,dev,args.epochs,1e-4,out,args.resume)
    if (out/'router_best.pt').exists():
        router.load_state_dict(torch.load(out/'router_best.pt',map_location=dev,weights_only=False)['model'])
    torch.save({'reference':reference.state_dict(),'candidate':candidate.state_dict(),'router':router.state_dict(),
                'seed':args.seed,'size':args.size,'region_size':router.region_size},out/'models.pt')
    print('Peak CUDA memory MiB:',torch.cuda.max_memory_allocated()/1024**2 if dev.type=='cuda' else 0)

if __name__=='__main__':main()
