"""Calibrate the UCRR regional harm budget and evaluate locked checkpoints."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from py_sod_metrics import Smeasure, WeightedFmeasure, Emeasure

from data_utils import RGBDSaliencyDataset, corrupt_depth, read_manifest, _stable_seed
from models import RGBReference, RGBDCandidate, UtilityRouter, route_predictions

FAMILIES=("clean","noise","holes","blur","shift","removal")

def loader(rows,size,seed,family,batch):
    ds=RGBDSaliencyDataset(rows,size=size,training=False,corruption=family,seed=seed)
    return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())

@torch.no_grad()
def records(refnet,candnet,router,rows,size,seed,family,severity,replicate,batch,device):
    """Stream predictions; deterministic severity/replicate corruptions are ID keyed."""
    ds=RGBDSaliencyDataset(rows,size=size,training=False,corruption="clean",seed=seed)
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    out=[]; refnet.eval();candnet.eval();router.eval()
    for b in dl:
        rgb=b['rgb'].to(device);d=b['depth'].to(device);v=b['valid'].to(device);gt=b['gt'].to(device)
        if family!="clean":
            dd=[];vv=[]
            for j,(dataset,sid) in enumerate(zip(b['dataset'],b['id'])):
                rng=np.random.default_rng(_stable_seed(str(seed),str(dataset),str(sid),family,str(severity),str(replicate)))
                dj,vj=corrupt_depth(d[j].cpu(),v[j].cpu(),family,severity,rng);dd.append(dj);vv.append(vj)
            d=torch.stack(dd).to(device);v=torch.stack(vv).to(device)
        pr=refnet(rgb);pc=candnet(rgb,d,v);score=router(pr,pc,d,v)
        for i in range(len(rgb)):
            out.append({'id':b['id'][i],'dataset':b['dataset'][i], 'ref':pr[i:i+1].cpu(), 'cand':pc[i:i+1].cpu(),
                        'score':score[i:i+1].cpu(), 'valid':v[i:i+1].cpu(),'gt':gt[i:i+1].cpu()})
    return out

def mae(pred,gt): return float((pred-gt).abs().mean())

def routed_mae(rows,threshold,region,full_metrics=True):
    vals=[];base=[];candidate_mae=[];chosen=[];harm=[]
    metrics=({name:{k:cls() for k,cls in [('sm',Smeasure),('wfm',WeightedFmeasure),('em',Emeasure)]}
              for name in ('rgb','rgbd','routed')} if full_metrics else None)
    for x in rows:
        er=F.avg_pool2d((x['ref']-x['gt']).abs(),region,region)
        ec=F.avg_pool2d((x['cand']-x['gt']).abs(),region,region)
        usable=F.avg_pool2d(x['valid'],region,region)>=0.5
        mask=((2*x['score']-1)>threshold)&usable
        b=float(er.mean());c=float(ec.mean());m=float((er+mask*(ec-er)).mean())
        base.append(b);candidate_mae.append(c);vals.append(m);chosen.append(float(mask.float().mean()));harm.append(max(0.,m-b))
        pred,_=route_predictions(x['ref'],x['cand'],x['score'],x['valid'],threshold,region)
    sod={}
    if metrics is not None:
        for x,pred in zip(rows,[route_predictions(x['ref'],x['cand'],x['score'],x['valid'],threshold,region)[0] for x in rows]):
            gt=x['gt'].squeeze().numpy().astype(bool)
            for name,p in (('rgb',x['ref']),('rgbd',x['cand']),('routed',pred)):
                arr=p.squeeze().numpy().astype(np.float32)
                for metric in metrics[name].values(): metric.step(arr,gt,normalize=False)
        for name,group in metrics.items():
            sod[f'{name}_Sm']=float(group['sm'].get_results()['sm'])
            sod[f'{name}_wFm']=float(group['wfm'].get_results()['wfm'])
            sod[f'{name}_Em']=float(group['em'].get_results()['em']['adp'])
    return {'mae':float(np.mean(vals)),'rgb_mae':float(np.mean(base)),'benefit':float(np.mean(base)-np.mean(vals)),
            'harm_rate':float(np.mean(np.asarray(harm)>1e-12)),'mean_harm':float(np.mean(harm)),
            'candidate_mae':float(np.mean(candidate_mae)) if candidate_mae else None,
            'selected_region_fraction':float(np.mean(chosen)),**sod}

def calibration_metrics(rows,threshold,region):
    # Aggregate all fixed corruption scenarios within source image before clipping harm.
    by_id={}
    for x in rows:
        er=F.avg_pool2d((x['ref']-x['gt']).abs(),region,region)
        ec=F.avg_pool2d((x['cand']-x['gt']).abs(),region,region)
        usable=F.avg_pool2d(x['valid'],region,region)>=0.5
        mask=((2*x['score']-1)>threshold)&usable
        b=float(er.mean());m=float((er+mask*(ec-er)).mean())
        q=by_id.setdefault(x['id'],{'base':[],'routed':[],'selected':[]})
        q['base'].append(b);q['routed'].append(m);q['selected'].append(float(mask.float().mean()))
    delta=np.array([np.mean(v['routed'])-np.mean(v['base']) for v in by_id.values()])
    return {'mae':float(np.mean([np.mean(v['routed']) for v in by_id.values()])),
            'rgb_mae':float(np.mean([np.mean(v['base']) for v in by_id.values()])),
            'benefit':float(-delta.mean()),'harm_rate':float(np.mean(delta>1e-12)),
            'mean_harm':float(np.maximum(delta,0).mean()),
            'selected_region_fraction':float(np.mean([np.mean(v['selected']) for v in by_id.values()])),
            'n_calibration_images':len(by_id),'n_image_condition_pairs':len(rows)}

def per_image_rows(rows,threshold,region,dataset,family,severity,replicate):
    out=[]
    for x in rows:
        er=F.avg_pool2d((x['ref']-x['gt']).abs(),region,region)
        ec=F.avg_pool2d((x['cand']-x['gt']).abs(),region,region)
        usable=F.avg_pool2d(x['valid'],region,region)>=0.5
        mask=((2*x['score']-1)>threshold)&usable
        b=float(er.mean());c=float(ec.mean());m=float((er+mask*(ec-er)).mean())
        out.append({'dataset':dataset,'id':x['id'],'family':family,'severity':severity,'replicate':replicate,
                    'rgb_mae':b,'candidate_mae':c,'routed_mae':m,'benefit':b-m,
                    'positive_harm':max(0.,m-b),'selected_region_fraction':float(mask.float().mean())})
    return out

def calibrate(cache,region,budget):
    # The budget is an upper bound on mean per-image positive MAE harm on calibration data.
    # Fixed 0.02 grid was declared in experiment_config.json before test evaluation.
    thresholds=np.linspace(-1.0,1.0,101).tolist()
    # Pool pixel errors once, then score all thresholds on tiny region grids.
    per_id={}
    for x in cache:
        er=F.avg_pool2d((x['ref']-x['gt']).abs(),region,region).squeeze().numpy().reshape(-1)
        ec=F.avg_pool2d((x['cand']-x['gt']).abs(),region,region).squeeze().numpy().reshape(-1)
        valid=(F.avg_pool2d(x['valid'],region,region).squeeze().numpy().reshape(-1)>=0.5)
        score=(2*x['score']-1).squeeze().numpy().reshape(-1)
        choose=(np.asarray(thresholds)[:,None] < score[None,:]) & valid[None,:]
        q=per_id.setdefault(x['id'],{'delta':np.zeros(len(thresholds)), 'base':0.,'selected':np.zeros(len(thresholds)),'count':0})
        q['delta']+=((ec-er)[None,:]*choose).mean(axis=1);q['base']+=float(er.mean())
        q['selected']+=choose.mean(axis=1);q['count']+=1
    delta=np.stack([q['delta']/q['count'] for q in per_id.values()])
    base=np.array([q['base']/q['count'] for q in per_id.values()])
    chosen=np.stack([q['selected']/q['count'] for q in per_id.values()])
    candidates=[]
    for i,t in enumerate(thresholds):
        m={'mae':float(np.mean(base+delta[:,i])),'rgb_mae':float(base.mean()),'benefit':float(-delta[:,i].mean()),
           'harm_rate':float(np.mean(delta[:,i]>1e-12)),'mean_harm':float(np.maximum(delta[:,i],0).mean()),
           'selected_region_fraction':float(chosen[:,i].mean()),'n_calibration_images':len(per_id),
           'n_image_condition_pairs':len(cache)}
        candidates.append((m['benefit'],t,m))
    eligible=[z for z in candidates if z[2]['mean_harm']<=budget+1e-12]
    # Among harm-budget compliant thresholds, maximize calibration benefit; conservative tie-break.
    best=max(eligible,key=lambda z:(z[0],z[1])) if eligible else candidates[-1]
    return {'threshold':best[1],'harm_budget':budget,'calibration':best[2],
            'criterion':'maximize calibration mean MAE benefit subject to mean per-image positive MAE harm <= budget',
            'n_calibration':len(cache)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--project-root',default='.');ap.add_argument('--size',type=int,default=320)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--budget',type=float,default=0.005);ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--calibrate',action='store_true')
    ap.add_argument('--threshold',type=float);ap.add_argument('--families',default='clean,noise,holes,blur,shift,removal')
    ap.add_argument('--severities',default='1,2,3');ap.add_argument('--replicates',type=int,default=1)
    ap.add_argument('--calibration-conditions',default='clean,noise:1,noise:2,noise:3,holes:1,holes:2,holes:3,blur:1,blur:2,blur:3,shift:1,shift:2,shift:3,removal')
    ap.add_argument('--mae-only',action='store_true',help='Skip slow S-measure, weighted F-measure, and E-measure calculations')
    a=ap.parse_args();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);size=ck.get('size',a.size);region=ck.get('region_size',16)
    rn=RGBReference(False);cn=RGBDCandidate(False);un=UtilityRouter(region)
    rn.load_state_dict(ck['reference']);cn.load_state_dict(ck['candidate']);un.load_state_dict(ck['router'])
    rn.to(device);cn.to(device);un.to(device)
    manifest_paths=[x for x in a.manifest.split(',') if x]
    rows=[]
    for manifest_path in manifest_paths: rows.extend(read_manifest(manifest_path,a.project_root))
    if a.calibrate:
        rows=[r for r in rows if r.get('partition')=='calibration']
    else:
        rows=[r for r in rows if r.get('partition')=='test']
    if a.limit: rows=rows[:a.limit]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.calibrate:
        if not rows: raise ValueError('No calibration rows found in manifest')
        cal=[]
        for condition in a.calibration_conditions.split(','):
            pieces=condition.split(':');family=pieces[0];severity=int(pieces[1]) if len(pieces)>1 else 0
            for rep in range(a.replicates):
                cal.extend(records(rn,cn,un,rows,size,a.seed,family,severity,rep,a.batch_size,device))
        result=calibrate(cal,region,a.budget);result['n_calibration']=len({x['id'] for x in cal})
        result.update({'seed':a.seed,'checkpoint':str(Path(a.checkpoint).resolve()),'manifest_paths':[str(Path(p).resolve()) for p in manifest_paths],'size':size,'region_size':region,
                       'calibration_conditions':a.calibration_conditions.split(',')})
        (out/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2));return
    if a.threshold is None: raise ValueError('Testing requires --threshold from a frozen calibration.json')
    if not rows: raise ValueError('No test rows found in manifest')
    fams=a.families.split(',');sevs=[int(x) for x in a.severities.split(',')];result=[];image_result=[];baseline_result=[]
    for fam in fams:
        levels=[0] if fam in ('clean','removal') else sevs
        for sev in levels:
            for rep in range(a.replicates):
                data=records(rn,cn,un,rows,size,a.seed,fam,sev,rep,a.batch_size,device)
                for ds in sorted({x['dataset'] for x in data}):
                    subset=[x for x in data if x['dataset']==ds]
                    result.append({'dataset':ds,'family':fam,'severity':sev,'replicate':rep,**routed_mae(subset,a.threshold,region,full_metrics=(fam=='clean' and not a.mae_only))})
                    image_result.extend(per_image_rows(subset,a.threshold,region,ds,fam,sev,rep))
                    for policy,threshold in (('fixed_zero',0.0),('validity_only',-1.0),('rgb_only',1.0)):
                        baseline_result.append({'policy':policy,'dataset':ds,'family':fam,'severity':sev,'replicate':rep,
                                                **routed_mae(subset,threshold,region,full_metrics=False)})
    path=out/'evaluation.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    with (out/'per_image_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(image_result[0]));w.writeheader();w.writerows(image_result)
    with (out/'gate_baselines.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(baseline_result[0]));w.writeheader();w.writerows(baseline_result)
    meta={'checkpoint':str(Path(a.checkpoint).resolve()),'manifest_paths':[str(Path(p).resolve()) for p in manifest_paths],'threshold':a.threshold,'seed':a.seed,
          'size':size,'region_size':region,'n_images':len(rows),'families':fams,'severities':sevs,'replicates':a.replicates,
          'metrics':'MAE, RGB-reference MAE, mean benefit, per-image positive-harm rate, mean positive MAE harm, routed-region fraction'}
    (out/'evaluation_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(path)

if __name__=='__main__':main()
