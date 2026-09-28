"""Train an RGB-D candidate without synthetic depth corruption (training ablation)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

from data_utils import read_manifest,RGBDSaliencyDataset
from models import RGBDCandidate
from train_ucrr import seed_all,batches,make_loader,train_seg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--project-root',default='.')
    ap.add_argument('--reference-run',required=True);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--size',type=int,default=320);ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--out',required=True)
    a=ap.parse_args();seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=read_manifest(a.manifest,a.project_root);fit=batches(rows,'fit');val=batches(rows,'validation')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train_loader=make_loader(fit,True,a.size,a.seed,'clean',a.batch_size)
    val_loader=make_loader(val,False,a.size,a.seed,'clean',a.batch_size)
    model=RGBDCandidate(True).to(device)
    best=train_seg(model,train_loader,val_loader,device,a.epochs,1e-4,out,'rgbd',a.seed,False)
    candidate=torch.load(out/'rgbd_best.pt',map_location=device,weights_only=False)['model']
    main_ck=torch.load(Path(a.reference_run)/'models.pt',map_location='cpu',weights_only=False)
    composed={'reference':main_ck['reference'],'candidate':candidate,'router':main_ck['router'],
              'seed':a.seed,'size':a.size,'region_size':main_ck.get('region_size',16),
              'ablation':'RGB-D candidate trained on clean depth only; router/reference retained from matching main seed',
              'initialization_note':'ResNet-18 pretrained initialization is the same source; random decoder/fusion initialization is independently seeded.'}
    torch.save(composed,out/'models.pt')
    meta={'seed':a.seed,'size':a.size,'batch_size':a.batch_size,'epochs':a.epochs,
          'fit_count':len(fit),'validation_count':len(val),'candidate_training_corruption':'clean only',
          'reference_run':str(Path(a.reference_run).resolve()),'best_validation_mae':best,
          'initialization_note':composed['initialization_note'],
          'status':'training ablation; evaluate candidate-only MAE against matched robust-training candidate'}
    (out/'ablation_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Saved clean-training ablation checkpoint to',out/'models.pt')

if __name__=='__main__':main()
