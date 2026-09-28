"""Rebuild five manuscript figures from saved experiment artifacts."""
from __future__ import annotations

import csv, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper'/'figure'
RUN=ROOT/'results'
sys.path.insert(0,str(ROOT/'src'))
from data_utils import read_manifest, _load_rgb, _load_gray, corrupt_depth, _stable_seed
from models import RGBReference, RGBDCandidate, UtilityRouter, route_predictions

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'pdf.fonttype':42,'svg.fonttype':'none','font.size':7,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                     'savefig.facecolor':'white'})
BLUE='#2774AE'; TEAL='#208E8A'; AMBER='#D19A29'; RED='#BE5866'; INK='#1F2D3D'; MUTED='#667788'
SEEDS=(17,29,43); DATASETS=('NJU2K','NLPR','SIP')

def rows(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.canvas.draw()
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.png',dpi=400,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def panel(ax,label,title):
    ax.text(0,.98,label,transform=ax.transAxes,weight='bold',fontsize=10,va='top',color=INK)
    ax.text(.08,.98,title,transform=ax.transAxes,weight='bold',fontsize=8,va='top',color=INK)

def fig1():
    fig=plt.figure(figsize=(11.2,5.1)); ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    cards=[('1','INPUTS','#E9F2FB'),('2','RGB REFERENCE','#ECF7F3'),('3','RGB-D CANDIDATE','#FDF0F0'),
           ('4','REGIONAL UTILITY','#FBF5E8'),('5','CALIBRATION','#EDF5FC'),('6','INFERENCE','#F2EFFB')]
    w=.153; gap=.012; x0=.014
    for i,(n,title,bg) in enumerate(cards):
        x=x0+i*(w+gap)
        ax.add_patch(FancyBboxPatch((x,.14),w,.78,boxstyle='round,pad=0.008,rounding_size=.018',fc=bg,ec='#D4DDE5',lw=.8))
        ax.add_patch(plt.Circle((x+.023,.875),.016,color=BLUE if i in (0,4) else [TEAL,RED,AMBER,BLUE,'#756AB1'][min(i-1,4)]))
        ax.text(x+.023,.875,n,ha='center',va='center',color='white',weight='bold',fontsize=9)
        ax.text(x+.043,.875,title,ha='left',va='center',fontsize=8.5,weight='bold',color=INK)
        if i<5 and i!=1:ax.add_patch(FancyArrowPatch((x+w+.002,.53),(x+w+gap-.002,.53),arrowstyle='-|>',mutation_scale=11,lw=1,color=BLUE))
    def box(i,y,s,h=.095,fc='white',fs=7):
        x=x0+i*(w+gap)+.012
        ax.add_patch(FancyBboxPatch((x,y),w-.024,h,boxstyle='round,pad=.005',fc=fc,ec='#C7D3DF',lw=.7))
        ax.text(x+(w-.024)/2,y+h/2,s,ha='center',va='center',fontsize=fs,color=INK,linespacing=1.3)
    box(0,.69,'RGB image  I');box(0,.55,'Depth  D  +  validity  V');box(0,.33,'Ground truth  Y\n(training / evaluation only)',h=.12,fc='#FFF8E9')
    box(1,.67,'ImageNet ResNet-18\nRGB encoder',h=.14);box(1,.44,'4-scale top-down\ndecoder',h=.14);box(1,.23,'RGB probability  R')
    box(2,.66,'RGB encoder   |   D,V encoder\n(two ResNet-18 streams)',h=.15);box(2,.44,'Four-scale feature fusion\n+ top-down decoder',h=.14);box(2,.23,'RGB-D probability  C')
    box(3,.70,'Freeze  R  and  C');box(3,.50,'Target: regional\nMAE(R) - MAE(C)',h=.14);box(3,.26,'8-channel features\n20 x 20 route-score grid',h=.14)
    box(4,.66,'218 held-out images\n14 fixed conditions',h=.13);box(4,.46,'Threshold grid search\nharm budget = 0.005',h=.13);box(4,.25,'Selected threshold\nper trained seed',h=.12)
    box(5,.67,'If valid depth absent:\nreturn R exactly',h=.14);box(5,.44,'Else select C or R\nper 16 x 16 pixel region',h=.14);box(5,.23,'Final saliency map')
    ax.text(.18,.945,'Independent parallel predictors',color=TEAL,fontsize=8,weight='bold')
    ax.text(.015,.075,'Training uses ground truth to fit predictors and utility head',color=AMBER,fontsize=8,weight='bold')
    ax.text(.67,.075,'Test inference uses no ground truth',color=BLUE,fontsize=8,weight='bold')
    save(fig,'fig1_architecture')

def manifest(dataset):
    p=ROOT/'manifests'/f'{dataset.lower()}_test_official.jsonl'
    return read_manifest(p,ROOT)
def sample_record(dataset,sid):
    return next(r for r in manifest(dataset) if str(r['id'])==str(sid))
def arrays(rec):
    rgb=_load_rgb(rec['rgb'],320);depth=_load_gray(rec['depth'],320);gt=(_load_gray(rec['gt'],320,True)>=.5).float();return rgb,depth,gt
def visrgb(t):return np.clip(t.permute(1,2,0).numpy(),0,1)
def visgray(t):return t.squeeze().numpy()

def fig2():
    rec=manifest('NJU2K')[0];rgb,d,gt=arrays(rec);v=torch.ones_like(d)
    states=[('Clean','clean',0),('Noise 3','noise',3),('Holes 3','holes',3),('Blur 3','blur',3),('Shift 3','shift',3),('Removal','removal',0)]
    fig,axs=plt.subplots(3,6,figsize=(11.2,5.5),gridspec_kw={'height_ratios':[1,1,1]})
    for j,(title,fam,sev) in enumerate(states):
        rng=np.random.default_rng(_stable_seed('17','NJU2K',str(rec['id']),fam,str(sev),'0'))
        dd,vv=corrupt_depth(d,v,fam,sev,rng)
        for i,arr in enumerate((visrgb(rgb),visgray(dd),visgray(vv))):
            axs[i,j].imshow(arr,cmap=None if i==0 else ('magma' if i==1 else 'gray'),vmin=None if i==0 else 0,vmax=None if i==0 else 1,interpolation='nearest')
            axs[i,j].axis('off')
        axs[0,j].set_title(title,weight='bold',fontsize=8,color=INK)
    for i,title in enumerate(('RGB unchanged','Depth input','Validity mask')):
        axs[i,0].text(-.04,.5,title,transform=axs[i,0].transAxes,ha='right',va='center',rotation=90,fontsize=8,weight='bold',color=INK)
    fig.subplots_adjust(left=.075,right=.995,top=.91,bottom=.08,wspace=.035,hspace=.09)
    fig.text(.075,.025,f'NJU2K test ID: {rec["id"]}  |  severity 3 shown for each family  |  operators reproduced from saved code',fontsize=7,color=MUTED)
    save(fig,'fig2_depth_corruptions')

def fig3():
    curves={}
    for seed in SEEDS:
        for mode,sub in [('Mixture',RUN/f'formal_seed{seed}'/f'test_seed{seed}_full'/'evaluation.csv'),('Clean-only',RUN/f'ablation_clean_candidate_seed{seed}'/'test'/'evaluation.csv')]:
            for r in rows(sub):curves[(seed,mode,r['dataset'],r['family'],int(r['severity']))]=float(r['candidate_mae'])
    fig,axs=plt.subplots(3,4,figsize=(11.2,7.2),sharex=True)
    for i,ds in enumerate(DATASETS):
        for j,fam in enumerate(('noise','holes','blur','shift')):
            ax=axs[i,j]
            for mode,color,marker in [('Clean-only',MUTED,'o'),('Mixture',TEAL,'s')]:
                vals=np.array([[curves[(seed,mode,ds,fam,k)] for k in (1,2,3)] for seed in SEEDS])
                mean=vals.mean(0);sd=vals.std(0,ddof=1)
                for row in vals:ax.plot([1,2,3],row,color=color,alpha=.18,lw=.8)
                ax.errorbar([1,2,3],mean,yerr=sd,color=color,marker=marker,lw=1.7,ms=4,capsize=2,label=mode if i==0 and j==0 else None)
            ax.grid(axis='y',color='#E5E9ED',lw=.5)
            ax.set_xticks([1,2,3]);ax.tick_params(labelsize=6)
            if i==0:ax.set_title(fam.title(),fontsize=9,weight='bold')
            if j==0:ax.set_ylabel(f'{ds}\nCandidate MAE',fontsize=8,weight='bold')
            if i==2:ax.set_xlabel('Severity',fontsize=7)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.005),fontsize=8)
    fig.subplots_adjust(left=.08,right=.99,top=.93,bottom=.12,wspace=.25,hspace=.32)
    fig.text(.08,.02,'Mean ± sample SD across 3 independent model seeds; faint lines show individual seeds. One deterministic corruption draw per image and condition.',fontsize=7,color=MUTED)
    save(fig,'fig3_corruption_training')

def fig4():
    labels=[('clean',0,'Clean'),('noise',3,'Noise 3'),('holes',3,'Holes 3'),('blur',3,'Blur 3'),('shift',3,'Shift 3'),('removal',0,'Removal')]
    metrics={}
    for seed in SEEDS:
        for r in rows(RUN/f'formal_seed{seed}'/f'test_seed{seed}_full'/'evaluation.csv'):
            metrics[(seed,r['dataset'],r['family'],int(r['severity']))]=r
    fig,axs=plt.subplots(3,3,figsize=(11.2,6.9))
    for i,ds in enumerate(DATASETS):
        for j,(col,title,color) in enumerate([('benefit','Benefit vs RGB',TEAL),('mean_harm','Positive harm',RED),('selected_region_fraction','Depth selected',BLUE)]):
            ax=axs[i,j];x=np.arange(len(labels))
            arr=np.array([[float(metrics[(seed,ds,fam,sev)][col]) for fam,sev,_ in labels] for seed in SEEDS])
            mean=arr.mean(0);sd=arr.std(0,ddof=1)
            ax.bar(x,mean,color=color,alpha=.82,width=.72)
            ax.errorbar(x,mean,yerr=sd,fmt='none',ecolor=INK,capsize=2,lw=.75)
            if j==1:ax.axhline(.005,color=AMBER,ls='--',lw=1.2)
            ax.set_xticks(x,[x[2] for x in labels],rotation=35,ha='right',fontsize=6)
            ax.grid(axis='y',color='#E5E9ED',lw=.5);ax.set_axisbelow(True)
            if i==0:ax.set_title(title,fontsize=9,weight='bold',color=INK)
            if j==0:ax.set_ylabel(ds,fontsize=9,weight='bold')
            if j==2:ax.set_ylim(0,1.35)
    fig.subplots_adjust(left=.07,right=.99,top=.93,bottom=.12,wspace=.22,hspace=.56)
    fig.text(.07,.025,'Mean ± SD over 3 seeds. Dashed line is the 0.005 calibration budget, not a test-set guarantee. Harm is image-level positive excess MAE.',fontsize=7,color=MUTED)
    save(fig,'fig4_router_diagnostics')

@torch.no_grad()
def model_outputs(rec,fam,sev):
    ck=torch.load(RUN/'formal_seed17'/'models.pt',map_location='cpu',weights_only=False)
    rn=RGBReference(False);cn=RGBDCandidate(False);un=UtilityRouter(16)
    rn.load_state_dict(ck['reference']);cn.load_state_dict(ck['candidate']);un.load_state_dict(ck['router'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rn.to(device).eval();cn.to(device).eval();un.to(device).eval()
    rgb,d,gt=arrays(rec);v=torch.ones_like(d)
    rng=np.random.default_rng(_stable_seed('17',str(rec['dataset']),str(rec['id']),fam,str(sev),'0'))
    d,v=corrupt_depth(d,v,fam,sev,rng)
    x=(rgb-torch.tensor((.485,.456,.406))[:,None,None])/torch.tensor((.229,.224,.225))[:,None,None]
    x=x[None].to(device);dd=d[None].to(device);vv=v[None].to(device)
    rr=rn(x);cc=cn(x,dd,vv);qq=un(rr,cc,dd,vv);pp,mask=route_predictions(rr,cc,qq,vv,-.1,16)
    return [visrgb(rgb),visgray(d),visgray(gt),rr.cpu().squeeze().numpy(),cc.cpu().squeeze().numpy(),mask.cpu().squeeze().numpy(),pp.cpu().squeeze().numpy()]

def fig5():
    per=rows(RUN/'formal_seed17'/'test_seed17_full'/'per_image_metrics.csv')
    def pick(fam,sev,metric,reverse=True,extra=None):
        subset=[r for r in per if r['dataset']=='NJU2K' and r['family']==fam and int(r['severity'])==sev and (extra(r) if extra else True)]
        return sorted(subset,key=metric,reverse=reverse)[0]
    helpful=pick('clean',0,lambda r:float(r['rgb_mae'])-float(r['candidate_mae']))
    harmful=pick('clean',0,lambda r:float(r['candidate_mae'])-float(r['rgb_mae']))
    shift=pick('shift',3,lambda r:float(r['routed_mae'])-float(r['rgb_mae']),extra=lambda r:r['id']!=harmful['id'])
    cases=[('Helpful depth',helpful,'clean',0),('Harmful depth',harmful,'clean',0),('Router failure',shift,'shift',3),('Depth removed',helpful,'removal',0)]
    fig,axs=plt.subplots(4,7,figsize=(11.2,6.9))
    names=['RGB','Depth','Ground truth','RGB output','RGB-D output','Route mask','Final output']
    provenance=[]
    for i,(title,r,fam,sev) in enumerate(cases):
        rec=sample_record('NJU2K',r['id']);imgs=model_outputs(rec,fam,sev)
        provenance.append({'row':i,'case':title,'dataset':'NJU2K','id':r['id'],'family':fam,'severity':sev,'seed':17})
        for j,(im,ax) in enumerate(zip(imgs,axs[i])):
            ax.imshow(im,cmap=None if j==0 else ('magma' if j==1 else ('Blues' if j==5 else 'gray')),vmin=None if j==0 else 0,vmax=None if j==0 else 1,interpolation='nearest')
            ax.axis('off')
            if i==0:ax.set_title(names[j],fontsize=7.5,weight='bold',color=INK)
        axs[i,0].text(-.04,.5,f'{title}\nID {r["id"]}',transform=axs[i,0].transAxes,ha='right',va='center',fontsize=7,weight='bold',color=INK)
    fig.subplots_adjust(left=.13,right=.995,top=.92,bottom=.06,wspace=.025,hspace=.13)
    fig.text(.13,.02,'Actual official-test images and seed-17 checkpoint. Cases selected by explicit MAE criteria; all masks at 320 x 320.',fontsize=7,color=MUTED)
    save(fig,'fig5_qualitative')
    (OUT/'fig5_case_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    fig1();fig2();fig3();fig4();fig5()
    print('Wrote five PDFs to',OUT)
