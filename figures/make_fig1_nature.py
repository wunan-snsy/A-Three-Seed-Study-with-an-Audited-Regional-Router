"""Data-grounded, double-column architecture figure for the RGB-D saliency study."""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE=Path(__file__).resolve().parents[1]/'paper'/'figure'
HERE.mkdir(parents=True,exist_ok=True)
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
import make_figures as src

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'pdf.fonttype':42,'svg.fonttype':'none','font.size':6.5,
                     'axes.linewidth':.6})
NAVY='#203346'; BLUE='#2D75A6'; TEAL='#148F89'; GOLD='#CC9740'; RED='#BD6570';
PALE='#F4F7F9'; EDGE='#D7E1E7'; GREY='#647587'; PURPLE='#6C65A8'

rec=src.sample_record('NJU2K','001528_left')
rgb,depth,gt=src.arrays(rec)
maps=src.model_outputs(rec,'clean',0)
rgb_img,dep_img,gt_img,ref_img,cand_img,route_mask,routed_img=maps
depth_img=np.asarray(dep_img)

fig=plt.figure(figsize=(7.2,4.5),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,100);ax.set_ylim(0,64);ax.axis('off')

def box(x,y,w,h,fc='white',ec=EDGE,rounding=1.5,lw=.7):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.18,rounding_size={rounding}',
                     facecolor=fc,edgecolor=ec,linewidth=lw,zorder=1)
    ax.add_patch(p);return p
def txt(x,y,s,size=6.5,color=NAVY,weight='normal',ha='left',va='center',z=8):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,zorder=z)
def arrow(x1,y1,x2,y2,color=BLUE,lw=1.2,style='-|>',rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,
                mutation_scale=8,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',zorder=5))
def image(x,y,w,h,data,cmap=None):
    xx=fig.add_axes([x/100,y/64,w/100,h/64]);xx.imshow(data,cmap=cmap,vmin=0 if cmap else None,
                   vmax=1 if cmap else None,interpolation='nearest');xx.axis('off')
    return xx
def stage(x,y,n,title,color):
    ax.add_patch(plt.Circle((x,y),1.6,color=color,zorder=6))
    txt(x,y,str(n),7,'white','bold','center')
    txt(x+2.5,y,title,7,NAVY,'bold')

# Strong editorial header and broad sections
ax.add_patch(Rectangle((0,57.4),100,6.6,facecolor=NAVY,edgecolor='none',zorder=0))
ax.add_patch(Rectangle((0,57.4),100,.45,facecolor=TEAL,edgecolor='none',zorder=1))
txt(2,61.3,'ROBUST RGB-D SALIENCY  /  CORRUPTION TRAINING + REGIONAL ROUTING',8,'white','bold')
txt(2,59,'Independent predictors  •  measured error utility  •  calibrated selection  •  exact missing-depth fallback',6,'#C7DDE8')
box(1.5,30.5,21,25,PALE);box(24,30.5,21,25,'#F2F8F8');box(46.5,30.5,22,25,'#F5F7FB');box(70,30.5,28.5,25,'#F9F7F2')
stage(4,53.2,1,'Inputs + depth stressors',BLUE)
stage(26.5,53.2,2,'Independent predictors',TEAL)
stage(49,53.2,3,'Regional utility',BLUE)
stage(72.5,53.2,4,'Calibrate + route',GOLD)
for xx,ww,cc in [(1.5,21,BLUE),(24,21,TEAL),(46.5,22,PURPLE),(70,28.5,GOLD)]:
    ax.add_patch(Rectangle((xx+.3,54.9),ww-.6,.55,facecolor=cc,edgecolor='none',zorder=2))

# Inputs, genuinely sourced thumbnails
image(3,36,8.5,14,rgb_img);image(12.5,36,8.5,14,depth_img,'magma')
txt(7.2,34.4,'RGB image',5.8,NAVY,'bold','center');txt(16.8,34.4,'Depth + validity',5.8,NAVY,'bold','center')
txt(3,33,'Clean · noise · holes',5.1,GREY)
txt(3,31.7,'Blur · shift · removal',5.1,GREY)
ax.add_patch(Rectangle((3.2,30.8),2.8,.35,facecolor=BLUE,edgecolor='none',zorder=3))
ax.add_patch(Rectangle((6.3,30.8),2.8,.35,facecolor=TEAL,edgecolor='none',zorder=3))
ax.add_patch(Rectangle((9.4,30.8),2.8,.35,facecolor=GOLD,edgecolor='none',zorder=3))

# Two independent pathways, no misleading sequential arrow
box(26,43.5,17,6,'white',ec='#A9D6D2');txt(34.5,47.5,'RGB reference',6.7,TEAL,'bold','center')
txt(34.5,45,'ResNet-18  →  decoder',5.6,NAVY,ha='center')
box(26,35.5,17,6,'white',ec='#A9D6D2');txt(34.5,39.5,'RGB-D candidate',6.7,TEAL,'bold','center')
txt(34.5,37,'2 encoders  →  4-scale fusion',5.3,NAVY,ha='center')
# Four decreasing feature scales make the two streams visibly computational.
for yy,colors in [(43.65,[BLUE,TEAL,BLUE,TEAL]),(35.65,[TEAL,PURPLE,TEAL,PURPLE])]:
    for k,c in enumerate(colors):
        ax.add_patch(Rectangle((29.3+k*2.8,yy),2.2,.43,facecolor=c,alpha=.85,edgecolor='none',zorder=4))
arrow(22.2,46.5,25.5,46.5);arrow(22.2,39,25.5,39)
txt(44,46.5,'R',7,TEAL,'bold','center');txt(44,39,'C',7,TEAL,'bold','center')

# Regional utility stream
image(48.5,39.2,8.3,10.3,ref_img,'gray');image(58,39.2,8.3,10.3,cand_img,'gray')
txt(52.7,37.6,'RGB output R',5.5,NAVY,'bold','center');txt(62.2,37.6,'RGB-D output C',5.5,NAVY,'bold','center')
txt(55,34.2,'16 × 16 px → 20 × 20 grid',5.1,BLUE,'bold','center')
# Utility is computed from actual seed-17 prediction errors and ground truth.
utility=(np.abs(ref_img-gt_img)-np.abs(cand_img-gt_img)).reshape(20,16,20,16).mean(axis=(1,3))
uax=fig.add_axes([63.5/100,32.1/64,2.8/100,2.8/64]);lim=max(.01,float(np.quantile(np.abs(utility),.95)))
uax.imshow(utility,cmap='coolwarm',vmin=-lim,vmax=lim,interpolation='nearest');uax.axis('off')
arrow(45,46.5,48,46.5);arrow(45,39,57.5,39,rad=.12)

# Calibration and inference (explicitly distinct)
box(72,44,24.2,6.2,'white',ec='#E2CDA2');txt(84.1,48.1,'Calibration: 218 held-out images',6.3,NAVY,'bold','center')
txt(84.1,45.8,'threshold grid  |  empirical harm ≤ 0.005',5.5,GREY,ha='center')
box(72,35,24.2,6.4,'white',ec='#E2CDA2');txt(84.1,39.1,'Inference: choose C or R per region',6.2,NAVY,'bold','center')
txt(84.1,36.6,'validity = 0  →  exact RGB fallback',5.6,RED,'bold','center')
arrow(68.5,42.4,71.4,42.4);arrow(84.1,43.7,84.1,41.6,color=GOLD)

# Three lanes make the scientific mechanism and ground-truth boundary explicit
box(1.5,19.4,97,8.4,'#F1F7F8',ec='#C3DDDB')
txt(3.2,25.2,'TRAIN',6.5,TEAL,'bold');txt(15,25.2,'RGB model: clean RGB',6.2,NAVY)
txt(43,25.2,'RGB-D model: mixed depth corruptions',6.2,NAVY)
txt(75,25.2,'Ground truth supervises both',6.0,GREY)
txt(3.2,21.7,'UTILITY',6.5,BLUE,'bold');txt(15,21.7,'Freeze both predictors; target u = MAE(R,Y) − MAE(C,Y)',6.2,NAVY)
txt(75,21.7,'8 pooled channels → score',6.0,GREY)
box(1.5,11.2,97,6.3,'#FBF6EA',ec='#E6D5AF')
txt(3.2,14.5,'CALIBRATE',6.5,GOLD,'bold');txt(20,14.5,'Choose a threshold using the separate calibration split',6.2,NAVY)
txt(70.5,14.5,'empirical budget only',6.0,RED,'bold')
box(1.5,2,97,7.3,'#EDF4FA',ec='#BFD6E5')
txt(3.2,5.8,'TEST',6.5,BLUE,'bold');txt(15,5.8,'No ground truth enters the router',6.2,NAVY)
txt(52,5.8,'Depth absent → R exactly',6.2,NAVY)
txt(77,5.8,'No safety guarantee',6.1,RED,'bold')

# Small contextual image anchored to the actual selected test ID (no synthetic photo)
txt(98,59.2,f'Illustrative test image: NJU2K {rec["id"]}',5.1,'#C7DDE8',ha='right')

fig.canvas.draw()
fig.savefig(HERE/'fig1_architecture.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(HERE/'fig1_architecture.svg',bbox_inches='tight',pad_inches=.03)
fig.savefig(HERE/'fig1_architecture.png',dpi=450,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
