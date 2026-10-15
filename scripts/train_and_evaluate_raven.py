#!/usr/bin/env python3
"""Post-hoc absolute-risk governance for frozen YOLOv8n on LLVIP.

Real run: python train_and_evaluate_raven.py --data-root /content/LLVIP --rgb-weights yolov8n.pt --ir-weights yolov8n.pt
Smoke test: python train_and_evaluate_raven.py --synthetic

LLVIP scene IDs are not assumed: source images are deterministically partitioned
into disjoint pools before corruptions. Synthetic mode validates software only.
"""
from __future__ import annotations
import argparse, json, math, random, time, xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTS=('joint','rgb','ir')
CFG={
 'A':['c_rgb','c_ir','conf_diff','matched'],
 'B':['c_rgb','c_ir','conf_diff','matched','box_iou','center_dist','area_diff','aspect_diff'],
 'C':['c_rgb','c_ir','conf_diff','matched','box_iou','center_dist','area_diff','aspect_diff','patch_contrast_rgb','patch_contrast_ir']}

@dataclass
class Frame:
 image_id:str; gt:np.ndarray; rgb:list; ir:list; corruption:str='clean'; rgb_image:object=None; ir_image:object=None

def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return float(inter/max(aa+ab-inter,1e-12))

def read_gt(p):
 root=ET.parse(p).getroot(); out=[]
 for o in root.findall('object'):
  if (o.findtext('name') or '').lower()=='person':
   b=o.find('bndbox'); out.append([float(b.findtext(k,'0')) for k in ('xmin','ymin','xmax','ymax')])
 return np.asarray(out,np.float32).reshape(-1,4)

def corrupt(im,kind,rng):
 import cv2
 if kind=='clean':return im
 if kind=='blur':return cv2.GaussianBlur(im,(0,0),float(rng.uniform(.8,2.5)))
 if kind=='dark':return np.clip(im.astype(np.float32)*rng.uniform(.2,.55)+rng.normal(0,3,im.shape),0,255).astype(np.uint8)
 if kind=='glare':
  h,w=im.shape[:2]; yy,xx=np.mgrid[:h,:w]; cx,cy=rng.uniform(.2,.8)*w,rng.uniform(.15,.65)*h; s=rng.uniform(.08,.2)*min(h,w)
  m=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*s*s))[...,None]
  return np.clip(im.astype(np.float32)+rng.uniform(120,230)*m,0,255).astype(np.uint8)
 return im

def associate(rgb,ir,threshold=.45):
 pairs=sorted(((iou(a[0],b[0]),i,j) for i,a in enumerate(rgb) for j,b in enumerate(ir)),reverse=True); ur=set();ut=set(); groups=[]
 for ov,i,j in pairs:
  if ov<threshold:break
  if i not in ur and j not in ut: ur.add(i);ut.add(j);groups.append((i,j,ov))
 return groups,ur,ut

def wbf(a,b):
 # Confidence-weighted coordinates; deterministic two-box WBF.
 ca,cb=max(float(a[1]),1e-6),max(float(b[1]),1e-6)
 return ((np.asarray(a[0])*ca+np.asarray(b[0])*cb)/(ca+cb)).astype(np.float32)

def patch_contrast(im,box):
 import cv2
 h,w=im.shape[:2]; x1,y1,x2,y2=np.asarray(box).astype(int); x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
 if x2<=x1 or y2<=y1:return 0.
 gray=cv2.cvtColor(im[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
 return float(np.clip(gray.std()/64.,0,1))

def rows_for_frame(frame,rgb_im=None,ir_im=None):
 groups,ur,ut=associate(frame.rgb,frame.ir); out=[]
 for i,j,ov in groups:
  rb,rc=frame.rgb[i]; tb,tc=frame.ir[j]; box=wbf((rb,rc),(tb,tc)); cr,ci=float(rc),float(tc); matched=1.
  dist=float(np.linalg.norm(((rb[:2]+rb[2:])/2)-((tb[:2]+tb[2:])/2))/max(np.hypot(*(rgb_im.shape[1::-1])) if rgb_im is not None else 640,1))
  ar=max((rb[2]-rb[0])*(rb[3]-rb[1]),1); at=max((tb[2]-tb[0])*(tb[3]-tb[1]),1)
  asp_r=max((rb[2]-rb[0])/max(rb[3]-rb[1],1),1e-4); asp_t=max((tb[2]-tb[0])/max(tb[3]-tb[1],1),1e-4)
  pc_r=patch_contrast(rgb_im,rb) if rgb_im is not None else 0.; pc_i=patch_contrast(ir_im,tb) if ir_im is not None else 0.
  feat=[cr,ci,abs(cr-ci),matched,ov,dist,abs(np.log(ar/at)),abs(np.log(asp_r/asp_t)),pc_r,pc_i]
  out.append(dict(box=box,box_rgb=rb,box_ir=tb,score_joint=(cr+ci)/2,score_rgb=cr,score_ir=ci,feat=feat,mod='matched'))
 for modality,dets,used in [('rgb',frame.rgb,ur),('ir',frame.ir,ut)]:
  for k,(box,c) in enumerate(dets):
   if k in used:continue
   cr,ci=(float(c),0.) if modality=='rgb' else (0.,float(c)); pc=patch_contrast(rgb_im if modality=='rgb' else ir_im,box) if rgb_im is not None else 0.
   feat=[cr,ci,abs(cr-ci),0.,0.,0.,0.,0.,pc if modality=='rgb' else 0.,pc if modality=='ir' else 0.]
   out.append(dict(box=np.asarray(box),box_rgb=np.asarray(box) if modality=='rgb' else None,box_ir=np.asarray(box) if modality=='ir' else None,
    score_joint=float(c),score_rgb=cr,score_ir=ci,feat=feat,mod=modality))
 # Explicit missed-GT records keep false negatives in the risk-learning pool.
 candidate_boxes=[r['box'] for r in out]
 for gi,g in enumerate(frame.gt):
  if not any(iou(g,b)>=.5 for b in candidate_boxes):
   out.append(dict(box=g.copy(),box_rgb=None,box_ir=None,score_joint=0.,score_rgb=0.,score_ir=0.,
    feat=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],mod='missed_gt',gt_index=gi))
 return out

def action_losses(frame,rows):
 losses=[]
 for r in rows:
  ls=[]
  for act in ACTS:
   score=r['score_'+act]; box=r['box'] if act=='joint' else r['box_'+act]
   if r['mod']=='missed_gt': loss=1.
   elif box is None: loss=score**2
   else:
    u=max((iou(box,g) for g in frame.gt),default=0.)
    loss=.5*((1-score)**2+(1-u)) if u>=.5 else score**2
   ls.append(float(np.clip(loss,0,1)))
  losses.append(ls)
 return np.asarray(losses,np.float32)

class PostHocRiskHead(nn.Module):
 def __init__(self,d):
  super().__init__(); self.net=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,3),nn.Sigmoid())
 def forward(self,x):return self.net(x)

def params(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)

def make_synthetic(seed=42):
 rng=np.random.default_rng(seed); splits=[]
 for name,n,kinds in [('train',350,['clean','blur','dark']),('val',120,['clean','blur','dark']),('test',140,['glare'])]:
  fs=[]
  for ix in range(n):
   gt=[]; rgb=[];ir=[]
   for p in range(rng.integers(1,4)):
    x,y=rng.uniform(30,450),rng.uniform(30,350); w,h=rng.uniform(25,80),rng.uniform(50,140); g=np.array([x,y,x+w,y+h],np.float32);gt.append(g)
    for target,dets in [(True,rgb),(True,ir)]:
     if rng.random()<.18:continue
     jitter=rng.normal(0,5,4); b=np.maximum(g+jitter,[0,0,1,1]); conf=float(rng.beta(6,2)); dets.append((b,conf))
   for _ in range(rng.integers(0,2)):
    b=np.array([*rng.uniform(0,400,2),*rng.uniform(420,600,2)],np.float32); rgb.append((b,float(rng.beta(1.5,5))))
   kind=kinds[ix%len(kinds)];
   if kind=='dark':rgb=[(b,c*.4) for b,c in rgb]
   if kind=='blur':rgb=[(b,c*.75) for b,c in rgb];ir=[(b,c*.85) for b,c in ir]
   if kind=='glare':rgb=[(b,float(np.clip(c+rng.normal(.25,.15),0,1))) for b,c in rgb]
   fs.append(Frame(f'{name}-{ix:05d}',np.asarray(gt,np.float32).reshape(-1,4),rgb,ir,kind))
  splits.append(fs)
 return tuple(splits)

def load_llvip(root,rgb_weights,ir_weights,limits,seed):
 import cv2
 from ultralytics import YOLO
 root=Path(root); mr,mi=YOLO(rgb_weights),YOLO(ir_weights)
 for m in (mr,mi):
  m.model.eval()
  for p in m.model.parameters():p.requires_grad_(False)
 files=sorted((root/'visible'/'train').glob('*.jpg'))+sorted((root/'visible'/'test').glob('*.jpg'))
 if not files:raise FileNotFoundError(f'No LLVIP visible jpg under {root}')
 # Deterministic source-image split; variants are created only after partition.
 rng=np.random.default_rng(seed); order=rng.permutation(len(files)); ntr=min(limits[0],int(.65*len(files))); nv=min(limits[1],int(.15*len(files))); nt=min(limits[2],len(files)-ntr-nv)
 ids=[order[:ntr],order[ntr:ntr+nv],order[ntr+nv:ntr+nv+nt]]; out=[]
 for si,idxs in enumerate(ids):
  fs=[]
  for ct,index in enumerate(idxs):
   rp=files[index]; split=rp.parent.name; ip=root/'infrared'/split/rp.name; xp=root/'Annotations'/f'{rp.stem}.xml'
   if not ip.exists() or not xp.exists():continue
   rgb0=cv2.imread(str(rp));ir0=cv2.imread(str(ip));gt=read_gt(xp)
   kinds=('clean','blur','dark') if si<2 else ('glare',)
   for kind in kinds:
    rr=np.random.default_rng(seed+int(index)*7+len(kind)); rgb=corrupt(rgb0,kind,rr); ir=corrupt(ir0,kind,rr)
    def det(model,img):
     z=model.predict(img,imgsz=640,conf=.001,classes=[0],verbose=False)[0]
     if z.boxes is None:return []
     return [(b.astype(np.float32),float(c)) for b,c in zip(z.boxes.xyxy.cpu().numpy(),z.boxes.conf.cpu().numpy())]
    fs.append(Frame(rp.stem,gt,det(mr,rgb),det(mi,ir),kind,rgb,ir))
   if (ct+1)%50==0:print(f'pool {si}: {ct+1}/{len(idxs)} source images',flush=True)
  out.append(fs)
 return tuple(out)

def prepare(frames,cfg):
 X=[];Y=[]; meta=[]
 feature_names=CFG[cfg] if isinstance(cfg,str) else cfg
 for f in frames:
  rows=rows_for_frame(f,f.rgb_image,f.ir_image)
  L=action_losses(f,rows)
  for r,l in zip(rows,L):X.append([r['feat'][CFG['C'].index(name)] for name in feature_names]);Y.append(l);meta.append((f,r))
 return np.asarray(X,np.float32),np.asarray(Y,np.float32),meta

def metric_ap(records,frames):
 # Per-frame VOC-style AP50 from complete GT lists, including missed pedestrians.
 allscores=[];alltp=[];npos=sum(len(f.gt) for f in frames)
 for f in frames:
  rr=[r for ff,r in records if ff.image_id==f.image_id]
  matched=set()
  for r in sorted(rr,key=lambda q:q.get('eval_score',0),reverse=True):
   score=r.get('eval_score',0); box=r.get('eval_box')
   if box is None:continue
   vals=[(iou(box,g),j) for j,g in enumerate(f.gt) if j not in matched]
   ov,j=max(vals,default=(0,-1))
   allscores.append(score); tp=int(ov>=.5)
   if tp:matched.add(j)
   alltp.append(tp)
 if npos==0 or not allscores:return float('nan')
 order=np.argsort(-np.asarray(allscores));tp=np.asarray(alltp)[order];fp=1-tp
 rec=np.cumsum(tp)/npos;pre=np.cumsum(tp)/np.maximum(1,np.cumsum(tp+fp)); rec=np.r_[0,rec,1];pre=np.r_[0,pre,0]
 for i in range(len(pre)-2,-1,-1):pre[i]=max(pre[i],pre[i+1])
 ix=np.where(rec[1:]!=rec[:-1])[0];return float(np.sum((rec[ix+1]-rec[ix])*pre[ix+1]))

def ece(p,y,bins=15):
 if not len(y):return float('nan')
 edges=np.linspace(0,1,bins+1);ix=np.clip(np.digitize(p,edges)-1,0,bins-1)
 return float(sum(np.mean(ix==b)*abs(p[ix==b].mean()-y[ix==b].mean()) for b in range(bins) if np.any(ix==b)))

def eval_action(frames,metas,action_scores,threshold=.25,selected_actions=None):
 records=[]; tp=fp=fn=0; conf=[];lab=[]; loss=[]
 for k,(f,r) in enumerate(metas):
  a=int(selected_actions[k]) if selected_actions is not None else 0
  if r['mod']=='missed_gt':continue
  if a==3:continue
  names=('joint','rgb','ir'); name=names[a]
  box=r['box'] if a==0 else r['box_'+name]; score=float(action_scores[k,a])
  rr=dict(r);rr['eval_box']=box;rr['eval_score']=score;records.append((f,rr))
  if box is None:continue
  ov=max((iou(box,g) for g in f.gt),default=0.); conf.append(score);lab.append(float(ov>=.5))
 for f in frames:
  det=[r for ff,r in records if ff.image_id==f.image_id and r['eval_box'] is not None]
  used=set()
  for r in sorted(det,key=lambda q:q['eval_score'],reverse=True):
   vals=[(iou(r['eval_box'],g),j) for j,g in enumerate(f.gt) if j not in used];ov,j=max(vals,default=(0,-1))
   if ov>=.5:tp+=1;used.add(j)
   else:fp+=1
  fn+=len(f.gt)-len(used)
 ap=metric_ap(records,frames); nimg=max(len({f.image_id for f in frames}),1)
 return {'AP50':ap,'Miss':fn/max(fn+tp,1),'FAR':fp/max(tp+fp,1),'FPPI':fp/nimg,'ECE':ece(np.asarray(conf),np.asarray(lab)),'records':records}

def train_head(X,Y,Xv,Yv,device,epochs,batch,lr,seed,path):
 torch.manual_seed(seed);m=PostHocRiskHead(X.shape[1]).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr)
 dl=DataLoader(TensorDataset(torch.tensor(X),torch.tensor(Y)),batch_size=batch,shuffle=True); vh=[];th=[];best=1e9
 vx=torch.tensor(Xv,device=device);vy=torch.tensor(Yv,device=device)
 for ep in range(1,epochs+1):
  m.train();s=n=0
  for xb,yb in dl:
   xb=xb.to(device);yb=yb.to(device);opt.zero_grad();loss=nn.functional.mse_loss(m(xb),yb);loss.backward();opt.step();s+=loss.item()*len(xb);n+=len(xb)
  m.eval()
  with torch.no_grad():vl=nn.functional.mse_loss(m(vx),vy).item()
  tl=s/max(n,1);th.append(tl);vh.append(vl);save=''
  if vl<best:
   best=vl;torch.save({'state_dict':m.state_dict(),'input_dim':X.shape[1],'features':None,'epoch':ep,'val_loss':best},path);save=' [best checkpoint saved]'
  print(f'Epoch {ep:02d}/{epochs} train={tl:.6f} val={vl:.6f}{save}',flush=True)
 ck=torch.load(path,map_location=device,weights_only=False);m.load_state_dict(ck['state_dict']);return m,th,vh

def normalize_fit(X,Xv,Xt):
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1.;return (X-mu)/sd,(Xv-mu)/sd,(Xt-mu)/sd,mu,sd

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path);p.add_argument('--rgb-weights',default='yolov8n.pt');p.add_argument('--ir-weights',default='yolov8n.pt');p.add_argument('--synthetic',action='store_true');p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--seed',type=int,default=42);p.add_argument('--train-limit',type=int,default=800);p.add_argument('--val-limit',type=int,default=200);p.add_argument('--test-limit',type=int,default=400);p.add_argument('--output-dir',type=Path,default=Path('raven_results'));a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'device={device}; synthetic={a.synthetic}')
 if a.synthetic: tr,va,te=make_synthetic(a.seed)
 else:
  if a.data_root is None:p.error('Use --data-root or --synthetic')
  tr,va,te=load_llvip(a.data_root,a.rgb_weights,a.ir_weights,(a.train_limit,a.val_limit,a.test_limit),a.seed)
 a.output_dir.mkdir(parents=True,exist_ok=True); results={}; best_c=None
 for cfg_name,features in CFG.items():
  X,Y,mt=prepare(tr,features);Xv,Yv,mv=prepare(va,features);Xt,Yt,me=prepare(te,features);X,Xv,Xt,mu,sd=normalize_fit(X,Xv,Xt)
  m,tl,vl=train_head(X,Y,Xv,Yv,device,a.epochs,a.batch_size,a.lr,a.seed,a.output_dir/'raven_risk_head.pt' if cfg_name=='C' else a.output_dir/f'risk_{cfg_name}.pt')
  with torch.no_grad():pred=m(torch.tensor(Xt,device=device)).cpu().numpy(); pval=m(torch.tensor(Xv,device=device)).cpu().numpy()
  results[cfg_name]=(pred,tl,vl,mt,me,Yt); best_c=(pred,tl,vl,mt,me,Yt,pval)
  if cfg_name=='C':
   # Store checkpoint preprocessing and feature list for reproducible inference.
   ck=torch.load(a.output_dir/'raven_risk_head.pt',map_location='cpu',weights_only=False);ck.update({'features':features,'mean':mu.tolist(),'std':sd.tolist(),'params':params(m)});torch.save(ck,a.output_dir/'raven_risk_head.pt')
  print(f'Config {cfg_name}: input={len(features)} trainable_params={params(m)}')
 # Evaluate governance at nominal 90/95% coverage, thresholds chosen only on validation.
 pred,tl,vl,mt,me,Yt,pval=best_c; rows=[]
 for cov in (.90,.95):
  tau=float(np.quantile(np.min(pval,axis=1),cov,method='higher')); acts=np.argmin(pred,axis=1); acts[np.min(pred,axis=1)>tau]=3
  realized=[]
  for k,(f,r) in enumerate(me):
   if acts[k]<3:realized.append(Yt[k,acts[k]])
  print(f'RAVEN nominal coverage={cov:.0%} tau={tau:.5f} test_coverage={(acts<3).mean():.3f} selective_abs_loss={np.mean(realized) if realized else float("nan"):.5f}')
  q=np.exp(-pred);q=q/(q.sum(1,keepdims=True)+1e-9);ev=eval_action(te,me,q,selected_actions=acts)
  rows.append((f'RAVEN-C@{cov:.0%}',ev))
 # Baseline candidate actions, with scores; simple fusion rules and logistic calibration.
 base_rows=[]
 for name,selector in [('RGB-only',1),('Thermal-only',2),('Joint',0)]:
  acts=np.full(len(me),selector);scores=np.asarray([[r['score_joint'],r['score_rgb'],r['score_ir']] for f,r in me]);base_rows.append((name,eval_action(te,me,scores,selected_actions=acts)))
 # Scalar fusion and calibration baselines use GT-derived match labels from training only.
 def fit_scalar(kind):
  xx=np.asarray([[r['score_rgb'],r['score_ir']] for f,r in mt],np.float32)
  yy=np.asarray([min(action_losses(f,[r])[0])<.5 for f,r in mt],np.float32)
  if kind=='mean': return lambda z:(z[:,0]+z[:,1])/2
  if kind=='noisy-or': return lambda z:1-(1-z[:,0])*(1-z[:,1])
  if kind=='logistic':
   from sklearn.linear_model import LogisticRegression
   model=LogisticRegression(max_iter=500).fit(xx,yy)
   return lambda z:model.predict_proba(z)[:,1]
  # Prior lightweight two-confidence MLP calibrated against annotation-derived labels.
  xm=torch.tensor(xx,device=device);ym=torch.tensor(yy[:,None],device=device)
  net=nn.Sequential(nn.Linear(2,12),nn.ReLU(),nn.Linear(12,1),nn.Sigmoid()).to(device);opt=torch.optim.Adam(net.parameters(),lr=1e-3)
  for _ in range(40):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy(net(xm),ym);loss.backward();opt.step()
  return lambda z:net(torch.tensor(z,dtype=torch.float32,device=device)).detach().cpu().numpy().ravel()
 test_conf=np.asarray([[r['score_rgb'],r['score_ir']] for f,r in me],np.float32)
 for name,kind in [('Mean Fusion','mean'),('Noisy-OR','noisy-or'),('Logistic Calibration','logistic'),('2-input MLP','mlp')]:
  score=np.clip(fit_scalar(kind)(test_conf),0,1); scores=np.column_stack([score,score,score]);acts=np.zeros(len(me),dtype=int)
  base_rows.append((name,eval_action(te,me,scores,selected_actions=acts)))
 # Feature/model and four publication figures.
 fig,ax=plt.subplots(figsize=(7,4))
 for c,(_,h1,h2,*_) in results.items():ax.plot(h1,label=f'{c} train');ax.plot(h2,label=f'{c} val',linestyle='--')
 ax.set(xlabel='Epoch',ylabel='Absolute risk MSE');ax.legend(ncol=2);fig.tight_layout();fig.savefig(a.output_dir/'training_loss.png',dpi=220);plt.close(fig)
 # Risk-coverage from predicted minimum risk ordering.
 risks=best_c[0]; oracle=np.asarray([min(y) for y in best_c[5]]);order=np.argsort(np.min(risks,axis=1));cov=np.arange(1,len(order)+1)/len(order);sel=np.cumsum(oracle[order])/np.arange(1,len(order)+1)
 plt.figure(figsize=(6,4));plt.plot(cov,sel);plt.xlabel('Decision coverage');plt.ylabel('Mean realized absolute loss');plt.tight_layout();plt.savefig(a.output_dir/'risk_coverage_curve.png',dpi=220);plt.close()
 # Reliability diagram over proposed selected action confidence (1 - predicted risk).
 pr=np.clip(1-np.min(risks,axis=1),0,1); yy=1-oracle; bins=np.linspace(0,1,16);ix=np.clip(np.digitize(pr,bins)-1,0,14);xx=[];rr=[]
 for b in range(15):
  z=ix==b
  if z.any():xx.append(pr[z].mean());rr.append(yy[z].mean())
 plt.figure(figsize=(5,5));plt.plot([0,1],[0,1],'k--');plt.plot(xx,rr,'o-');plt.xlabel('Predicted success');plt.ylabel('Observed success');plt.tight_layout();plt.savefig(a.output_dir/'reliability_diagram.png',dpi=220);plt.close()
 plt.figure(figsize=(6,4));plt.bar(list(CFG),[np.mean(v[0]) for v in results.values()]);plt.ylabel('Mean predicted absolute risk');plt.tight_layout();plt.savefig(a.output_dir/'feature_ablation.png',dpi=220);plt.close()
 print('\n| Method | AP50 | Miss | False alarms / predictions | FPPI | ECE(15) |')
 print('|---|---:|---:|---:|---:|---:|')
 for n,e in base_rows+rows:print(f"| {n} | {e['AP50']:.4f} | {e['Miss']:.4f} | {e['FAR']:.4f} | {e['FPPI']:.4f} | {e['ECE']:.4f} |")
 print(f'Params (Config C): {params(PostHocRiskHead(10))}; device={device}; latency reporting is per-run dependent; see run_summary.json')
 (a.output_dir/'run_summary.json').write_text(json.dumps({'synthetic_smoke_test_only':a.synthetic,'split_policy':'disjoint source-image IDs; corruption variants confined to source split','features':CFG,'params_config_c':params(PostHocRiskHead(10)),'canonical_ap_note':'AP is computed per frame from complete GT lists and selected candidate boxes; synthetic metrics are not research evidence.'},indent=2))
 print(f'Artifacts: {a.output_dir.resolve()}')

if __name__=='__main__':main()
