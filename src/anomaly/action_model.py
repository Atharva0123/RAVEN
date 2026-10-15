"""Lightweight temporal pose anomaly model for RAVEN tracks.

Only the small GRU head is trained. YOLO and pose backbones stay frozen.
The model learns weakly-supervised track-window anomaly scores; its output
    needs validation-split threshold calibration before it is used for review.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn

FEATURE_DIM = 60

class TemporalPoseAnomalyHead(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=32):
        super().__init__()
        self.feature_dim=int(feature_dim); self.hidden_dim=int(hidden_dim)
        self.gru=nn.GRU(self.feature_dim,self.hidden_dim,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(self.hidden_dim),nn.Linear(self.hidden_dim,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, windows):
        if windows.shape[-1] != self.feature_dim: raise ValueError(f'expected {self.feature_dim} features')
        batch_shape=windows.shape[:-2]
        seq=windows.reshape(-1,windows.shape[-2],windows.shape[-1])
        encoded,_=self.gru(seq)
        frame_logits=self.head(encoded).squeeze(-1)
        # Video labels are weak: only some frames in an anomalous clip contain
        # the event. Pool the most anomalous quarter of frames so short events
        # are not averaged away by long stretches of ordinary activity.
        top_count=max(1,(frame_logits.shape[1]+3)//4)
        logits=frame_logits.topk(top_count,dim=1).values.mean(dim=1)
        return logits.reshape(batch_shape)
    def save(self,path,metadata=None):
        payload={'feature_dim':self.feature_dim,'hidden_dim':self.hidden_dim,
                 'review_threshold':float(getattr(self,'review_threshold',70.0)),
                 'state_dict':self.state_dict()}
        if metadata: payload['metadata']=metadata
        torch.save(payload,Path(path))
    @classmethod
    def load(cls,path,map_location='cpu'):
        data=torch.load(Path(path),map_location=map_location,weights_only=True)
        model=cls(data['feature_dim'],data['hidden_dim']);model.load_state_dict(data['state_dict'])
        model.review_threshold=float(data.get('review_threshold',70.0));model.training_metadata=data.get('metadata',{})
        model.eval();return model

def encode_track_features(row, frame_shape):
    """Build the same 60-D frozen-pose + normalized-motion vector at train/inference."""
    height,width=frame_shape[:2]; box=row.get('bbox') or (0,0,0,0)
    x1,y1,x2,y2=map(float,box); bw=max(1.,x2-x1);bh=max(1.,y2-y1)
    points=row.get('keypoints_normalized') or []
    pose=[]
    for i in range(17):
        if i < len(points) and len(points[i]) >= 3:
            pose.extend((float(points[i][0]),float(points[i][1]),float(points[i][2])))
        else: pose.extend((0.,0.,0.))
    velocity=row.get('velocity') or (0.,0.); acceleration=row.get('acceleration') or (0.,0.)
    motion=[(x1+x2)/(2*max(width,1)),(y1+y2)/(2*max(height,1)),bw/max(width,1),bh/max(height,1),
            float(velocity[0])/bh,float(velocity[1])/bh,float(acceleration[0])/bh,
            float(acceleration[1])/bh,float(row.get('rgb_confidence',row.get('confidence',0)) or 0)]
    values=np.asarray(pose+motion,dtype=np.float32)
    return np.nan_to_num(values,nan=0.,posinf=0.,neginf=0.).clip(-10,10)

class TrackAnomalyRuntime:
    """Bounded per-ID windows and throttled CPU/GPU inference."""
    def __init__(self,model,window=32,minimum_history=8,infer_every=2,
                 sample_period_seconds=.5,max_tracks=512,device='cpu'):
        if int(window) < 1 or not 1 <= int(minimum_history) <= int(window):
            raise ValueError("window/minimum_history must satisfy 1 <= minimum_history <= window")
        if int(max_tracks) < 1 or float(sample_period_seconds) < 0:
            raise ValueError("max_tracks must be positive and sample_period_seconds non-negative")
        self.model=model.to(device).eval();self.window=int(window);self.minimum_history=int(minimum_history)
        self.infer_every=max(1,int(infer_every));self.sample_period=float(sample_period_seconds)
        self.max_tracks=int(max_tracks);self.device=device;self.histories={};self.cache={};self.last_sample={};self.tick=0
    @torch.inference_mode()
    def update(self,track_id,features,timestamp=None):
        key=str(track_id); now=float(timestamp) if timestamp is not None else None
        features=np.asarray(features,dtype=np.float32).reshape(-1)
        if features.size != self.model.feature_dim:
            raise ValueError(f"features must have dimension {self.model.feature_dim}")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        previous=self.last_sample.get(key)
        if previous is not None and now is not None and now-previous<self.sample_period:
            return self.cache.get(key)
        history=self.histories.setdefault(key,deque(maxlen=self.window));history.append(features)
        if now is not None:self.last_sample[key]=now
        if len(self.histories)>self.max_tracks:
            oldest=next(iter(self.histories))
            if oldest!=key:
                self.histories.pop(oldest,None);self.cache.pop(oldest,None);self.last_sample.pop(oldest,None)
        self.tick+=1
        if len(history)<self.minimum_history:return self.cache.get(key)
        if self.tick%self.infer_every and key in self.cache:return self.cache[key]
        values=np.zeros((self.window,FEATURE_DIM),dtype=np.float32); seq=np.asarray(history,dtype=np.float32)
        values[-len(seq):]=seq
        tensor=torch.from_numpy(values).unsqueeze(0).to(self.device)
        score=float(torch.sigmoid(self.model(tensor)).detach().cpu().item()*100)
        self.cache[key]=max(0.,min(100.,score));return self.cache[key]
    def clear(self):self.histories.clear();self.cache.clear();self.last_sample.clear();self.tick=0
