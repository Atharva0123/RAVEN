"""Threat-aware trajectory smoothing and Ramer-Douglas-Peucker pruning."""
import math

def _dist(p, a, b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    if dx == dy == 0: return math.dist(p,a)
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.dist(p,(a[0]+t*dx,a[1]+t*dy))

def _rdp(points, eps):
    if len(points)<=2:return list(points)
    ds=[_dist(p,points[0],points[-1]) for p in points[1:-1]]
    if not ds:return [points[0],points[-1]]
    i=max(range(1,len(points)-1),key=lambda j:ds[j-1])
    if ds[i-1]>eps:return _rdp(points[:i+1],eps)[:-1]+_rdp(points[i:],eps)
    return [points[0],points[-1]]

class AdaptiveTrajectorySmoother:
    """Reduce low-score paths aggressively; preserve high-score detail."""
    def __init__(self, low_threshold=35, high_threshold=70, low_epsilon=10., high_epsilon=1.5,
                 low_max_points=12, high_max_points=20, low_window_seconds=8.,
                 high_window_seconds=30., smoothing_window=3):
        if not 0<=low_threshold<high_threshold<=100: raise ValueError('invalid score thresholds')
        self.low_threshold,self.high_threshold=low_threshold,high_threshold
        self.low_epsilon,self.high_epsilon=low_epsilon,high_epsilon
        self.low_max_points,self.high_max_points=low_max_points,high_max_points
        self.low_window_seconds,self.high_window_seconds=low_window_seconds,high_window_seconds
        self.smoothing_window=max(1,int(smoothing_window))
    @staticmethod
    def _xy(item): return tuple(map(float,item['center'] if isinstance(item,dict) else item[:2]))
    def simplify(self, trajectory, threat_score):
        if not trajectory:return []
        score=max(0.,min(100.,float(threat_score))); low=score<=self.low_threshold; high=score>=self.high_threshold
        ratio=max(0.,min(1.,(score-self.low_threshold)/(self.high_threshold-self.low_threshold)))
        eps=self.low_epsilon if low else self.high_epsilon if high else self.low_epsilon+(self.high_epsilon-self.low_epsilon)*ratio
        cap=self.low_max_points if low else self.high_max_points
        window=self.low_window_seconds if low else self.high_window_seconds
        data=list(trajectory); ts=[float(x.get('timestamp',0)) for x in data if isinstance(x,dict)]
        if ts and ts[-1]>0:
            cutoff=ts[-1]-window; data=[x for x in data if not isinstance(x,dict) or float(x.get('timestamp',0))>=cutoff]
        if not data:return []
        coords=[self._xy(x) for x in data]; smooth=list(coords)
        if low and self.smoothing_window>1:
            radius=self.smoothing_window//2; smooth=[]
            for i,p in enumerate(coords):
                group=coords[max(0,i-radius):min(len(coords),i+radius+1)]
                smooth.append((sum(q[0] for q in group)/len(group),sum(q[1] for q in group)/len(group)))
            smooth[0],smooth[-1]=coords[0],coords[-1]
        selected=_rdp(smooth,eps); indices=[]; cursor=0
        for p in selected:
            j=min(range(cursor,len(coords)),key=lambda k:math.dist(coords[k],p)); indices.append(j); cursor=min(j+1,len(coords)-1)
        indices=sorted(set(indices))
        # Keep recent high-score observations at a minimum density even when
        # the path is geometrically straight; those timestamps matter to review.
        if high and len(data) > 2:
            recent_count=min(len(data), max(12, cap // 2))
            recent_start=len(data)-recent_count
            indices=sorted(set(indices).union(range(recent_start, len(data))))
        if len(indices)>cap:
            stride=(len(indices)-1)/max(1,cap-1); indices=sorted(set(indices[round(i*stride)] for i in range(cap)))
        return [data[i] for i in indices] if isinstance(data[0],dict) else [coords[i] for i in indices]
    def apply_to_results(self, results):
        out=[]
        for row in results:
            item=dict(row)
            score=max(float(item.get('anomaly_score',0) or 0),
                      float(item.get('risk_score',0) or 0))
            item['trajectory']=self.simplify(item.get('trajectory',[]),score); out.append(item)
        return out
