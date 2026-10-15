"""Small dependency-free metrics over caller-supplied labels and predictions."""
import math
import numpy as np


def _iou(a, b):
    iw = max(0.0, min(a[2],b[2])-max(a[0],b[0]))
    ih = max(0.0, min(a[3],b[3])-max(a[1],b[1]))
    inter = iw*ih
    union = max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1])+max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])-inter
    return inter/union if union else 0.0


def calculate_map(ground_truth, predictions, iou_threshold=.5):
    """Compute interpolated AP per class and mean AP over present GT classes.

    Inputs map frame IDs to lists of {bbox, class_id, ...}; predictions also
    require confidence/conf. Empty ground truth returns 0 rather than a claim.
    """
    classes = sorted({int(item.get("class_id",0)) for rows in ground_truth.values() for item in rows})
    if not classes:
        return 0.0
    average_precisions = []
    for class_id in classes:
        gt_by_frame = {frame: [g for g in rows if int(g.get("class_id",0)) == class_id]
                       for frame, rows in ground_truth.items()}
        total_gt = sum(map(len, gt_by_frame.values()))
        detections = [(float(d.get("confidence",d.get("conf",0))), frame, d)
                      for frame, rows in predictions.items() for d in rows
                      if int(d.get("class_id",0)) == class_id]
        detections.sort(key=lambda row: row[0], reverse=True)
        matched = {frame: set() for frame in gt_by_frame}
        tp, fp = [], []
        for _, frame, detection in detections:
            candidates = gt_by_frame.get(frame, [])
            overlaps = [(_iou(detection["bbox"], gt["bbox"]), i) for i,gt in enumerate(candidates) if i not in matched.get(frame,set())]
            best = max(overlaps, default=(0.0,-1))
            is_match = best[0] >= iou_threshold
            tp.append(1 if is_match else 0); fp.append(0 if is_match else 1)
            if is_match: matched.setdefault(frame,set()).add(best[1])
        if not tp:
            average_precisions.append(0.0); continue
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall, precision = tp_cum / total_gt, tp_cum / np.maximum(tp_cum+fp_cum,1)
        recall = np.concatenate(([0.0],recall,[1.0])); precision=np.concatenate(([0.0],precision,[0.0]))
        for i in range(len(precision)-2,-1,-1): precision[i]=max(precision[i],precision[i+1])
        points=np.where(recall[1:] != recall[:-1])[0]
        average_precisions.append(float(np.sum((recall[points+1]-recall[points])*precision[points+1])))
    return float(np.mean(average_precisions))


def calculate_idf1(ground_truth_tracks, predicted_tracks):
    """Compute IDF1 from per-frame {track_id, bbox} observations."""
    frames = set(ground_truth_tracks) | set(predicted_tracks)
    gt_counts, pred_counts, pair_counts = {}, {}, {}
    total_gt = total_pred = 0
    for frame in frames:
        gt, pred = ground_truth_tracks.get(frame,[]), predicted_tracks.get(frame,[])
        total_gt += len(gt); total_pred += len(pred)
        for item in gt: gt_counts[item["track_id"]]=gt_counts.get(item["track_id"],0)+1
        for item in pred: pred_counts[item["track_id"]]=pred_counts.get(item["track_id"],0)+1
        candidates=[]
        for g in gt:
            for p in pred:
                overlap=_iou(g["bbox"],p["bbox"])
                if overlap >= .5: candidates.append((overlap,g["track_id"],p["track_id"]))
        used_g,used_p=set(),set()
        for _,g,p in sorted(candidates,reverse=True):
            if g not in used_g and p not in used_p:
                pair_counts[g,p]=pair_counts.get((g,p),0)+1;used_g.add(g);used_p.add(p)
    gt_ids,pred_ids=list(gt_counts),list(pred_counts)
    matrix=np.zeros((len(gt_ids),len(pred_ids)))
    for i,g in enumerate(gt_ids):
        for j,p in enumerate(pred_ids): matrix[i,j]=pair_counts.get((g,p),0)
    matched=0
    if matrix.size:
        try:
            from scipy.optimize import linear_sum_assignment
            rows,cols=linear_sum_assignment(-matrix);matched=int(matrix[rows,cols].sum())
        except ImportError:
            remaining=matrix.copy()
            while remaining.size and remaining.max()>0:
                i,j=np.unravel_index(np.argmax(remaining),remaining.shape);matched+=int(remaining[i,j]);remaining[i,:]=0;remaining[:,j]=0
    denom=total_gt+total_pred
    return 2*matched/denom if denom else 0.0


def calculate_rank1(query_embeddings, gallery_embeddings, query_ids=None, gallery_ids=None):
    """Return top-1 accuracy; IDs are required to define correctness."""
    queries=np.asarray(query_embeddings,dtype=float);gallery=np.asarray(gallery_embeddings,dtype=float)
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embeddings must be compatible 2D arrays")
    if query_ids is None or gallery_ids is None:
        raise ValueError("query_ids and gallery_ids are required to calculate rank-1 accuracy")
    if len(queries)!=len(query_ids) or len(gallery)!=len(gallery_ids):
        raise ValueError("identity labels must match embedding counts")
    if not len(queries) or not len(gallery): return 0.0
    qnorm=queries/np.maximum(np.linalg.norm(queries,axis=1,keepdims=True),1e-12)
    gnorm=gallery/np.maximum(np.linalg.norm(gallery,axis=1,keepdims=True),1e-12)
    best=np.argmax(qnorm@gnorm.T,axis=1)
    return float(np.mean([query_ids[i]==gallery_ids[j] for i,j in enumerate(best)]))


def calculate_false_alarm_rate(total_hours, false_alarms):
    if total_hours <= 0: raise ValueError("total_hours must be positive")
    if false_alarms < 0: raise ValueError("false_alarms cannot be negative")
    return false_alarms / total_hours
