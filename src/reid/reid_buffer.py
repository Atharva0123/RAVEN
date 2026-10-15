"""Optional appearance Re-ID buffer with a deterministic offline fallback."""

import time

import cv2
import numpy as np


class ReIDBuffer:
    def __init__(self, buffer_timeout=30.0, similarity_threshold=0.82,
                 feature_extractor=None, enabled=True, clock=time.time,
                 modality="rgb"):
        self.buffer_timeout = float(buffer_timeout)
        self.similarity_threshold = float(similarity_threshold)
        self.feature_extractor = feature_extractor
        self.enabled = bool(enabled)
        self.clock = clock
        self.modality = str(modality).lower()
        self.buffer = {}

    def extract_feature(self, crop):
        if crop is None or crop.size == 0:
            return np.zeros(32, dtype=np.float32)
        if self.feature_extractor is not None:
            feature = np.asarray(self.feature_extractor(crop), dtype=np.float32).reshape(-1)
        else:
            # Color and gradient descriptors are deterministic and dependency-light.
            image = crop
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, (32, 64), interpolation=cv2.INTER_AREA)
            # Trim the box edges, which often contain more background than
            # clothing, then retain coarse top-to-bottom appearance layout.
            image = image[:, 2:30]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            if self.modality == "thermal":
                # False-color palettes vary across thermal cameras and do not
                # encode clothing color. Match silhouette/gradient only.
                vertical = [float(gray[start:end].mean()) for start, end in
                            ((0, 16), (16, 32), (32, 48), (48, 64))]
                feature = np.asarray(vertical, dtype=np.float32)
                grad, _ = np.histogram(np.arctan2(gy, gx), bins=8, range=(-np.pi, np.pi),
                                       weights=np.hypot(gx, gy))
                feature = np.concatenate([feature, grad.astype(np.float32)])
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                parts = []
                for start, end in ((0, 16), (16, 32), (32, 48), (48, 64)):
                    hist = cv2.calcHist([hsv[start:end]], [0, 1], None,
                                        [8, 4], [0, 180, 0, 256]).flatten()
                    hist_norm = float(np.linalg.norm(hist))
                    if hist_norm:
                        hist /= hist_norm
                    grad, _ = np.histogram(np.arctan2(gy[start:end], gx[start:end]),
                                           bins=8, range=(-np.pi, np.pi),
                                           weights=np.hypot(gx[start:end], gy[start:end]))
                    grad = grad.astype(np.float32)
                    grad_norm = float(np.linalg.norm(grad))
                    if grad_norm:
                        grad /= grad_norm
                    parts.extend((hist, 0.5 * grad))
                feature = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm else feature

    def update_and_match(self, current_tracks, frame, *, timestamp=None):
        # Video processing can be much slower than source time on CPU. Re-ID
        # expiry and matching must use media time when it is available, or
        # identical clothing can be matched across unrelated moments.
        now = float(timestamp) if timestamp is not None else self.clock()
        self.buffer = {tid: data for tid, data in self.buffer.items()
                       if now - data["last_seen"] <= self.buffer_timeout}
        active_ids = {track.get("track_id") for track in current_tracks}
        # Never let a recovered appearance claim an ID already used by another
        # observation in this frame. Raw tracker IDs and canonical Re-ID IDs
        # can differ after a recovery, so reserve both as matching proceeds.
        claimed_ids = {str(track_id) for track_id in active_ids if track_id is not None}
        recovered_this_frame = set()
        height, width = frame.shape[:2]
        updated = []
        for track in current_tracks:
            bbox = [float(v) for v in track["bbox"]]
            x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2, y2 = min(width, int(bbox[2])), min(height, int(bbox[3]))
            crop = frame[y1:y2, x1:x2]
            feature = self.extract_feature(crop)
            tracker_id = str(track["track_id"])
            tid = tracker_id
            recovered = False
            similarity = None
            old = self.buffer.get(tid)
            if self.enabled and old is None and feature.size and np.linalg.norm(feature):
                candidates = []
                for old_id, data in self.buffer.items():
                    if (old_id in active_ids or old_id in claimed_ids or
                            old_id in recovered_this_frame or data["embedding"].shape != feature.shape):
                        continue
                    if data.get("bbox"):
                        old_w = max(1.0, data["bbox"][2] - data["bbox"][0])
                        old_h = max(1.0, data["bbox"][3] - data["bbox"][1])
                        new_w = max(1.0, bbox[2] - bbox[0])
                        new_h = max(1.0, bbox[3] - bbox[1])
                        ratio = (new_w / new_h) / (old_w / old_h)
                        if not 0.4 <= ratio <= 2.5:
                            continue
                    gap = now - float(data["last_seen"])
                    if gap < 0 or gap > self.buffer_timeout:
                        continue
                    old_center = ((data["bbox"][0] + data["bbox"][2]) / 2,
                                  (data["bbox"][1] + data["bbox"][3]) / 2)
                    new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    frame_diagonal = float(np.hypot(width, height))
                    max_displacement = frame_diagonal * min(.4, .03 + .08 * gap)
                    if np.hypot(new_center[0] - old_center[0],
                                new_center[1] - old_center[1]) > max_displacement:
                        continue
                    candidate_similarity = float(np.dot(feature, data["embedding"]))
                    if candidate_similarity >= self.similarity_threshold:
                        candidates.append((candidate_similarity, old_id))
                if candidates:
                    candidates.sort(reverse=True)
                    best = candidates[0]
                    runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
                    # Avoid silently merging people when visual similarity is ambiguous.
                    if best[0] - runner_up >= .05:
                        similarity, tid = best
                        recovered = True
                        recovered_this_frame.add(tid)
            claimed_ids.add(str(tid))
            previous = self.buffer.get(tid)
            if previous and previous["embedding"].shape == feature.shape:
                if similarity is None and feature.size and np.linalg.norm(feature):
                    similarity = float(np.dot(feature, previous["embedding"]))
                # Keep a stable track prototype while allowing gradual appearance
                # changes (pose, exposure, partial occlusion) to update it.
                feature = 0.2 * feature + 0.8 * previous["embedding"]
                norm = float(np.linalg.norm(feature))
                if norm:
                    feature /= norm
            self.buffer[tid] = {"embedding": feature, "last_seen": now, "bbox": bbox}
            result = track.copy()
            result.update(
                track_id=tid,
                tracker_id=tracker_id,
                reid_canonical_track_id=tid,
                recovered=recovered,
                reid_status=("disabled" if not self.enabled else
                             "appearance_match" if recovered else
                             "tracker_continuity" if previous else "new_track"),
                reid_similarity=round(float(similarity), 4) if similarity is not None else None,
                reid_method=("custom_feature_extractor" if self.feature_extractor is not None
                             else "color_gradient_descriptor"),
                reid_embedding=feature.astype(float).tolist(),
                identity_source="appearance_reid" if recovered else "tracker",
            )
            updated.append(result)
        return updated
