"""Temporal RGB/thermal cross-modal event-ranking architecture prototype.

The architecture uses bidirectional multi-head cross-attention and GRU
encoders over paired feature sequences. The bundled ``best_model.pth`` is a
synthetic smoke-test baseline for validating model loading, tensor flow,
attention, and latency. It is not a trained or validated threat detector, and
its scores must not be used for decisions. Real-world event ranking requires
properly labeled paired data and held-out evaluation.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import torch
from torch import nn


class TemporalCrossModalFusion(nn.Module):
    """Encode motion in each modality and exchange information as a prototype.

    Input: ``rgb_seq`` and ``thermal_seq`` shaped ``(B, T, D)``.
    Output: one logit per sequence ``(B,)``; sigmoid is an uncalibrated model
    score, never a threat probability without task-appropriate validation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or hidden_dim % num_heads:
            raise ValueError("feature_dim/hidden_dim must be positive and hidden_dim divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        # Motion signal is a frame-to-frame feature difference. Prepending a
        # zero keeps T unchanged; concatenation gives (B,T,2D).
        self.rgb_motion_encoder = nn.GRU(2 * feature_dim, hidden_dim, batch_first=True)
        self.thermal_motion_encoder = nn.GRU(2 * feature_dim, hidden_dim, batch_first=True)

        # Bidirectional temporal cross-attention: RGB queries IR keys/values,
        # then IR queries RGB keys/values. Each output is (B,T,H).
        self.rgb_queries_thermal = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.thermal_queries_rgb = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.rgb_norm = nn.LayerNorm(hidden_dim)
        self.thermal_norm = nn.LayerNorm(hidden_dim)

        # Pool over time, concatenate both updated streams, and classify.
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _with_motion(sequence: torch.Tensor) -> torch.Tensor:
        # sequence (B,T,D); delta (B,T,D), with frame zero's delta set to 0.
        delta = torch.zeros_like(sequence)
        if sequence.shape[1] > 1:
            delta[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        return torch.cat((sequence, delta), dim=-1)  # (B,T,2D)

    def forward(self, rgb_seq: torch.Tensor, thermal_seq: torch.Tensor,
                return_attention: bool = False):
        # Accept a single sequence (T,D) as a convenience; preserve batched API.
        squeeze_batch = rgb_seq.ndim == 2
        if squeeze_batch:
            rgb_seq = rgb_seq.unsqueeze(0)           # (1,T,D)
            thermal_seq = thermal_seq.unsqueeze(0)   # (1,T,D)
        if rgb_seq.ndim != 3 or thermal_seq.ndim != 3:
            raise ValueError("both inputs must have shape (B,T,D) or (T,D)")
        if rgb_seq.shape != thermal_seq.shape:
            raise ValueError("RGB and thermal inputs must have identical (B,T,D) shapes")
        if rgb_seq.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {rgb_seq.shape[-1]}")
        if rgb_seq.shape[1] < 1:
            raise ValueError("sequence length must be at least one")

        rgb_encoded, _ = self.rgb_motion_encoder(self._with_motion(rgb_seq))
        # rgb_encoded: (B,T,H)
        thermal_encoded, _ = self.thermal_motion_encoder(self._with_motion(thermal_seq))
        # thermal_encoded: (B,T,H)

        rgb_cross, rgb_to_thermal_weights = self.rgb_queries_thermal(
            query=rgb_encoded, key=thermal_encoded, value=thermal_encoded,
            need_weights=return_attention, average_attn_weights=False)
        thermal_cross, thermal_to_rgb_weights = self.thermal_queries_rgb(
            query=thermal_encoded, key=rgb_encoded, value=rgb_encoded,
            need_weights=return_attention, average_attn_weights=False)
        rgb_fused = self.rgb_norm(rgb_encoded + rgb_cross)       # (B,T,H)
        thermal_fused = self.thermal_norm(thermal_encoded + thermal_cross)  # (B,T,H)

        rgb_pooled = rgb_fused.mean(dim=1)             # (B,H), temporal mean pool
        thermal_pooled = thermal_fused.mean(dim=1)     # (B,H)
        logits = self.classifier(torch.cat((rgb_pooled, thermal_pooled), dim=-1)).squeeze(-1)
        # logits (B,); sigmoid(logits) is the learned 0-to-1 anomaly/risk score.
        if squeeze_batch:
            logits = logits.squeeze(0)
        if return_attention:
            return logits, {"rgb_queries_thermal": rgb_to_thermal_weights,
                            "thermal_queries_rgb": thermal_to_rgb_weights}
        return logits

    def predict_score(self, rgb_seq: torch.Tensor, thermal_seq: torch.Tensor) -> torch.Tensor:
        """Return sigmoid scores in [0,1] without gradients."""
        self.eval()
        with torch.inference_mode():
            return torch.sigmoid(self(rgb_seq, thermal_seq))

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Provenance defaults to unverified. Callers must explicitly provide
        # real-data validation metadata before a deployment may score with it.
        metadata.setdefault("validated_on_real_data", False)
        metadata.setdefault("synthetic", False)
        torch.save({"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim,
                    "state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            feature_dim = int(payload["feature_dim"])
            hidden_dim = int(payload["hidden_dim"])
            metadata = payload.get("metadata", {})
        else:
            # Also accept a plain model.state_dict(), as used by small research
            # and synthetic smoke-training scripts. Dimensions are recoverable
            # from the first GRU input projection: (3H, 2D).
            state_dict = payload
            projection = state_dict.get("rgb_motion_encoder.weight_ih_l0")
            if projection is None or projection.ndim != 2:
                raise ValueError("unrecognized TemporalCrossModalFusion checkpoint format")
            hidden_dim = int(projection.shape[0] // 3)
            feature_dim = int(projection.shape[1] // 2)
            metadata = {}
            sidecar = path.with_suffix(".metadata.json")
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        model = cls(feature_dim, hidden_dim)
        model.load_state_dict(state_dict)
        model.eval()
        model.training_metadata = metadata
        return model


class TemporalCrossModalRuntime:
    """Bounded per-track sequence builder for streaming model inference."""

    def __init__(self, model: TemporalCrossModalFusion, *, window: int = 32,
                 minimum_history: int = 8, max_tracks: int = 512, device="cpu"):
        if window < 1 or not 1 <= minimum_history <= window:
            raise ValueError("window/minimum_history are invalid")
        self.model = model.to(device).eval()
        self.window = int(window)
        self.minimum_history = int(minimum_history)
        self.max_tracks = int(max_tracks)
        self.device = torch.device(device)
        self.history = {}

    @torch.inference_mode()
    def update(self, track_id, rgb_features, thermal_features):
        key = str(track_id)
        rgb = torch.as_tensor(rgb_features, dtype=torch.float32).reshape(-1).detach().cpu().numpy()
        thermal = torch.as_tensor(thermal_features, dtype=torch.float32).reshape(-1).detach().cpu().numpy()
        if rgb.size != self.model.feature_dim or thermal.size != self.model.feature_dim:
            raise ValueError(f"stream features must each have dimension {self.model.feature_dim}")
        if key not in self.history and len(self.history) >= self.max_tracks:
            self.history.pop(next(iter(self.history)))
        if key not in self.history:
            self.history[key] = (deque(maxlen=self.window), deque(maxlen=self.window))
        rgb_history, thermal_history = self.history[key]
        rgb_history.append(rgb)
        thermal_history.append(thermal)
        if len(rgb_history) < self.minimum_history:
            return None
        # Repeat the oldest real observation to form a fixed window. No synthetic
        # features are created, and repeated padding has zero initial motion.
        rgb_array = np.asarray(rgb_history, dtype=np.float32)
        thermal_array = np.asarray(thermal_history, dtype=np.float32)
        padding = self.window - len(rgb_array)
        if padding:
            rgb_array = np.concatenate((np.repeat(rgb_array[:1], padding, axis=0), rgb_array))
            thermal_array = np.concatenate((np.repeat(thermal_array[:1], padding, axis=0), thermal_array))
        rgb_tensor = torch.from_numpy(rgb_array).unsqueeze(0).to(self.device)
        thermal_tensor = torch.from_numpy(thermal_array).unsqueeze(0).to(self.device)
        return float(self.model.predict_score(rgb_tensor, thermal_tensor).detach().cpu().item())

    def clear(self):
        self.history.clear()
