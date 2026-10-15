"""Trainable cross-modal consensus model for RGB and thermal track features.

The network returns an anomaly logit. Identity tier is a learned input to the
network and a bounded learned alert-threshold adjustment; it is never inferred
from appearance. Weapon context is represented as an input embedding, so it is
learned jointly with posture and modality evidence rather than bypassing fusion.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


IDENTITY_TIERS = ("authorized", "unknown", "flagged")
DEFAULT_FEATURE_DIM = 128
POSTURE_FEATURE_DIM = 17 * 3


class LearnedConsensusFusion(nn.Module):
    """Tiny cross-attention transformer over RGB, IR, and threat tokens.

    Inputs have shape ``(batch, feature_dim)`` (or ``(feature_dim,)`` for one
    track). RGB/IR features may concatenate normalized box (4), Re-ID vector,
    17 normalized keypoints (x, y, confidence), posture confidence and image
    quality. ``identity_tier`` accepts integer IDs or one-hot vectors.
    """

    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM, hidden_dim: int = 96,
                 num_heads: int = 4, identity_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.identity_embedding = nn.Embedding(len(IDENTITY_TIERS), identity_dim)
        self.rgb_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.thermal_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.threat_projection = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU())
        self.identity_projection = nn.Linear(identity_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                      batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fusion = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2,
                                                  dropout=dropout, batch_first=True,
                                                  norm_first=True, activation="gelu")
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        # Learned, bounded shift to the decision boundary by identity tier.
        self.tier_threshold_shift = nn.Embedding(len(IDENTITY_TIERS), 1)
        nn.init.zeros_(self.tier_threshold_shift.weight)

    @staticmethod
    def _tier_ids(identity_tier, batch: int, device) -> torch.Tensor:
        if isinstance(identity_tier, str):
            ids = torch.tensor(IDENTITY_TIERS.index(identity_tier.lower()), device=device)
        elif isinstance(identity_tier, (list, tuple)) and identity_tier and isinstance(identity_tier[0], str):
            ids = torch.tensor([IDENTITY_TIERS.index(value.lower()) for value in identity_tier], device=device)
        else:
            ids = torch.as_tensor(identity_tier, device=device)
        if ids.ndim and ids.shape[-1] == len(IDENTITY_TIERS):
            ids = ids.argmax(dim=-1)
        ids = ids.long().reshape(-1)
        if ids.numel() == 1:
            ids = ids.expand(batch)
        if ids.numel() != batch or torch.any((ids < 0) | (ids >= len(IDENTITY_TIERS))):
            raise ValueError("identity_tier must identify authorized, unknown, or flagged per sample")
        return ids

    def forward(self, rgb_features: torch.Tensor, thermal_features: torch.Tensor,
                identity_tier: torch.Tensor | int | str = 1,
                weapon_context: torch.Tensor | None = None,
                return_details: bool = False):
        squeeze = rgb_features.ndim == 1
        if squeeze:
            rgb_features, thermal_features = rgb_features.unsqueeze(0), thermal_features.unsqueeze(0)
        if rgb_features.shape != thermal_features.shape or rgb_features.shape[-1] != self.feature_dim:
            raise ValueError(f"RGB and thermal features must have matching (..., {self.feature_dim}) shapes")
        batch, device = rgb_features.shape[0], rgb_features.device
        tiers = self._tier_ids(identity_tier, batch, device)
        if weapon_context is None:
            weapon_context = torch.zeros(batch, 2, device=device, dtype=rgb_features.dtype)
        else:
            weapon_context = torch.as_tensor(weapon_context, device=device, dtype=rgb_features.dtype)
            if weapon_context.ndim == 1:
                weapon_context = weapon_context.unsqueeze(0)
            if weapon_context.shape == (batch,):
                weapon_context = torch.stack((weapon_context, torch.ones_like(weapon_context)), dim=-1)
            if weapon_context.shape != (batch, 2):
                raise ValueError("weapon_context must have shape (batch, 2): confidence and presence")
        rgb = self.rgb_projection(rgb_features)
        thermal = self.thermal_projection(thermal_features)
        threat = self.threat_projection(weapon_context)
        identity = self.identity_projection(self.identity_embedding(tiers))
        # Each modality queries the other; image quality/confidence in feature
        # vectors lets attention learn which stream to trust in poor lighting.
        modalities = torch.stack((rgb, thermal), dim=1)
        attended, weights = self.cross_attention(modalities, modalities, modalities,
                                                  need_weights=return_details)
        tokens = torch.stack((self.norm1(modalities + attended).mean(dim=1),
                              threat, identity), dim=1)
        encoded = self.fusion(tokens).mean(dim=1)
        logits = self.head(encoded).squeeze(-1)
        threshold = 0.5 + 0.2 * torch.tanh(self.tier_threshold_shift(tiers).squeeze(-1))
        if squeeze:
            logits, threshold = logits.squeeze(0), threshold.squeeze(0)
        if not return_details:
            return logits
        probability = torch.sigmoid(logits)
        details = {"probability": probability, "alert_threshold": threshold,
                   "alert": probability >= threshold, "identity_tier_id": tiers,
                   "weapon_context": weapon_context, "cross_attention": weights}
        return logits, details

    @classmethod
    def load(cls, path, map_location="cpu"):
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
        model = cls(feature_dim=checkpoint.get("feature_dim", DEFAULT_FEATURE_DIM),
                    hidden_dim=checkpoint.get("hidden_dim", 96),
                    num_heads=checkpoint.get("num_heads", 4),
                    identity_dim=checkpoint.get("identity_dim", 8))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim,
                    "num_heads": self.cross_attention.num_heads,
                    "identity_dim": self.identity_embedding.embedding_dim,
                    "identity_tiers": IDENTITY_TIERS, "state_dict": self.state_dict()}, path)
