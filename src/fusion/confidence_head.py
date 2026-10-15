"""RAVEN confidence calibration and post-hoc S-DEC risk-governance heads.

The active S-DEC Config B is an 867-parameter ``8 -> 32 -> 16 -> 3``
network. Its canonical input order is ``[c_rgb, c_ir, abs_conf_diff,
box_iou, center_dist, scale_ratio, aspect_diff, matched_flag]``; the loader
adapts this to the trained checkpoint order. The legacy two-input scalar
``ConfidenceFusionHead`` remains available for backward compatibility.
"""
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConfidenceFusionHead(nn.Module):
    """Map standardized RGB/thermal confidences to a consensus probability.

    Standardization is part of the model: the training mean and scale are saved
    in the checkpoint and applied identically by forward() at inference.
    """

    FEATURE_COUNT = 2

    def __init__(self, input_dim=2, hidden_dim=12, use_mlp=True,
                 feature_mean=None, feature_std=None):
        super().__init__()
        if input_dim != self.FEATURE_COUNT:
            raise ValueError("the confidence head requires RGB and thermal confidence")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_mlp = bool(use_mlp)
        mean = torch.zeros(input_dim) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(input_dim) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (input_dim,) or std.shape != (input_dim,):
            raise ValueError("feature mean and std must each have shape [2]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("feature mean/std must be finite and std must be positive")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        # Return logits internally so BCEWithLogitsLoss remains stable.
        self.network = (nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
                        if self.use_mlp else nn.Linear(input_dim, 1))

    def forward_logits(self, features):
        if features.shape[-1] != self.FEATURE_COUNT:
            raise ValueError("features must have shape [..., 2]")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        features = features.clamp(0, 1)
        standardized = (features - self.feature_mean) / self.feature_std
        return self.network(standardized).squeeze(-1)

    def forward(self, features):
        return torch.sigmoid(self.forward_logits(features))

    @torch.inference_mode()
    def predict_score(self, rgb_confidence, thermal_confidence):
        device = next(self.parameters()).device
        features = torch.tensor([[rgb_confidence, thermal_confidence]],
                                dtype=self.feature_mean.dtype, device=device)
        return float(self(features).item())

    def fit(self, features, labels, epochs=300, learning_rate=1.0,
            weight_decay=1e-3, optimizer_name="lbfgs"):
        if features.ndim != 2 or features.shape[1] != self.FEATURE_COUNT:
            raise ValueError("features must be [N, 2]")
        features = features.to(dtype=torch.float32)
        labels = labels.to(device=features.device, dtype=torch.float32).reshape(-1)
        if len(labels) != len(features) or not torch.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
            raise ValueError("labels must match features and be finite in [0,1]")
        if not torch.isfinite(features).all() or ((features < 0) | (features > 1)).any():
            raise ValueError("features must be finite and in [0,1]")
        if epochs < 1 or weight_decay < 0:
            raise ValueError("epochs must be positive and weight_decay non-negative")
        # Fit population statistics on training data only; guard constant columns.
        with torch.no_grad():
            self.feature_mean.copy_(features.mean(dim=0).to(self.feature_mean.device))
            scale = features.std(dim=0, unbiased=False)
            scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            self.feature_std.copy_(scale.to(self.feature_std.device))
        features = features.to(self.feature_mean.device)
        labels = labels.to(self.feature_mean.device)
        if optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate,
                                          max_iter=20, line_search_fn="strong_wolfe")
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        else:
            raise ValueError("optimizer_name must be 'lbfgs' or 'adamw'")

        # Exclude biases from the L2 penalty, matching conventional weight decay.
        regularized = [p for name, p in self.named_parameters() if name.endswith("weight")]
        def objective():
            logits = self.forward_logits(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if optimizer_name == "lbfgs":
                loss = loss + weight_decay * sum(p.square().sum() for p in regularized)
            return loss

        losses = []
        for _ in range(epochs):
            if optimizer_name == "lbfgs":
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective()
                    loss.backward()
                    return loss
                optimizer.step(closure)
                # Report the objective at the final accepted parameters.
                loss = objective().detach()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = objective()
                loss.backward()
                optimizer.step()
                loss = loss.detach()
            losses.append(float(loss.cpu()))
        return losses

    def save(self, path):
        torch.save({"model_type": "confidence_fusion_mlp", "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim, "use_mlp": self.use_mlp,
                    "feature_mean": self.feature_mean.detach().cpu(),
                    "feature_std": self.feature_std.detach().cpu(),
                    "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path, map_location="cpu"):
        data = torch.load(Path(path), map_location=map_location, weights_only=True)
        # Accept both the repo format and the RAVEN.ipynb paper checkpoint format.
        state = data.get("state_dict", data)
        if "feature_mean" in data:
            mean, std = data["feature_mean"], data["feature_std"]
        elif "mean" in data and "std" in data:
            mean, std = data["mean"], data["std"]
        else:
            mean, std = torch.zeros(2), torch.ones(2)
        hidden = int(data.get("hidden_dim", 12))
        model = cls(input_dim=int(data.get("input_dim", 2)), hidden_dim=hidden,
                    use_mlp=data.get("use_mlp", data.get("model_type", "mlp_conf2") != "linear"),
                    feature_mean=mean, feature_std=std)
        # Notebook checkpoints store Sequential layers without the trailing sigmoid.
        if "network.0.weight" not in state and "0.weight" in state:
            state = {f"network.{k}": v for k, v in state.items()}
        state = dict(state)
        state.setdefault("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        state.setdefault("feature_std", torch.as_tensor(std, dtype=torch.float32))
        model.load_state_dict(state)
        model.eval()
        return model


class PostHocRiskHead(nn.Module):
    """S-DEC Config B head for post-hoc absolute-risk arbitration.

    The network is ``8 -> 32 -> 16 -> 3`` with ReLU hidden layers and a
    Sigmoid output, for exactly 867 trainable parameters (816 weights and
    51 biases). It predicts risks for joint, RGB-only, and IR-only actions.
    Public features use ``[c_rgb, c_ir, abs_conf_diff, box_iou,
    center_dist, scale_ratio, aspect_diff, matched_flag]``; the saved
    checkpoint permutation is ``[c_rgb, c_ir, abs_conf_diff, matched_flag,
    box_iou, center_dist, scale_ratio, aspect_diff]``. This remains separate
    from the legacy two-input :class:`ConfidenceFusionHead` API.
    """

    # Canonical public schema. The supplied Config B checkpoint was trained
    # with matched_flag at index 3; forward() adapts to that saved order.
    FEATURE_NAMES = ("c_rgb", "c_ir", "abs_conf_diff", "box_iou",
                     "center_dist", "scale_ratio", "aspect_diff", "matched_flag")
    CHECKPOINT_FEATURE_INDICES = (0, 1, 2, 7, 3, 4, 5, 6)
    ACTION_NAMES = ("joint_consensus", "ir_vetoed_rgb_dominant",
                    "rgb_vetoed_thermal_dominant")

    def __init__(self, input_dim=8, feature_mean=None, feature_std=None,
                 abstain_threshold=0.45):
        super().__init__()
        if input_dim != len(self.FEATURE_NAMES):
            raise ValueError(f"S-DEC governance head requires 8 inputs, received {input_dim}")
        self.input_dim = input_dim
        self.abstain_threshold = float(abstain_threshold)
        mean = torch.zeros(input_dim) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(input_dim) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (input_dim,) or std.shape != (input_dim,):
            raise ValueError("governance feature mean/std must each have shape [8]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("governance feature statistics must be finite with positive scales")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(),
                                 nn.Linear(16, 3), nn.Sigmoid())

    def forward(self, features):
        if features.shape[-1] != self.input_dim:
            raise ValueError("governance features must have shape [..., 8]")
        # The bare Colab state_dict predates the canonical geometry-first
        # order and expects [conf, diff, matched, geometry...].
        indices = torch.as_tensor(self.CHECKPOINT_FEATURE_INDICES,
                                  device=features.device)
        checkpoint_features = features.index_select(-1, indices)
        checkpoint_features = (checkpoint_features - self.feature_mean) / self.feature_std
        return self.net(checkpoint_features)

    @staticmethod
    def _box_features(b_rgb, b_ir, img_shape):
        """Return IoU, normalized center distance, signed log area ratio, aspect delta."""
        if b_rgb is None or b_ir is None:
            return 0.0, 1.0, 0.0, 0.0, 0.0
        a = [float(v) for v in b_rgb]
        b = [float(v) for v in b_ir]
        if len(a) != 4 or len(b) != 4:
            raise ValueError("RGB and IR boxes must be [x1,y1,x2,y2]")
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
        area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        overlap = intersection / max(area_a + area_b - intersection, 1e-12)
        if len(img_shape) < 2:
            raise ValueError("img_shape must contain image height and width")
        height, width = float(img_shape[0]), float(img_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        center_a = ((a[0]+a[2])/(2*width), (a[1]+a[3])/(2*height))
        center_b = ((b[0]+b[2])/(2*width), (b[1]+b[3])/(2*height))
        distance = min(1.0, ((center_a[0]-center_b[0])**2 +
                             (center_a[1]-center_b[1])**2)**0.5)
        area_a, area_b = max(area_a, 1.0), max(area_b, 1.0)
        aspect_a = max(a[3]-a[1], 0.0) / max(a[2]-a[0], 1.0)
        aspect_b = max(b[3]-b[1], 0.0) / max(b[2]-b[0], 1.0)
        return overlap, distance, math.log(area_a/area_b), abs(aspect_a-aspect_b), 1.0

    @classmethod
    def load(cls, path, map_location="cpu", abstain_threshold=0.45):
        """Load a compatible trained S-DEC state dict or fail without fallback."""
        data = torch.load(Path(path), map_location=map_location, weights_only=True)
        state = data.get("state_dict", data) if isinstance(data, dict) else data
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain a state dictionary")
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        # Reject incompatible scalar checkpoints instead of silently treating
        # their consensus scores as three-action absolute risks.
        first = state.get("net.0.weight")
        if first is None or tuple(first.shape) != (32, 8):
            raise ValueError("checkpoint is not a trained Config B S-DEC head (expected net.0.weight shape [32, 8])")
        last = state.get("net.4.weight")
        if last is None or tuple(last.shape) != (3, 16):
            raise ValueError("checkpoint does not have the expected 3-action risk output")
        mean = data.get("feature_mean", data.get("mean")) if isinstance(data, dict) else None
        std = data.get("feature_std", data.get("std")) if isinstance(data, dict) else None
        model = cls(8, mean, std, abstain_threshold)
        # The supplied prototype saves a bare network state_dict without
        # normalization buffers; those inputs were unstandardized.
        state.setdefault("feature_mean", model.feature_mean.detach().clone())
        state.setdefault("feature_std", model.feature_std.detach().clone())
        model.load_state_dict(state, strict=True)
        model.to(map_location).eval()
        return model

    @torch.inference_mode()
    def predict_governance(self, b_rgb, c_rgb, b_ir, c_ir, img_shape,
                           available_actions=None):
        """Return ``(0..100 score, action name, abstain)`` for one paired candidate."""
        c_rgb = min(1.0, max(0.0, float(c_rgb or 0.0)))
        c_ir = min(1.0, max(0.0, float(c_ir or 0.0)))
        overlap, distance, scale_ratio, aspect_diff, matched = self._box_features(b_rgb, b_ir, img_shape)
        values = [c_rgb, c_ir, abs(c_rgb-c_ir), overlap, distance,
                  scale_ratio, aspect_diff, matched]
        x = torch.tensor([values], dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        risks = self(x)[0]
        valid_indices = (list(range(len(self.ACTION_NAMES))) if available_actions is None else
                         [index for index, name in enumerate(self.ACTION_NAMES)
                          if name in set(available_actions)])
        if not valid_indices:
            raise ValueError("available_actions must include at least one known governance action")
        valid_risks = risks[valid_indices]
        min_risk, relative_action_idx = torch.min(valid_risks, dim=0)
        action_idx = valid_indices[int(relative_action_idx.item())]
        abstain = bool(min_risk.item() > self.abstain_threshold)
        if abstain:
            action_name = "abstain"
            score = round(100.0 * (1.0 - float(min_risk.item())))
        else:
            action_name = self.ACTION_NAMES[int(action_idx)]
            confidence = (max(c_rgb, c_ir) if int(action_idx) == 0
                          else c_rgb if int(action_idx) == 1 else c_ir)
            score = round(100.0 * confidence)
        return int(max(0, min(100, score))), action_name, abstain
