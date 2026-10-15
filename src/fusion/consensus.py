"""Deterministic confidence-weighted RGB/thermal consensus."""


class ConsensusFusion:
    """Deterministic review-score fallback when no trained confidence head is installed."""
    def __init__(self, model_path=None, weights=None,
                 disagreement_threshold=0.4, minimum_modal_confidence=0.05):
        # Preserve optional legacy argument for caller compatibility.
        self.model_path = str(model_path) if model_path is not None else None
        self.weights = weights or {"posture": 0.55, "tracking": 0.30,
                                   "recovery": 0.05, "zone": 0.10}
        self.disagreement_threshold = float(disagreement_threshold)
        self.minimum_modal_confidence = float(minimum_modal_confidence)

    def evaluate(self, rgb_data, thermal_data, zone_context=0):
        rgb_conf = min(1.0, max(0.0, float((rgb_data or {}).get("posture_conf", 0.0))))
        thermal_conf = min(1.0, max(0.0, float((thermal_data or {}).get("posture_conf", 0.0))))
        rgb_valid = rgb_data is not None and rgb_conf >= self.minimum_modal_confidence
        thermal_valid = thermal_data is not None and thermal_conf >= self.minimum_modal_confidence
        track_conf = max(float((rgb_data or {}).get("track_conf", 0.0)),
                         float((thermal_data or {}).get("track_conf", 0.0)))
        recovery = int(bool((rgb_data or {}).get("recovered", False) or
                            (thermal_data or {}).get("recovered", False)))
        zone = min(1.0, max(0.0, float(zone_context)))
        if rgb_valid and thermal_valid:
            modality_conf, disagreement = (rgb_conf + thermal_conf) / 2, abs(rgb_conf - thermal_conf)
        elif rgb_valid:
            modality_conf, disagreement = rgb_conf * 0.7, 1.0
        elif thermal_valid:
            modality_conf, disagreement = thermal_conf * 0.7, 1.0
        else:
            modality_conf, disagreement = 0.0, 0.0
        score = 100 * (self.weights["posture"] * modality_conf +
                       self.weights["tracking"] * min(1.0, max(0.0, track_conf)) +
                       self.weights["recovery"] * recovery + self.weights["zone"] * zone)
        review = ((rgb_valid and thermal_valid and disagreement > self.disagreement_threshold)
                  or not (rgb_valid or thermal_valid))
        evidence = {"rgb_contribution": rgb_conf, "thermal_contribution": thermal_conf,
                    "track_confidence": track_conf, "occlusion_recovery": recovery,
                    "zone_context": zone, "disagreement_level": disagreement,
                    "available_modalities": [name for name, valid in
                        (("rgb", rgb_valid), ("thermal", thermal_valid)) if valid],
                    "fusion_method": "weighted_signal_consensus_v2",
                    "score_is_risk_probability": False}
        return int(round(score)), evidence, review
