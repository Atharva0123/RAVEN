"""Train the temporal fusion architecture on deliberately synthetic sequences.

This is a software smoke-training exercise only. The random/noise labels do
not represent validated real-world anomaly labels and the resulting checkpoint
must not be interpreted as an operational threat detector.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fusion.temporal_cross_modal import TemporalCrossModalFusion


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "models" / "best_model.pth"
METADATA = OUTPUT.with_suffix(".metadata.json")


def main() -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # Keep this small synthetic run friendly to the local workstation.
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    # Each input is shaped (100, 32, 128): N clips, T frames, D features.
    normal_rgb = torch.randn(100, 32, 128)
    normal_thermal = normal_rgb + 0.05 * torch.randn_like(normal_rgb)
    anomaly_rgb = normal_rgb + 0.8 * torch.randn_like(normal_rgb)
    anomaly_thermal = normal_thermal + 0.8 * torch.randn_like(normal_thermal)

    rgb = torch.cat((normal_rgb, anomaly_rgb), dim=0)
    thermal = torch.cat((normal_thermal, anomaly_thermal), dim=0)
    labels = torch.cat((torch.zeros(100), torch.ones(100)), dim=0)
    permutation = torch.randperm(len(labels))
    dataset = TensorDataset(rgb[permutation], thermal[permutation], labels[permutation])
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = TemporalCrossModalFusion(feature_dim=128).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        seen = 0
        for rgb_batch, thermal_batch, target in loader:
            rgb_batch = rgb_batch.to(device)
            thermal_batch = thermal_batch.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            probabilities = torch.sigmoid(model(rgb_batch, thermal_batch))
            loss = loss_fn(probabilities, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        print(f"epoch={epoch}/5 bce={total_loss / max(seen, 1):.5f}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep the requested plain state_dict checkpoint format.
    torch.save(model.state_dict(), OUTPUT)
    METADATA.write_text(json.dumps({
        "training_source": "synthetic_random_noise_smoke_test",
        "synthetic": True,
        "validated_on_real_data": False,
        "feature_dim": 128,
        "sequence_length": 32,
        "normal_sequences": 100,
        "synthetic_anomaly_sequences": 100,
        "epochs": 5,
        "operational_warning": "Not a real-world anomaly detector; do not use scores as alerts.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved={OUTPUT}", flush=True)
    print(f"metadata={METADATA}", flush=True)
    print("checkpoint_status=synthetic_and_unvalidated", flush=True)


if __name__ == "__main__":
    main()
