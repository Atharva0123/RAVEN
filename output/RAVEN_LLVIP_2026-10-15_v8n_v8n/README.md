# Archived `v8n_v8n` LLVIP results

This folder contains the compact result and provenance files selected from the `raven_results_v8n_v8n` archive provided on 2026-10-21. The run summary records an output directory of `/content/raven_results/v8n_v8n`, Python 3.13.15, PyTorch 2.11.0+cu130, and Ultralytics 8.4.177. It evaluated the full 3,463-pair LLVIP test split with frozen YOLOv8n detectors for both modalities.

## Files

- `run_summary.json`: run configuration, versions, benchmark metrics, hashes, and evaluation metadata.
- `pairing_summary.json`: detector pairing, weights and hashes, test-pair count, and candidate counts.
- `table8_benchmark.csv`: compact clean benchmark table.
- `degradation_summary.csv`: aggregate clean and simulated-degradation metrics by policy.
- `degradation_per_run.csv`: seed-level and baseline records underlying the aggregate table.
- `degradation_bootstrap.csv`: paired image-bootstrap comparisons and intervals.
- `degradation_map50.png`: plot of degradation AP@50 results.

`table8_benchmark.csv` is the compact four-method clean table for the seed-42 benchmark; its RAVEN S-DEC row is 0.7056 AP@50. It does not include WBF or ablation rows. Use `degradation_summary.csv` for the clean policy comparison, WBF and ablations; RAVEN variants there report means and standard deviations over three head seeds, while simple baselines are single-run. `degradation_per_run.csv` contains the underlying run records.

The archive also contained model checkpoints, serialized per-image records, image-level error analysis, and training logs. Those are not included here because the checked-in notebook and code are the source for rerunning the experiment, while these large or secondary artifacts are not needed to inspect the reported tables. The LLVIP data and detector weights are not included.

The notebook describes the score as **custom all-point AP@50**. Degradation results are from deterministic simulated perturbations; image-level bootstrap intervals do not account for scene/clip clustering or training uncertainty. See the repository reproducibility guide and paper audit for limitations.
