#!/usr/bin/env python3
"""Runner for PALF Cross-Fitted Ablation Experiment (v18).

Usage:
    python scripts/116_run_palf_crossfit_ablation.py --config configs/iclr/palf_crossfit_ablation.yaml --mode smoke
    python scripts/116_run_palf_crossfit_ablation.py --config configs/iclr/palf_crossfit_ablation.yaml --mode production
    python scripts/116_run_palf_crossfit_ablation.py --config configs/iclr/palf_crossfit_ablation.yaml --mode status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (
    AblationCondition,
    CONDITIONS,
    N_ROI,
    run_ablation_experiment,
    compute_seed_metrics,
    compute_component_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("palf_ablation_runner")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict):
    """Load FC, SC, labels, and subject ordering.

    Returns upper-triangle features (n_subjects, 6670).
    """
    fc_mats = np.load(REPO_ROOT / cfg["data"]["fc_path"])
    sc_mats = np.load(REPO_ROOT / cfg["data"]["sc_path"])

    subjects_df = pd.read_csv(REPO_ROOT / cfg["data"]["subjects_path"])
    n_subjects = len(subjects_df)

    # Validate dimensions
    assert fc_mats.shape[0] == n_subjects, f"FC subjects {fc_mats.shape[0]} != manifest {n_subjects}"
    assert sc_mats.shape[0] == n_subjects, f"SC subjects {sc_mats.shape[0]} != manifest {n_subjects}"
    assert fc_mats.shape[1] == 116, f"FC matrix size {fc_mats.shape[1]} != 116"

    # Extract upper triangle features (n_subjects, 6670)
    iu = np.triu_indices(116, k=1)
    fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)

    assert fc.shape == (n_subjects, 6670), f"FC features {fc.shape} != ({n_subjects}, 6670)"
    assert sc.shape == (n_subjects, 6670), f"SC features {sc.shape} != ({n_subjects}, 6670)"

    # Check for NaN/Inf
    assert np.all(np.isfinite(fc)), "FC contains NaN/Inf"
    assert np.all(np.isfinite(sc)), "SC contains NaN/Inf"

    return fc, sc, n_subjects


def load_prior(path: str) -> np.ndarray:
    """Load ROI prior vector from CSV."""
    df = pd.read_csv(REPO_ROOT / path)
    # Expect columns: roi_index, roi_label, prior_score (or raw_score)
    if "prior_score" in df.columns:
        scores = df["prior_score"].values
    elif "raw_score" in df.columns:
        scores = df["raw_score"].values
    else:
        raise ValueError(f"Prior CSV has no score column: {df.columns.tolist()}")
    assert len(scores) == N_ROI, f"Prior has {len(scores)} entries, expected {N_ROI}"
    return scores.astype(np.float64)


def load_targets(cfg: dict, n_subjects: int, task_key: str) -> np.ndarray:
    """Load target variable for a specific task."""
    label_path = REPO_ROOT / cfg["targets"][task_key]["label_path"]
    y = np.load(label_path)
    assert len(y) == n_subjects, f"Labels {len(y)} != subjects {n_subjects}"
    return y.astype(np.float64)


def file_hash(path: str) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_run_fingerprint(cfg: dict, data_hashes: dict) -> str:
    """Compute deterministic fingerprint for this run configuration."""
    import hashlib
    content = json.dumps({
        "config": cfg,
        "data_hashes": data_hashes,
    }, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_split_manifest(result, output_dir: Path):
    """Save split manifest with outer/fusion/selection indices."""
    manifest = []
    for split in result.splits:
        manifest.append({
            "seed": split.seed,
            "outer_fold": split.outer_fold,
            "condition": split.condition_id,
            "train_idx": split.train_idx.tolist(),
            "test_idx": split.test_idx.tolist(),
            "fusion_weights": split.fusion_weights,
            "fc_alpha": split.fc_selected_alpha,
            "sc_alpha": split.sc_selected_alpha,
            "fp_params": split.fp_selected_params,
        })
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def save_outer_metrics(result, output_dir: Path):
    """Save per-split outer metrics."""
    rows = []
    for split in result.splits:
        for model_name, metrics in [
            ("FC", split.fc_metrics),
            ("SC", split.sc_metrics),
            ("FP", split.fp_metrics),
            ("fused", split.fused_metrics),
            ("equal_weight", split.equal_weight_metrics),
        ]:
            rows.append({
                "task": result.task_name,
                "seed": split.seed,
                "outer_fold": split.outer_fold,
                "condition": split.condition_id,
                "model": model_name,
                "pearson": metrics["pearson"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "n_train": len(split.train_idx),
                "n_test": len(split.test_idx),
            })
    pd.DataFrame(rows).to_csv(output_dir / "outer_metrics.csv", index=False)


def save_coefficients(result, output_dir: Path):
    """Save final FC coefficient vectors for all conditions."""
    coeff_dir = output_dir / "coefficients"
    coeff_dir.mkdir(exist_ok=True)
    for split in result.splits:
        fp_final = split.fp_final
        if fp_final.alpha is not None and fp_final.scaler is not None:
            # Save alpha (dual coefficients) and metadata
            fname = f"cond_{split.condition_id}_seed{split.seed:02d}_fold{split.outer_fold:02d}.npz"
            np.savez(
                coeff_dir / fname,
                alpha=fp_final.alpha,
                y_mean=fp_final.y_mean,
                y_std=fp_final.y_std,
                scaler_mean=fp_final.scaler.mean_,
                scaler_scale=fp_final.scaler.scale_,
                selected_params= np.array([fp_final.selected_params.get("lambda_fc", 0.0),
                                           fp_final.selected_params.get("lambda_l", 0.0)]),
            )


def save_fusion_weights(result, output_dir: Path):
    """Save fusion weights for each split."""
    rows = []
    for split in result.splits:
        fw = split.fusion_weights
        w_primary = fw.get("FP", fw.get("FC", 0.5))
        rows.append({
            "task": result.task_name,
            "seed": split.seed,
            "outer_fold": split.outer_fold,
            "condition": split.condition_id,
            "w_primary": w_primary,
            "w_SC": fw.get("SC", 0.5),
        })
    pd.DataFrame(rows).to_csv(output_dir / "fusion_weights.csv", index=False)


def _stringify_keys(obj):
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_stringify_keys(item) for item in obj]
    return obj


def save_selection_scores(result, output_dir: Path):
    """Save selection scores for each split."""
    sel_dir = output_dir / "selection_scores"
    sel_dir.mkdir(exist_ok=True)
    for split in result.splits:
        fname = f"cond_{split.condition_id}_seed{split.seed:02d}_fold{split.outer_fold:02d}.json"
        with open(sel_dir / fname, "w") as f:
            json.dump(_stringify_keys(split.selection_scores), f, indent=2, default=str)


def save_validation_report(output_dir: Path, report: dict):
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


def save_run_manifest(output_dir: Path, cfg: dict, fingerprint: str, data_hashes: dict, start_time: float):
    manifest = {
        "config": cfg,
        "fingerprint": fingerprint,
        "data_hashes": data_hashes,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pid": os.getpid(),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def save_protocol(output_dir: Path, cfg: dict, fingerprint: str):
    protocol = {
        "experiment": "PALF Cross-Fitted Ablation",
        "fingerprint": fingerprint,
        "conditions": {k: {"id": v.id, "name": v.name, "use_anisotropy": v.use_anisotropy,
                           "use_network": v.use_network, "lambda_l_grid": list(v.lambda_l_grid)}
                       for k, v in CONDITIONS.items()},
        "ridge_grid": cfg["ridge_grid"],
        "fusion_weight_step": 0.05,
        "selection_rule": "Pearson first, RMSE tiebreak, MAE tiebreak, larger SC weight on exact tie",
        "fusion_selection_rule": "Pearson first, RMSE tiebreak, MAE tiebreak",
        "cross_fitting": "Each OOF prediction excludes held-out subjects from preprocessing, selection, and fitting",
        "final_reselection": "3-fold CV on full outer-training set",
        "solver_scale": f"C = 2 * N_EDGE = {2 * 6670}",
    }
    with open(output_dir / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="PALF Cross-Fitted Ablation Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["smoke", "production", "status"], default="production",
                        help="Run mode: smoke (1 seed), production (all), status (check progress)")
    parser.add_argument("--task", choices=["working_memory", "fluid_intelligence", "both"], default="both",
                        help="Which task(s) to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start_time = time.time()

    if args.mode == "status":
        # Check existing run
        base = Path(REPO_ROOT / cfg.get("output_base", "outputs/iclr/palf_crossfit_ablation_v1"))
        if base.exists():
            for f in sorted(base.iterdir()):
                if f.is_file():
                    print(f"  {f.name}: {f.stat().st_size:,} bytes")
            if (base / "COMPLETE").exists():
                print("STATUS: COMPLETE")
            else:
                print("STATUS: INCOMPLETE")
        else:
            print("STATUS: NOT STARTED")
        return

    # Load data
    logger.info("Loading data...")
    fc, sc, n_subjects = load_data(cfg)
    logger.info(f"Loaded {n_subjects} subjects, FC shape={fc.shape}, SC shape={sc.shape}")

    # Load targets
    targets = {}
    for task_key in cfg["targets"]:
        targets[task_key] = load_targets(cfg, n_subjects, task_key)
        logger.info(f"  {task_key}: y shape={targets[task_key].shape}, "
                     f"mean={targets[task_key].mean():.4f}, std={targets[task_key].std():.4f}")

    # Load priors
    priors = {}
    for task_key in cfg["priors"]:
        priors[task_key] = {}
        for prior_type, prior_path in cfg["priors"][task_key].items():
            priors[task_key][prior_type] = load_prior(prior_path)
            logger.info(f"  {task_key}/{prior_type}: range=[{priors[task_key][prior_type].min():.4f}, {priors[task_key][prior_type].max():.4f}]")

    # Compute data hashes for provenance
    data_hashes = {
        "fc": hashlib.sha256(fc.tobytes()).hexdigest()[:16],
        "sc": hashlib.sha256(sc.tobytes()).hexdigest()[:16],
    }
    for task_key in targets:
        data_hashes[task_key] = hashlib.sha256(targets[task_key].tobytes()).hexdigest()[:16]

    fingerprint = compute_run_fingerprint(cfg, data_hashes)
    logger.info(f"Run fingerprint: {fingerprint}")

    # Determine output directory
    smoke_mode = args.mode == "smoke"
    base_dir = cfg.get("smoke_output_base" if smoke_mode else "output_base",
                       "outputs/iclr/palf_crossfit_ablation_smoke" if smoke_mode else "outputs/iclr/palf_crossfit_ablation_v1")
    output_base = Path(REPO_ROOT / base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Save protocol and manifest
    save_protocol(output_base, cfg, fingerprint)
    save_run_manifest(output_base, cfg, fingerprint, data_hashes, start_time)

    # Determine which tasks to run
    task_keys = [args.task] if args.task != "both" else list(cfg["targets"].keys())

    all_results = {}
    for task_key in task_keys:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation for: {task_key}")
        logger.info(f"{'='*60}")

        task_output = output_base / task_key
        task_output.mkdir(exist_ok=True)

        # Check for existing checkpoint
        checkpoint_file = task_output / "checkpoint.pkl"
        if args.resume and checkpoint_file.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "rb") as f:
                result = pickle.load(f)
        else:
            # Use matched prior for this task
            matched_prior = priors[task_key]["matched"]

            result = run_ablation_experiment(
                X_fc=fc,
                X_sc=sc,
                y=targets[task_key],
                task_name=task_key,
                roi_prior=matched_prior,
                seeds=cfg["seeds"],
                n_outer_folds=cfg["outer_folds"],
                ridge_grid=cfg["ridge_grid"],
                n_fusion_folds=cfg.get("fusion_folds", 3),
                n_inner=cfg.get("inner_folds", 3),
                n_final_cv=cfg.get("final_cv_folds", 3),
                n_rois=cfg["n_rois"],
                output_dir=str(task_output),
                condition_ids=cfg.get("conditions", ["R0", "R1", "R2", "R3"]),
                smoke_mode=smoke_mode,
            )

            # Save checkpoint
            with open(checkpoint_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")

        all_results[task_key] = result

        # Save outputs
        save_split_manifest(result, task_output)
        save_outer_metrics(result, task_output)
        save_coefficients(result, task_output)
        save_fusion_weights(result, task_output)
        save_selection_scores(result, task_output)

        # Seed-level and component summaries
        seed_df = compute_seed_metrics(result)
        seed_df.to_csv(task_output / "seed_metrics.csv", index=False)

        comp_df = compute_component_summary(result)
        comp_df.to_csv(task_output / "component_summary.csv", index=False)

        logger.info(f"Saved outputs to {task_output}")

    # Mark complete
    (output_base / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    elapsed = time.time() - start_time
    logger.info(f"\nAll tasks complete. Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info("STATUS: COMPLETE_FOR_AUTHOR_REVIEW")


if __name__ == "__main__":
    main()
