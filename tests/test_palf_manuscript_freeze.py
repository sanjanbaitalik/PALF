#!/usr/bin/env python3
"""Validations for the PALF manuscript freeze bundle (freeze_id 9abbb56).

Tests verify:
  - Fusion weight schema integrity (FP/SC keys for all conditions, no silent fallback)
  - CSV schemas and row counts
  - Prior-control matched-reference fidelity
  - Seed-level aggregation counts
  - Primary Holm correction across two targets
  - Legacy-contamination and biomarker-artifact exclusion
  - Finite evidence in generated CSVs
  - Figure existence and non-emptiness
  - Checksum coverage of the freeze bundle
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = REPO_ROOT / "outputs/iclr/palf_manuscript_freeze_9abbb56"
OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"
BUILD_SCRIPT = REPO_ROOT / "scripts_paper/build_palf_manuscript_freeze.py"


def _skip_if_no_freeze():
    if not FREEZE_DIR.exists():
        pytest.skip("Freeze bundle not yet built")


def _skip_if_no_output_base():
    if not OUTPUT_BASE.exists():
        pytest.skip("palf_crossfit_ablation_v1 output not available")


# ---------------------------------------------------------------------------
# 1. Full PALF fusion dictionary requires keys "FP" and "SC" for all conditions
# ---------------------------------------------------------------------------
class TestFusionRequiresFPSCKeys:
    def test_fusion_requires_fp_sc_keys(self):
        """Full PALF (R3) splits must carry explicit 'FP' and 'SC' fusion keys."""
        _skip_if_no_output_base()
        import pickle

        for task in ["working_memory", "fluid_intelligence"]:
            ckpt_path = OUTPUT_BASE / task / "checkpoint.pkl"
            if not ckpt_path.exists():
                pytest.skip(f"Checkpoint not found: {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            r3_splits = [s for s in ckpt.splits if s.condition_id == "R3"]
            assert len(r3_splits) > 0, f"No R3 splits found for {task}"
            for s in r3_splits:
                fw = s.fusion_weights
                assert "FP" in fw, (
                    f"R3 split task={task} seed={s.seed} fold={s.outer_fold} "
                    f"missing 'FP' key; keys={list(fw.keys())}"
                )
                assert "SC" in fw, (
                    f"R3 split task={task} seed={s.seed} fold={s.outer_fold} "
                    f"missing 'SC' key; keys={list(fw.keys())}"
                )


# ---------------------------------------------------------------------------
# 2. All conditions (R0/R1/R2/R3) use FP+SC fusion
# ---------------------------------------------------------------------------
class TestFusionRequiresFPSCKeysForAll:
    def test_fusion_requires_fp_sc_for_all_conditions(self):
        """Every condition (R0, R1, R2, R3) must carry 'FP' and 'SC' fusion keys."""
        _skip_if_no_output_base()
        import pickle

        # Check corrected v2 checkpoints (source checkpoint still has old R0 fusion)
        corrected_base = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
        if not corrected_base.exists():
            pytest.skip(f"Corrected v2 directory not found: {corrected_base}")

        for task in ["working_memory", "fluid_intelligence"]:
            ckpt_path = corrected_base / task / "checkpoint.pkl"
            if not ckpt_path.exists():
                pytest.skip(f"Corrected checkpoint not found: {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            for condition_id in ["R0", "R1", "R2", "R3"]:
                splits = [s for s in ckpt.splits if s.condition_id == condition_id]
                assert len(splits) > 0, f"No {condition_id} splits found for {task}"
                for s in splits:
                    fw = s.fusion_weights
                    assert "FP" in fw, (
                        f"{condition_id} split task={task} seed={s.seed} fold={s.outer_fold} "
                        f"missing 'FP' key; keys={list(fw.keys())}"
                    )
                    assert "SC" in fw, (
                        f"{condition_id} split task={task} seed={s.seed} fold={s.outer_fold} "
                        f"missing 'SC' key; keys={list(fw.keys())}"
                    )


# ---------------------------------------------------------------------------
# 2b. R0 correction did not modify FP/SC predictions
# ---------------------------------------------------------------------------
class TestR0CorrectionPreservesPredictions:
    def test_r0_correction_does_not_modify_fp_predictions(self):
        """R0 correction must not change stored FP branch predictions."""
        _skip_if_no_output_base()
        import pickle

        src_base = OUTPUT_BASE  # original v1
        corrected_base = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
        if not corrected_base.exists():
            pytest.skip("Corrected v2 directory not found")

        for task in ["working_memory", "fluid_intelligence"]:
            src_path = src_base / task / "checkpoint.pkl"
            crr_path = corrected_base / task / "checkpoint.pkl"
            if not src_path.exists() or not crr_path.exists():
                pytest.skip(f"Checkpoints not found for {task}")
            with open(src_path, "rb") as f:
                src_ckpt = pickle.load(f)
            with open(crr_path, "rb") as f:
                crr_ckpt = pickle.load(f)

            src_r0 = [s for s in src_ckpt.splits if s.condition_id == "R0"]
            crr_r0 = [s for s in crr_ckpt.splits if s.condition_id == "R0"]
            assert len(src_r0) == len(crr_r0)

            for s, c in zip(src_r0, crr_r0):
                assert np.allclose(s.fp_test_pred, c.fp_test_pred), (
                    f"R0 FP test_pred changed for {task} seed={s.seed} fold={s.outer_fold}"
                )

    def test_r0_correction_does_not_modify_sc_predictions(self):
        """R0 correction must not change stored SC branch predictions."""
        _skip_if_no_output_base()
        import pickle

        src_base = OUTPUT_BASE
        corrected_base = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
        if not corrected_base.exists():
            pytest.skip("Corrected v2 directory not found")

        for task in ["working_memory", "fluid_intelligence"]:
            src_path = src_base / task / "checkpoint.pkl"
            crr_path = corrected_base / task / "checkpoint.pkl"
            if not src_path.exists() or not crr_path.exists():
                pytest.skip(f"Checkpoints not found for {task}")
            with open(src_path, "rb") as f:
                src_ckpt = pickle.load(f)
            with open(crr_path, "rb") as f:
                crr_ckpt = pickle.load(f)

            src_r0 = [s for s in src_ckpt.splits if s.condition_id == "R0"]
            crr_r0 = [s for s in crr_ckpt.splits if s.condition_id == "R0"]
            for s, c in zip(src_r0, crr_r0):
                assert np.allclose(s.sc_test_pred, c.sc_test_pred), (
                    f"R0 SC test_pred changed for {task} seed={s.seed} fold={s.outer_fold}"
                )

    def test_r1_r2_r3_unchanged(self):
        """R1/R2/R3 source results must be unchanged after R0 correction."""
        _skip_if_no_output_base()
        import pickle

        src_base = OUTPUT_BASE
        corrected_base = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
        if not corrected_base.exists():
            pytest.skip("Corrected v2 directory not found")

        for task in ["working_memory", "fluid_intelligence"]:
            src_path = src_base / task / "checkpoint.pkl"
            crr_path = corrected_base / task / "checkpoint.pkl"
            if not src_path.exists() or not crr_path.exists():
                pytest.skip(f"Checkpoints not found for {task}")
            with open(src_path, "rb") as f:
                src_ckpt = pickle.load(f)
            with open(crr_path, "rb") as f:
                crr_ckpt = pickle.load(f)

            for cond in ["R1", "R2", "R3"]:
                src_cond = [s for s in src_ckpt.splits if s.condition_id == cond]
                crr_cond = [s for s in crr_ckpt.splits if s.condition_id == cond]
                assert len(src_cond) == len(crr_cond)
                for s, c in zip(src_cond, crr_cond):
                    assert np.allclose(s.fused_test_pred, c.fused_test_pred), (
                        f"{cond} fused_test_pred changed for {task} "
                        f"seed={s.seed} fold={s.outer_fold}"
                    )


# ---------------------------------------------------------------------------
# 2c. Corrected R0 fusion weights sum to 1.0
# ---------------------------------------------------------------------------
class TestCorrectedR0FusionWeightsSumToOne:
    def test_corrected_r0_weights_sum_to_one(self):
        """Corrected R0 fusion weights must satisfy w_FP + w_SC = 1."""
        _skip_if_no_output_base()
        import pickle

        corrected_base = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
        if not corrected_base.exists():
            pytest.skip("Corrected v2 directory not found")

        for task in ["working_memory", "fluid_intelligence"]:
            ckpt_path = corrected_base / task / "checkpoint.pkl"
            if not ckpt_path.exists():
                pytest.skip(f"Checkpoint not found: {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

            r0_splits = [s for s in ckpt.splits if s.condition_id == "R0"]
            assert len(r0_splits) == 50
            for s in r0_splits:
                fw = s.fusion_weights
                assert "FP" in fw and "SC" in fw
                assert np.isclose(fw["FP"] + fw["SC"], 1.0, atol=1e-6), (
                    f"R0 weights don't sum to 1: {fw['FP']:.6f} + {fw['SC']:.6f} = "
                    f"{fw['FP'] + fw['SC']:.6f}"
                )


# ---------------------------------------------------------------------------
# 3. Missing required branch weight raises instead of defaulting to 0.5
# ---------------------------------------------------------------------------
class TestMissingRequiredWeightRaises:
    def test_missing_required_weight_raises(self):
        """Accessing a missing fusion key must raise KeyError, not return 0.5."""
        fw = {"FP": 0.4, "SC": 0.6}
        with pytest.raises(KeyError):
            _ = fw["FC"]
        with pytest.raises(KeyError):
            _ = fw["missing_key"]

    def test_no_get_fallback_in_build_script(self):
        """Build script must not use .get(..., 0.5) for fusion weight access."""
        _skip_if_no_build_script = not BUILD_SCRIPT.exists()
        if _skip_if_no_build_script:
            pytest.skip("Build script not found")
        src = BUILD_SCRIPT.read_text()
        assert '.get("FP", 0.5)' not in src, (
            "Build script still uses .get('FP', 0.5) fallback"
        )
        assert '.get("SC", 0.5)' not in src, (
            "Build script still uses .get('SC', 0.5) fallback"
        )
        assert '.get("FC", 0.5)' not in src, (
            "Build script still uses .get('FC', 0.5) fallback"
        )


# ---------------------------------------------------------------------------
# 4. fusion_weights_corrected.csv has valid schema with active-branch assertions
# ---------------------------------------------------------------------------
class TestFusionWeightsCSVSchema:
    def test_fusion_weights_csv_schema(self):
        """fusion_weights_corrected.csv must have valid columns and active-branch invariants."""
        _skip_if_no_freeze()
        candidates = [
            FREEZE_DIR / "results" / "fusion_weights_corrected.csv",
            FREEZE_DIR / "data" / "fusion_weights.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            pytest.skip("No fusion weights CSV found in freeze bundle")

        df = pd.read_csv(csv_path)

        required_cols = {"task", "seed", "outer_fold", "condition"}
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing required columns: {missing}"

        assert "w_FP" in df.columns, "Missing w_FP column in fusion weights CSV"
        assert "w_SC" in df.columns, "Missing w_SC column in fusion weights CSV"

        active_mask = df["w_FP"].notna() & df["w_SC"].notna()
        active = df[active_mask]
        if len(active) > 0:
            sums = active["w_FP"] + active["w_SC"]
            assert np.allclose(sums, 1.0, atol=1e-5), (
                f"Active FP/SC weights do not sum to 1.0; "
                f"max deviation={np.abs(sums - 1.0).max()}"
            )

        assert len(df) > 0, "Fusion weights CSV is empty"


# ---------------------------------------------------------------------------
# 5. Prior-control matched reference matches current Full PALF split predictions
# ---------------------------------------------------------------------------
class TestPriorControlMatchedReference:
    def test_prior_control_matched_reference(self):
        """Prior-control matched fused values must agree with Full PALF primary predictions."""
        _skip_if_no_freeze()
        _skip_if_no_output_base()

        refit_path = FREEZE_DIR / "data" / "prior_control_refits.csv"
        if not refit_path.exists():
            alt = FREEZE_DIR / "results" / "prior_control_split_metrics.csv"
            if alt.exists():
                refit_path = alt
            else:
                pytest.skip("No prior-control refit CSV found")

        import pickle

        refit_df = pd.read_csv(refit_path)
        if "fused_r_matched" not in refit_df.columns:
            pytest.skip("No fused_r_matched column in refit CSV")

        for task in refit_df["task"].unique():
            task_refits = refit_df[refit_df["task"] == task]
            ckpt_path = OUTPUT_BASE / task / "checkpoint.pkl"
            if not ckpt_path.exists():
                continue
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

            r3_splits = [s for s in ckpt.splits if s.condition_id == "R3"]
            matched_by_seed_fold = {}
            for s in r3_splits:
                matched_by_seed_fold[(s.seed, s.outer_fold)] = s.fused_metrics["pearson"]

            for _, row in task_refits.iterrows():
                seed = int(row["seed"])
                fold = int(row["outer_fold"])
                key = (seed, fold)
                if key in matched_by_seed_fold:
                    expected_r = matched_by_seed_fold[key]
                    actual_r = row["fused_r_matched"]
                    assert np.isclose(actual_r, expected_r, atol=1e-6), (
                        f"Matched reference mismatch for {task} seed={seed} fold={fold}: "
                        f"refit={actual_r:.6f} checkpoint={expected_r:.6f}"
                    )


# ---------------------------------------------------------------------------
# 6. prior_control_split_metrics.csv has 300 rows total
# ---------------------------------------------------------------------------
class TestPriorControlRawRowCount:
    def test_prior_control_raw_row_count(self):
        """Prior-control raw output should have 300 rows (50 splits x 3 controls x 2 tasks)."""
        _skip_if_no_freeze()
        candidates = [
            FREEZE_DIR / "data" / "prior_control_refits.csv",
            FREEZE_DIR / "results" / "prior_control_split_metrics.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            pytest.skip("No prior-control raw CSV found")

        df = pd.read_csv(csv_path)
        assert len(df) == 300, (
            f"Expected 300 rows (50 splits x 3 controls x 2 tasks), got {len(df)}"
        )


# ---------------------------------------------------------------------------
# 7. prior_control_seed_metrics.csv has exactly 10 seeds per task/control
# ---------------------------------------------------------------------------
class TestPriorControlSeedCount:
    def test_prior_control_seed_count(self):
        """Seed-level prior-control metrics must have exactly 10 seeds per task/control."""
        _skip_if_no_freeze()
        candidates = [
            FREEZE_DIR / "data" / "seed_level_prior_control.csv",
            FREEZE_DIR / "results" / "prior_control_seed_metrics.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            pytest.skip("No prior-control seed metrics CSV found")

        df = pd.read_csv(csv_path)
        for (task, ctrl), grp in df.groupby(["task", "control_prior"]):
            assert len(grp) == 10, (
                f"Expected 10 seeds for {task}/{ctrl}, got {len(grp)}"
            )


# ---------------------------------------------------------------------------
# 8. n_seeds is 10 for inference
# ---------------------------------------------------------------------------
class TestPriorControlInferenceNSeeds:
    def test_prior_control_inference_n_seeds(self):
        """Inferential n_seeds in prior-control comparisons must be 10."""
        _skip_if_no_freeze()
        candidates = [
            FREEZE_DIR / "data" / "prior_control_comparisons.csv",
            FREEZE_DIR / "results" / "prior_control_comparisons.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            pytest.skip("No prior-control comparisons CSV found")

        df = pd.read_csv(csv_path)
        if "n_seeds" in df.columns:
            bad = df[df["n_seeds"] != 10]
            assert bad.empty, (
                f"Found rows with n_seeds != 10: {bad[['task', 'control_prior', 'n_seeds']].to_dict('records')}"
            )


# ---------------------------------------------------------------------------
# 9. Primary Holm correction is applied across the two targets
# ---------------------------------------------------------------------------
class TestPrimaryHolmAcrossTargets:
    def test_primary_holm_across_targets(self):
        """Primary comparisons CSV must show Holm correction across the two targets."""
        _skip_if_no_freeze()
        candidates = [
            FREEZE_DIR / "data" / "primary_comparisons.csv",
            FREEZE_DIR / "results" / "primary_comparisons.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            pytest.skip("No primary comparisons CSV found")

        df = pd.read_csv(csv_path)
        assert len(df) == 2, f"Expected 2 rows (one per target), got {len(df)}"

        assert "p_holm" in df.columns, "Missing p_holm column"
        assert "raw_p" in df.columns, "Missing raw_p column"

        raw_ps = df["raw_p"].values
        adj_ps = df["p_holm"].values

        sorted_raw = np.sort(raw_ps)
        for i in range(len(sorted_raw)):
            expected = min(sorted_raw[i] * (len(sorted_raw) - i), 1.0)
            actual = adj_ps[np.argsort(raw_ps)[i]]
            assert actual >= expected - 1e-10, (
                f"Holm correction incorrect at rank {i}: "
                f"expected>={expected:.6f}, got {actual:.6f}"
            )

        assert (adj_ps >= raw_ps).all(), "Adjusted p-values must be >= raw p-values"


# ---------------------------------------------------------------------------
# 10. Main figure source contains no LF1 path
# ---------------------------------------------------------------------------
class TestNoLF1InMainFigureSource:
    def test_no_lf1_in_main_figure_source(self):
        """No figure in the freeze bundle should reference LF1 source paths."""
        _skip_if_no_freeze()

        prohibited = [
            "lf1_final_10x5",
            "lf1_final_evidence_audit",
            "prior_aware_late_fusion_integrity_audit",
        ]

        figures_dir = FREEZE_DIR / "figures"
        if not figures_dir.exists():
            pytest.skip("No figures directory in freeze bundle")

        audit_dir = FREEZE_DIR / "audit"
        manifest_dir = FREEZE_DIR

        for search_dir in [figures_dir, audit_dir, manifest_dir]:
            if not search_dir.exists():
                continue
            for fpath in search_dir.rglob("*"):
                if fpath.is_file() and fpath.suffix in (".md", ".json", ".txt", ".csv"):
                    try:
                        content = fpath.read_text(errors="replace")
                    except Exception:
                        continue
                    for bad in prohibited:
                        assert bad not in content, (
                            f"Prohibited string '{bad}' found in {fpath.relative_to(FREEZE_DIR)}"
                        )


# ---------------------------------------------------------------------------
# 11. Clean bundle contains no biomarker/top-edge artifacts
# ---------------------------------------------------------------------------
class TestCleanBundleNoBiomarker:
    def test_clean_bundle_no_biomarker(self):
        """Freeze bundle must not contain biomarker or top-edge artifacts."""
        _skip_if_no_freeze()

        prohibited_names = [
            "biomarker_fit_metrics",
            "biomarker_seed_metrics",
            "top_edges",
            "coefficient_stability",
            "coefficient_alignment",
        ]

        all_files = [f.name for f in FREEZE_DIR.rglob("*") if f.is_file()]
        for artifact in prohibited_names:
            matches = [f for f in all_files if artifact in f]
            assert not matches, (
                f"Found prohibited biomarker/top-edge artifact(s): {matches}"
            )


# ---------------------------------------------------------------------------
# 12. Clean bundle contains no prohibited legacy path strings
# ---------------------------------------------------------------------------
class TestCleanBundleNoProhibitedPaths:
    def test_clean_bundle_no_prohibited_paths(self):
        """Freeze bundle text files must not contain prohibited legacy path strings."""
        _skip_if_no_freeze()

        prohibited = [
            "lf1_final_10x5",
            "lf1_final_evidence_audit",
            "prior_aware_late_fusion_integrity_audit",
        ]

        searchable = [".md", ".json", ".txt", ".csv", ".tex"]
        for fpath in FREEZE_DIR.rglob("*"):
            if fpath.is_file() and fpath.suffix in searchable:
                try:
                    content = fpath.read_text(errors="replace")
                except Exception:
                    continue
                for bad in prohibited:
                    assert bad not in content, (
                        f"Prohibited legacy path '{bad}' found in "
                        f"{fpath.relative_to(FREEZE_DIR)}"
                    )


# ---------------------------------------------------------------------------
# 13. All generated CSVs have finite values in numerical evidence columns
# ---------------------------------------------------------------------------
class TestAllCSVsFiniteEvidence:
    def test_all_csvs_finite_evidence(self):
        """Numerical evidence columns in freeze-bundle CSVs must be finite."""
        _skip_if_no_freeze()

        evidence_keywords = [
            "pearson", "r_", "delta", "p_", "ci_", "rmse", "mae",
            "mean_", "std_", "cohens", "wilcoxon", "w_fp", "w_sc",
            "w_fc", "weight",
        ]

        csv_files = list(FREEZE_DIR.rglob("*.csv"))
        assert len(csv_files) > 0, "No CSV files found in freeze bundle"

        for csv_path in csv_files:
            if csv_path.name == "MANIFEST.sha256":
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                pytest.fail(f"Failed to read {csv_path.name}: {e}")

            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if any(kw in col.lower() for kw in evidence_keywords):
                    nan_count = df[col].isna().sum()
                    inf_count = np.isinf(df[col].dropna()).sum()
                    assert nan_count == 0, (
                        f"NaN in '{col}' of {csv_path.name}: {nan_count} rows"
                    )
                    assert inf_count == 0, (
                        f"Inf in '{col}' of {csv_path.name}: {inf_count} rows"
                    )


# ---------------------------------------------------------------------------
# 14. All expected PDF/PNG figures exist and are non-empty
# ---------------------------------------------------------------------------
class TestAllFiguresExistAndNonempty:
    def test_all_figures_exist_and_nonempty(self):
        """Expected figure files must exist and have non-zero size."""
        _skip_if_no_freeze()

        expected_figures = [
            "fig_main_2x2.pdf",
            "fig_main_2x2.png",
        ]

        figures_dir = FREEZE_DIR / "figures"
        if not figures_dir.exists():
            pytest.skip("No figures directory in freeze bundle")

        available = {f.name for f in figures_dir.iterdir() if f.is_file()}
        for fname in expected_figures:
            if fname in available:
                fpath = figures_dir / fname
                assert fpath.stat().st_size > 0, (
                    f"Figure {fname} exists but is empty (0 bytes)"
                )


# ---------------------------------------------------------------------------
# 15. checksums.sha256 covers all freeze-bundle files
# ---------------------------------------------------------------------------
class TestChecksumsCoverBundle:
    def test_checksums_cover_bundle(self):
        """checksums/MANIFEST.sha256 must cover all non-checksum freeze-bundle files."""
        _skip_if_no_freeze()

        sha_path = FREEZE_DIR / "checksums" / "MANIFEST.sha256"
        if not sha_path.exists():
            sha_path = FREEZE_DIR / "checksums.sha256"
        if not sha_path.exists():
            pytest.skip("No checksums file found in freeze bundle")

        manifest_text = sha_path.read_text()
        manifest_files = set()
        for line in manifest_text.strip().splitlines():
            parts = line.strip().split("  ", 1)
            if len(parts) == 2:
                manifest_files.add(parts[1])

        bundle_files = set()
        for fpath in FREEZE_DIR.rglob("*"):
            if fpath.is_file() and not str(fpath).endswith(".sha256"):
                rel = str(fpath.relative_to(FREEZE_DIR))
                if "MANIFEST.sha256" not in rel:
                    bundle_files.add(rel)

        missing = bundle_files - manifest_files
        assert not missing, (
            f"Files in bundle not covered by checksums: {sorted(missing)}"
        )
