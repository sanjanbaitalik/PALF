"""Phase 3A-FIX: corrected R0 baseline wrappers.

Wraps the validated same-solver FP+SC cross-fitted fusion procedure from
``metascfc.experiments.palf_crossfit_ablation`` with caching and an access log.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    N_ROI,
    generate_crossfit_oof,
    reselect_and_fit_final,
    search_fusion_weights,
)

logger = logging.getLogger(__name__)

RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _idx_hash(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


class AccessLogger:
    """Records every subject-ID batch requested during Phase 3A-FIX."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: List[dict] = []
        self._holdout: set = set()

    def set_holdout(self, holdout_ids):
        self._holdout = set(str(x) for x in holdout_ids)

    def log(self, purpose: str, subject_ids) -> None:
        ids = [str(x) for x in subject_ids]
        overlap = sorted(set(ids) & self._holdout)
        rec = {"purpose": purpose, "n": len(ids), "subject_ids": ids,
               "holdout_overlap": overlap}
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if overlap:
            raise ValueError(f"HOLDOUT ACCESS VIOLATION in {purpose}: {overlap}")

    def assert_no_holdout(self) -> None:
        bad = [r for r in self.records if r["holdout_overlap"]]
        if bad:
            raise ValueError(f"Holdout IDs appeared in access log: {bad[:3]}")


class R0Baseline:
    """Corrected R0 baseline with caching.

    R0 = generalized same-solver no-prior FP + SC Ridge + cross-fitted convex
    FP+SC fusion (validated procedure in palf_crossfit_ablation).
    """

    def __init__(
        self,
        X_fc: np.ndarray,
        X_sc: np.ndarray,
        y_wm: np.ndarray,
        y_fi: np.ndarray,
        subject_ids: np.ndarray,
        access_logger: Optional[AccessLogger] = None,
    ):
        assert X_fc.ndim == 2 and X_fc.shape[1] == 6670, X_fc.shape
        assert X_sc.shape == X_fc.shape, (X_sc.shape, X_fc.shape)
        n = X_fc.shape[0]
        assert y_wm.shape == (n,) and y_fi.shape == (n,), (y_wm.shape, y_fi.shape)
        self.X_fc = X_fc
        self.X_sc = X_sc
        self.y = {"WM": y_wm.astype(np.float64), "FI": y_fi.astype(np.float64)}
        self.subject_ids = np.asarray(subject_ids).astype(str)
        self.access = access_logger
        self._cache: Dict[tuple, np.ndarray] = {}
        self._condition = CONDITIONS["R0"]
        self._prior = np.ones(N_ROI) / N_ROI

    # ── caching ─────────────────────────────────────────────────────────
    def _cache_key(self, *parts) -> tuple:
        return parts

    def _log(self, purpose: str, idx: np.ndarray) -> None:
        if self.access is not None:
            self.access.log(purpose, self.subject_ids[np.asarray(idx, dtype=int)])

    # ── 12A. fit_r0_predict ─────────────────────────────────────────────
    def fit_r0_predict(
        self,
        task: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        seed: int,
        outer_fold: int,
        cache_tag: str = "",
    ) -> np.ndarray:
        """Fit corrected R0 on train_idx, predict val_idx (fused FP+SC)."""
        key = ("fit", task, _idx_hash(train_idx), _idx_hash(val_idx),
               int(seed), int(outer_fold), cache_tag)
        if key in self._cache:
            return self._cache[key].copy()

        self._log(f"fit_r0_predict[{task}][{cache_tag}]", train_idx)
        self._log(f"fit_r0_predict[{task}][{cache_tag}]", val_idx)

        y = self.y[task]
        oof = generate_crossfit_oof(
            self.X_fc, self.X_sc, y, train_idx, self._condition, self._prior,
            int(seed), int(outer_fold), ridge_grid=RIDGE_GRID,
            n_fusion_folds=3, n_inner=3, n_rois=N_ROI,
        )
        weights, _ = search_fusion_weights(
            y[train_idx], {"FP": oof.fp_oof, "SC": oof.sc_oof}, ["FP", "SC"],
        )
        _, sc_final, fp_final = reselect_and_fit_final(
            self.X_fc, self.X_sc, y, train_idx, val_idx, self._condition,
            self._prior, int(seed), int(outer_fold), ridge_grid=RIDGE_GRID,
            n_final_cv=3, n_rois=N_ROI,
        )
        pred = weights["FP"] * fp_final.test_pred + weights["SC"] * sc_final.test_pred
        self._cache[key] = pred.copy()
        return pred

    # ── 12B. crossfit_r0_within ─────────────────────────────────────────
    def _partition(self, scope_idx: np.ndarray, seed: int, tag: str,
                   n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Deterministic partition of scope_idx into n_folds (B, C) pairs."""
        scope_idx = np.asarray(scope_idx, dtype=int)
        h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        perm = rng.permutation(len(scope_idx))
        sizes = np.full(n_folds, len(scope_idx) // n_folds)
        sizes[: len(scope_idx) % n_folds] += 1
        folds = []
        cur = 0
        for k in range(n_folds):
            start, stop = cur, cur + sizes[k]
            c_local = perm[start:stop]
            b_local = np.concatenate([perm[:start], perm[stop:]])
            folds.append((scope_idx[b_local], scope_idx[c_local]))
            cur = stop
        return folds

    def crossfit_r0_within(
        self,
        task: str,
        scope_idx: np.ndarray,
        seed: int,
        tag: str,
        n_folds: int = 3,
    ) -> np.ndarray:
        """Leakage-safe R0 predictions for every subject in scope_idx.

        Each subject's prediction comes from an R0 model that did not train on it.
        Returns an array aligned with scope_idx order.
        """
        scope_idx = np.asarray(scope_idx, dtype=int)
        key = ("crossfit", task, _idx_hash(scope_idx), int(seed), tag, n_folds)
        if key in self._cache:
            return self._cache[key].copy()

        folds = self._partition(scope_idx, seed, tag, n_folds)
        pos = {int(g): i for i, g in enumerate(scope_idx)}
        out = np.full(len(scope_idx), np.nan)
        for fi, (B, C) in enumerate(folds):
            pred = self.fit_r0_predict(
                task, B, C, seed, 700 + fi, cache_tag=f"{tag}_cf{fi}"
            )
            for g, v in zip(C, pred):
                out[pos[int(g)]] = v
        assert np.all(np.isfinite(out)), "crossfit_r0_within left NaNs"
        self._cache[key] = out.copy()
        return out
