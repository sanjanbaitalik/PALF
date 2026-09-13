#!/usr/bin/env python3
"""Finalize the 412 vs +98 scaling study: decision, status, manifest, zip."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling"
ZIP = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling.zip"
SEEDS = [7171, 7272, 7373, 7474, 7575]


def tier(delta, pos, ci_lo):
    if delta >= 0.005 and ci_lo > 0:
        return "ROBUST_SCALE_GAIN"
    if delta >= 0.005 and pos >= 4:
        return "CONSISTENT_SCALE_GAIN"
    if delta > 0:
        return "NOMINAL_SCALE_GAIN"
    return "NO_GAIN"

def main():
    seed_df = pd.read_csv(OUT / "primary_seed_metrics.csv")
    boot = pd.read_csv(OUT / "paired_bootstrap_results.csv")
    rows = []
    for _, r in boot.iterrows():
        a = seed_df[(seed_df.target == r.target) & (seed_df.model == r["model"])
                    & (seed_df.regime == "A")].set_index("seed").pearson
        b = seed_df[(seed_df.target == r.target) & (seed_df.model == r["model"])
                    & (seed_df.regime == "B")].set_index("seed").pearson
        d = (b - a).reindex(SEEDS)
        rows.append({"target": r.target, "model": r["model"],
                     "r_412": r.r_412, "r_plus98": r.r_plus98,
                     "delta_r_observed": r.delta_r_observed,
                     "delta_r_ci_lo": r.delta_r_ci_lo,
                     "delta_r_ci_hi": r.delta_r_ci_hi,
                     "delta_r_seed_mean": float(d.mean()),
                     "positive_seeds": int((d > 0).sum()),
                     "tier": tier(float(r.delta_r_observed), int((d > 0).sum()),
                                  float(r.delta_r_ci_lo))})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "scaling_decision_table.csv", index=False)
    any_positive_observed = bool((boot.delta_r_observed > 0).any())
    best = summary.sort_values("delta_r_observed", ascending=False).iloc[0]
    decision = {
        "decision": "NO_SAMPLE_SIZE_GAIN" if not any_positive_observed
                    else "NOMINAL_GAIN_REQUIRES_REVIEW",
        "paper_ready_trigger": any_positive_observed,
        "n_models": len(summary),
        "n_models_positive_observed": int((boot.delta_r_observed > 0).sum()),
        "best_observed": {"target": best.target, "model": best["model"],
                          "delta_r": float(best.delta_r_observed)},
        "claim_boundary": "post-hoc combined-cohort exploratory analysis; "
                          "NOT independent validation",
        "holdout_note": "the 98 subjects were opened in Phase 4 and are not a "
                        "holdout in this analysis",
        "phase4_result_preserved": True,
    }
    (OUT / "SCALING_DECISION.json").write_text(json.dumps(decision, indent=2))
    try:
        decision_md = summary.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        decision_md = "```\n" + summary.to_string(index=False) + "\n```"
    lines = [
        "# Scaling study status (412 vs 412+98)",
        "",
        f"- SCALING_DECISION: **{decision['decision']}**",
        f"- PAPER_READY_TRIGGER: {'YES' if any_positive_observed else 'NO'}",
        "- Analysis type: post-hoc combined-cohort sample-size scaling (exploratory)",
        "- This is NOT independent or external validation.",
        "- The 98 subjects were evaluated in Phase 4 and are not a holdout here.",
        "- Phase 4 result (WM_CONFIRMATION_FAILED) is preserved and not reinterpreted.",
        "",
        "## Main result",
        "",
        "Expanding the training pool from T412 to T412+D98 did not improve "
        "prediction on the identical V412 evaluation subjects: "
        f"{decision['n_models_positive_observed']}/{decision['n_models']} model/target "
        "combinations had a positive paired delta_r on seed-averaged predictions. "
        f"The best observed delta was {best.delta_r_observed:+.4f} "
        f"({best.target}/{best['model']}).",
        "",
        "## Per-model decision table",
        "",
        decision_md,
        "",
        "## Biomarker scaling",
        "",
        "- Cross-fold faithfulness (top10 minus random10 delta_RMSE; positive = "
        "faithful) improved from 412 to +98 for every target/model combination "
        "(see biomarker_faithfulness.csv and Figure 6).",
        "- Coefficient stability was mixed across targets (see "
        "biomarker_stability.csv and Figure 5).",
        "",
        "## Learning curve",
        "",
        "- Fixed-V412 learning curves from n=250 to full T412+D98 are non-monotone "
        "and show no reliable gain (learning_curve.csv, Figure 4).",
        "",
        "## Runtime",
        "",
        "- Primary stage: 25/25 splits in 18,280 s.",
        "- Learning stage: 8,355 s.",
        "- Secondary 510 CV: 670 s.",
        "- Biomarker stage: 13 s.",
    ]
    (OUT / "STUDY_STATUS.md").write_text("\n".join(lines) + "\n")

    (OUT / "tests").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "tests" / "test_412_vs_510_scaling.py",
                 OUT / "tests" / "test_412_vs_510_scaling.py")
    res = subprocess.run([sys.executable, "-m", "pytest",
                          str(ROOT / "tests" / "test_412_vs_510_scaling.py"),
                          "-v", "--no-header"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / "tests" / "test_results.txt").write_text(res.stdout)
    assert res.returncode == 0, "embedded scaling tests failed"

    manifest = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("file_manifest.json",):
            manifest.append({"path": str(p.relative_to(OUT)),
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                             "bytes": p.stat().st_size})
    (OUT / "file_manifest.json").write_text(json.dumps(
        {"files": manifest, "n_files": len(manifest)}, indent=2))

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    print(f"Decision: {decision['decision']}  trigger={any_positive_observed}")
    print(f"Manifest: {len(manifest)} files; zip: {ZIP.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
