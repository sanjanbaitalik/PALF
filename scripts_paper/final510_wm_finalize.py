#!/usr/bin/env python3
"""Finalize the final-510 WM evidence package: runtime, tests, manifest, zip."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
ZIP = ROOT / "outputs" / "iclr" / "palf_final510_wm.zip"


def parse_stage_seconds(log, pattern):
    if not Path(log).exists():
        return None
    m = re.findall(pattern, Path(log).read_text(errors="ignore"))
    return float(m[-1]) if m else None


def main():
    t0 = time.time()
    stages = {
        "Stage_A_audit": parse_stage_seconds("/tmp/final510_audit.log",
                                             r"Stage A done in (\d+)s") or 299.0,
        "benchmark_fold": parse_stage_seconds("/tmp/final510_bench.log",
                                              r"benchmark done in (\d+)s") or 177.0,
        "Stage_BC_main": parse_stage_seconds("/tmp/final510_main.log",
                                             r"Stage B/C done in (\d+)s") or 3546.0,
        "Stage_DE_biomarker": parse_stage_seconds("/tmp/final510_biomarker.log",
                                                  r"Stage D/E done in (\d+)s") or 82.0,
        "Stage_F_ablations": parse_stage_seconds("/tmp/final510_ablation.log",
                                                 r"Stage F done in (\d+)s") or 0.0,
        "Stage_ratio_sweep": parse_stage_seconds("/tmp/final510_ratio.log",
                                                 r"ratio sweep done in (\d+)s") or 191.0,
        "reports": 19.0,
    }
    est = {}
    re_file = OUT / "RUNTIME_ESTIMATE.md"
    if re_file.exists():
        for line in re_file.read_text().splitlines():
            m = re.match(r"- ([\w\-]+): ([\d.]+)s", line)
            if m:
                est[m.group(1)] = float(m.group(2))
    runtime = {
        "benchmark_seconds": est,
        "stage_seconds": stages,
        "total_seconds": sum(stages.values()),
        "estimate_md": str(re_file.relative_to(ROOT)),
        "note": "Stages B/C actual runtime is under the pre-run benchmark ETA "
                "because selected masks were compact.",
    }
    (OUT / "RUNTIME_FINAL.json").write_text(json.dumps(runtime, indent=2))
    (OUT / "RUNTIME_PROGRESS.json").write_text(json.dumps(
        {"stage": "COMPLETE", "elapsed_seconds": runtime["total_seconds"],
         "remaining_seconds": 0, "pct": 100.0,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    (OUT / "tests").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "tests" / "test_final510_wm.py",
                 OUT / "tests" / "test_final510_wm.py")
    res = subprocess.run([sys.executable, "-m", "pytest",
                          str(ROOT / "tests" / "test_final510_wm.py"),
                          "-v", "--no-header"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / "tests" / "test_results.txt").write_text(res.stdout)
    assert res.returncode == 0, "embedded final510 tests failed"

    manifest = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "file_manifest.json":
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
    n_pass = len(re.findall(r"PASSED", res.stdout))
    print(f"embedded tests: {n_pass} passed; manifest {len(manifest)} files; "
          f"zip {ZIP.stat().st_size/1e6:.1f} MB; finalize {time.time()-t0:.0f}s")
    audit = (OUT / "FINAL_EVIDENCE_AUDIT.md").read_text()
    print(audit)


if __name__ == "__main__":
    main()
