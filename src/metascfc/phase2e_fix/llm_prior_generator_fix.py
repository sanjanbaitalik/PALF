"""LLM pair-prior generation via Ollama REST API with proper seed/temperature."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
SYSTEM_NAMES = ["VIS", "SM", "DAN", "VAN", "LIMBIC", "FPN", "DMN", "SUBCORTICAL", "CEREBELLAR"]
SEEDS = [31, 37, 43, 47, 53]
TEMPERATURE = 0.2
MAX_RETRIES = 3
TIMEOUT = 300


def build_all_pairs() -> List[Tuple[str, str]]:
    """Return all 45 unordered system pairs."""
    pairs = []
    for i, s1 in enumerate(SYSTEM_NAMES):
        for j, s2 in enumerate(SYSTEM_NAMES):
            if j >= i:
                pairs.append((s1, s2))
    assert len(pairs) == 45, f"Expected 45 pairs, got {len(pairs)}"
    return pairs


def pair_key(s1: str, s2: str) -> str:
    """Canonical pair key: smaller alphabetically first."""
    a, b = sorted([s1, s2])
    return f"{a}-{b}" if a != b else a


def build_prompt(task: str) -> str:
    """Build the LLM prompt for pair-prior generation."""
    if task == "working_memory":
        target_desc = (
            "individual differences in working-memory capacity / maintenance and manipulation"
        )
        question_desc = (
            "which large-scale brain-system STRUCTURE–FUNCTION COUPLING relationships are "
            "expected to be most informative for between-person variation in this target, "
            "relative to fluid reasoning?"
        )
    elif task == "fluid_intelligence":
        target_desc = (
            "individual differences in fluid reasoning / novel problem solving and rule induction"
        )
        question_desc = (
            "which large-scale brain-system STRUCTURE–FUNCTION COUPLING relationships are "
            "expected to be most informative for between-person variation in this target, "
            "relative to working-memory-specific maintenance?"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    pairs_text = ", ".join(
        [f"{a}-{b}" if a != b else a for a, b in build_all_pairs()]
    )

    prompt = (
        f"/no_think\n"
        f"Target:\n{target_desc}\n\n"
        f"Question:\n{question_desc}\n\n"
        f"You are predicting individual differences in brain function from brain structure. "
        f"The features are SC-FC coupling: for each edge, how well does structural connectivity "
        f"predict functional connectivity across individuals.\n\n"
        f"Systems: {', '.join(SYSTEM_NAMES)}\n\n"
        f"For every one of the 45 unordered pairs return:\n"
        f"  system_a: <name>\n"
        f"  system_b: <name>\n"
        f"  sf_coupling_relevance: <0.0-1.0>\n"
        f"  reason_short: <<=25 words>\n\n"
        f"Return a JSON array of exactly 45 objects. No coefficient direction. "
        f"No subject-specific prediction.\n"
        f"Pairs: {pairs_text}\n\n"
        f"Return ONLY the JSON array."
    )
    return prompt


def call_ollama_rest(
    prompt: str,
    model: str = "qwen3.8:27b",
    temperature: float = TEMPERATURE,
    seed: int = 42,
) -> Dict[str, Any]:
    """Call Ollama REST API with proper seed/temperature."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama API not available at " + OLLAMA_API_URL)
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def parse_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON response, handling thinking tokens and formatting issues."""
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", response_text)

    # Try to extract JSON array
    start = cleaned.find("[")
    if start < 0:
        raise ValueError("No JSON array found in response")

    # Find matching ]
    depth = 0
    end = start
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
        if depth == 0:
            end = i + 1
            break
    else:
        raise ValueError("Unmatched brackets in response")

    json_str = cleaned[start:end]

    # Fix missing commas between objects
    json_str = re.sub(r"\}\s*\{", "},{", json_str)

    # Fix newlines inside strings
    in_str = False
    chars = []
    for c in json_str:
        if c == '"' and (not chars or chars[-1] != "\\"):
            in_str = not in_str
        if in_str and c in ("\n", "\r"):
            chars.append(" ")
        else:
            chars.append(c)
    json_str = "".join(chars)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try removing trailing commas
        json_str = re.sub(r",\s*\]", "]", json_str)
        json_str = re.sub(r",\s*\}", "}", json_str)
        data = json.loads(json_str)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")

    return data


def validate_generation(
    data: List[Dict[str, Any]],
    expected_pairs: List[Tuple[str, str]],
) -> Tuple[bool, str]:
    """Validate a single LLM generation against expected pairs."""
    if len(data) != 45:
        return False, f"Expected 45 pairs, got {len(data)}"

    seen = set()
    for item in data:
        sa = item.get("system_a", "")
        sb = item.get("system_b", "")
        score = item.get("sf_coupling_relevance", None)

        if score is None:
            return False, f"Missing sf_coupling_relevance for {sa}-{sb}"

        try:
            score_f = float(score)
        except (ValueError, TypeError):
            return False, f"Non-numeric score for {sa}-{sb}: {score}"

        if not (0 <= score_f <= 1):
            return False, f"Score out of [0,1] for {sa}-{sb}: {score_f}"

        pk = pair_key(sa, sb)
        if pk in seen:
            return False, f"Duplicate pair: {pk}"
        seen.add(pk)

    # Check all expected pairs present
    for a, b in expected_pairs:
        pk = pair_key(a, b)
        if pk not in seen:
            return False, f"Missing pair: {pk}"

    return True, "OK"


def generate_prior_for_task(
    task: str,
    model: str = "qwen3.8:27b",
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate genuine LLM pair priors for a single task.

    Returns (pair_df, provenance_dict).
    """
    expected_pairs = build_all_pairs()
    prompt = build_prompt(task)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    print(f"  Generating {task} priors...")
    print(f"  Prompt SHA256: {prompt_hash}")
    print(f"  Model: {model}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Temperature: {TEMPERATURE}")

    # Collect raw responses
    raw_responses = {}
    pair_scores = {pair_key(a, b): [] for a, b in expected_pairs}
    raw_response_hashes = {}
    seed_status = {}

    for seed in SEEDS:
        success = False
        for attempt in range(MAX_RETRIES):
            print(f"    Seed {seed} attempt {attempt + 1}...", end=" ", flush=True)
            t0 = time.time()

            try:
                resp_data = call_ollama_rest(prompt, model=model, temperature=TEMPERATURE, seed=seed)
                response_text = resp_data.get("response", "")
                elapsed = time.time() - t0

                # Save raw response
                raw_key = f"{task}_raw_seed{seed}"
                raw_responses[raw_key] = response_text
                raw_response_hashes[raw_key] = hashlib.sha256(response_text.encode()).hexdigest()

                # Parse
                data = parse_llm_response(response_text)
                valid, msg = validate_generation(data, expected_pairs)

                if valid:
                    # Extract scores
                    for item in data:
                        sa = item.get("system_a", "")
                        sb = item.get("system_b", "")
                        score = float(item.get("sf_coupling_relevance", 0.0))
                        pk = pair_key(sa, sb)
                        pair_scores[pk].append(score)

                    print(f"{elapsed:.0f}s OK ({len(data)} pairs)")
                    success = True
                    seed_status[seed] = "OK"
                    break
                else:
                    print(f"{elapsed:.0f}s INVALID: {msg}")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"{elapsed:.0f}s ERROR: {e}")

        if not success:
            raise RuntimeError(
                f"Failed to generate valid prior for {task} seed {seed} "
                f"after {MAX_RETRIES} attempts"
            )

    # Aggregate
    records = []
    for a, b in expected_pairs:
        pk = pair_key(a, b)
        scores = pair_scores[pk]
        assert len(scores) == 5, f"Expected 5 scores for {pk}, got {len(scores)}"
        records.append({
            "system_a": a,
            "system_b": b,
            "pair_name": pk,
            "n_seeds": len(scores),
            "median_score": float(np.median(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        })

    df = pd.DataFrame(records)

    # Normalize to [0, 1]
    med_min, med_max = df["median_score"].min(), df["median_score"].max()
    if med_max > med_min:
        df["normalized_score"] = (df["median_score"] - med_min) / (med_max - med_min)
    else:
        df["normalized_score"] = 0.5

    # Provenance
    provenance = {
        "task": task,
        "model": model,
        "model_digest": "",
        "temperature": TEMPERATURE,
        "seeds": SEEDS,
        "prompt_sha256": prompt_hash,
        "raw_response_hashes": raw_response_hashes,
        "n_seeds_collected": 5,
        "n_pairs": 45,
        "validation_status": "45/45",
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_status": seed_status,
    }

    # Try to get model digest
    try:
        tags_resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        tags_data = tags_resp.json()
        for m in tags_data.get("models", []):
            if m.get("name") == model:
                provenance["model_digest"] = m.get("digest", "")
                break
    except Exception:
        pass

    return df, provenance, raw_responses


def create_control_priors(
    wm_df: pd.DataFrame,
    fi_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Create task-specific control priors."""
    controls = {}

    # Task-specific shuffled
    for task_name, task_df in [("working_memory", wm_df), ("fluid_intelligence", fi_df)]:
        seed = 7301 if task_name == "working_memory" else 7302
        shuffled = task_df.copy()
        rng = np.random.RandomState(seed)
        vals = shuffled["normalized_score"].values.copy()
        rng.shuffle(vals)
        shuffled["normalized_score"] = vals
        controls[f"{task_name}_shuffled"] = shuffled

    # Task-specific random
    for task_name in ["working_memory", "fluid_intelligence"]:
        seed = 7303 if task_name == "working_memory" else 7304
        rng = np.random.RandomState(seed)
        random_df = wm_df.copy()  # structure only
        random_df["normalized_score"] = rng.uniform(0, 1, len(random_df))
        controls[f"{task_name}_random"] = random_df

    # Cross-task
    controls["cross_task_wm"] = fi_df.copy()  # WM uses FI's prior
    controls["cross_task_fi"] = wm_df.copy()  # FI uses WM's prior

    return controls


def compute_prior_diagnostics(wm_df: pd.DataFrame, fi_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute prior diagnostics."""
    wm_scores = wm_df["normalized_score"].values
    fi_scores = fi_df["normalized_score"].values

    pearson_corr = float(np.corrcoef(wm_scores, fi_scores)[0, 1])
    spearman_corr = float(pd.Series(wm_scores).corr(pd.Series(fi_scores), method="spearman"))

    wm_top5 = set(wm_df.nlargest(5, "normalized_score").index)
    fi_top5 = set(fi_df.nlargest(5, "normalized_score").index)
    wm_top10 = set(wm_df.nlargest(10, "normalized_score").index)
    fi_top10 = set(fi_df.nlargest(10, "normalized_score").index)

    return {
        "wm_fi_pearson": pearson_corr,
        "wm_fi_spearman": spearman_corr,
        "top5_overlap": len(wm_top5 & fi_top5),
        "top10_overlap": len(wm_top10 & fi_top10),
        "wm_top10": wm_df.nlargest(10, "normalized_score")[["pair_name", "normalized_score"]].to_dict("records"),
        "fi_top10": fi_df.nlargest(10, "normalized_score")[["pair_name", "normalized_score"]].to_dict("records"),
    }


def freeze_priors(output_dir: Path, all_files: List[Path]) -> None:
    """Create FROZEN marker and SHA256SUMS.txt."""
    frozen_path = output_dir / "FROZEN"
    frozen_path.write_text(f"Frozen at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    sha256_path = output_dir / "SHA256SUMS.txt"
    lines = []
    for f in sorted(all_files):
        if f.is_file():
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{h}  {f.name}")
    sha256_path.write_text("\n".join(lines) + "\n")


def validate_frozen_priors(output_dir: Path) -> bool:
    """Validate FROZEN marker and checksums before loading priors."""
    frozen_path = output_dir / "FROZEN"
    if not frozen_path.exists():
        raise RuntimeError("FROZEN marker not found. Cannot load priors.")

    sha256_path = output_dir / "SHA256SUMS.txt"
    if not sha256_path.exists():
        raise RuntimeError("SHA256SUMS.txt not found.")

    for line in sha256_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        expected_hash, fname = line.split("  ", 1)
        fpath = output_dir / fname
        if not fpath.exists():
            raise RuntimeError(f"Missing file: {fname}")
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {fname}")

    return True
