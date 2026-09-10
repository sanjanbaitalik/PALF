"""
LLM network-interaction prior generation.

Generates system-pair structure-function coupling relevance scores
using contrastive LLM prompting. No HCP data is used.
"""
import json
import hashlib
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

SYSTEM_NAMES = [
    "VIS", "SM", "DATN", "LIMBIC", "FPN", "DMN", "VAS",
    "SUBCORTICAL", "CEREBELLAR"
]

SYSTEM_DESCRIPTIONS = {
    "VIS": "Visual cortex (occipital lobe): primary and association visual areas",
    "SM": "Somatomotor cortex (pre/postcentral gyrus, supplementary motor area)",
    "DATN": "Dorsal attention network (intraparietal sulcus, frontal eye fields)",
    "LIMBIC": "Limbic/ventral attention (insula, anterior cingulate, frontal operculum)",
    "FPN": "Frontoparietal control network (DLPFC, inferior parietal, lateral temporal)",
    "DMN": "Default mode network (mPFC, PCC/precuneus, angular gyrus, hippocampal formation)",
    "VAS": "Ventral attention/salience network (TPJ, ventral frontal, temporal pole)",
    "SUBCORTICAL": "Subcortical structures (thalamus, basal ganglia, hippocampus, amygdala)",
    "CEREBELLAR": "Cerebellum and vermis",
}

WM_PROMPT_TEMPLATE = """You are a cognitive neuroscientist predicting individual differences in human cognition.

CONTEXT:
- We have resting-state functional connectivity (FC) and diffusion-based structural connectivity (SC) brain data from 412 healthy adults.
- Each edge in the connectome represents the strength of structural or functional coupling between two brain regions.
- We want to predict BETWEEN-SUBJECT differences in Working Memory (NIH List Sorting task performance).
- Working Memory specifically involves the ACTIVE MAINTENANCE and MANIPULATION of information over short delays.

TASK:
Score the importance of STRUCTURE-FUNCTION COUPLING INTERACTION between each pair of brain systems
for predicting individual differences in Working Memory.

The 9 brain systems are:
{system_list}

For each UNORDERED system pair (including within-system pairs), provide:
1. sf_coupling_relevance: a score from 0.0 to 1.0 indicating how informative the STRUCTURAL-FUNCTIONAL
   coupling between these two systems is for individual differences in Working Memory.
2. reason_short: a brief justification (<= 25 words).

IMPORTANT RULES:
- High scores mean individual differences in the SC-FC RELATIONSHIP between these systems are informative.
- Generic cognitive involvement is NOT sufficient for a high score.
- Focus on systems whose SC-FC coupling DIFFERENTIATES individuals with better vs worse working memory.
- WM maintenance hubs: DLPFC, IPS, supplementary motor area, thalamus.
- Contrast against fluid intelligence / abstract reasoning: do NOT score general reasoning hubs highly
  unless they specifically support short-term maintenance.
- Working memory maintenance relies on frontoparietal loops (FPN-Subcortical coupling) and
  thalamo-cortical circuits.

Return a JSON array of objects with keys: system_a, system_b, sf_coupling_relevance, reason_short.
Return ONLY the JSON array, no other text."""

FI_PROMPT_TEMPLATE = """You are a cognitive neuroscientist predicting individual differences in human cognition.

CONTEXT:
- We have resting-state functional connectivity (FC) and diffusion-based structural connectivity (SC) brain data from 412 healthy adults.
- Each edge in the connectome represents the strength of structural or functional coupling between two brain regions.
- We want to predict BETWEEN-SUBJECT differences in Fluid Intelligence (Penn Matrix Reasoning task performance).
- Fluid Intelligence involves NOVEL problem-solving, ABSTRACT PATTERN DETECTION, and RULE INDUCTION.

TASK:
Score the importance of STRUCTURE-FUNCTION COUPLING INTERACTION between each pair of brain systems
for predicting individual differences in Fluid Intelligence.

The 9 brain systems are:
{system_list}

For each UNORDERED system pair (including within-system pairs), provide:
1. sf_coupling_relevance: a score from 0.0 to 1.0 indicating how informative the STRUCTURAL-FUNCTIONAL
   coupling between these two systems is for individual differences in Fluid Intelligence.
2. reason_short: a brief justification (<= 25 words).

IMPORTANT RULES:
- High scores mean individual differences in the SC-FC RELATIONSHIP between these systems are informative.
- Generic cognitive involvement is NOT sufficient for a high score.
- Focus on systems whose SC-FC coupling DIFFERENTIATES individuals with better vs worse abstract reasoning.
- Contrast against working memory / short-term maintenance: do NOT score maintenance-specific hubs highly
  unless they specifically support novel reasoning and rule discovery.
- Fluid intelligence relies on distributed frontoparietal-parietal circuits and cortico-cerebellar loops
  for abstract relational reasoning.
- Dorsal attention and ventral attention/salience systems support goal-directed reasoning selection.

Return a JSON array of objects with keys: system_a, system_b, sf_coupling_relevance, reason_short.
Return ONLY the JSON array, no other text."""


def _build_system_list_str() -> str:
    lines = []
    for name in SYSTEM_NAMES:
        lines.append(f"- {name}: {SYSTEM_DESCRIPTIONS[name]}")
    return "\n".join(lines)


def _call_ollama(prompt: str, model: str = "qwen3.8:27b", temperature: float = 0.2,
                  seed: int = 42) -> str:
    """Call Ollama API for a single generation."""
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "seed": seed},
        "stream": False,
    }
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return ""


def _parse_pair_scores(response_text: str) -> List[Dict]:
    """Parse LLM JSON response into pair scores."""
    text = response_text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        text = text.replace("'", '"')
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response:\n{response_text[:500]}")
            return []

    return data


def generate_pair_prior(
    task: str,
    model: str = "qwen3.8:27b",
    temperature: float = 0.2,
    seeds: List[int] = None,
    output_dir: str = "outputs/iclr/palf_phase2e_li_sfc_ncr/priors/llm_network_interaction",
) -> pd.DataFrame:
    """Generate LLM network-interaction pair prior for a task.

    Returns DataFrame with columns:
        system_a, system_b, pair_name, median_score, mean_score, std_score,
        min_score, max_score, normalized_score
    """
    if seeds is None:
        seeds = [31, 37, 43, 47, 53]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_list_str = _build_system_list_str()

    if task == "working_memory":
        prompt = WM_PROMPT_TEMPLATE.format(system_list=system_list_str)
    elif task == "fluid_intelligence":
        prompt = FI_PROMPT_TEMPLATE.format(system_list=system_list_str)
    else:
        raise ValueError(f"Unknown task: {task}")

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    # Generate all pairs
    all_pairs = []
    for i, s1 in enumerate(SYSTEM_NAMES):
        for j, s2 in enumerate(SYSTEM_NAMES):
            if j >= i:
                all_pairs.append((s1, s2))

    # Collect scores from multiple seeds
    seed_responses = {}
    all_scores = {pair: [] for pair in all_pairs}

    for seed in seeds:
        print(f"  Generating {task} prior with seed {seed}...")
        response = _call_ollama(prompt, model=model, temperature=temperature, seed=seed)

        if not response:
            print(f"  WARNING: Empty response for seed {seed}")
            continue

        seed_responses[str(seed)] = {
            "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            "response_preview": response[:200],
        }

        pair_data = _parse_pair_scores(response)
        for item in pair_data:
            sa = item.get("system_a", "")
            sb = item.get("system_b", "")
            score = float(item.get("sf_coupling_relevance", 0.0))
            key = (sa, sb) if (sa, sb) in all_pairs else (sb, sa)
            if key in all_scores:
                all_scores[key].append(score)

    # Aggregate
    records = []
    for (sa, sb) in all_pairs:
        scores = all_scores[(sa, sb)]
        pair_name = f"{sa}-{sb}" if sa != sb else sa

        if scores:
            median_s = float(np.median(scores))
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores))
            min_s = float(np.min(scores))
            max_s = float(np.max(scores))
        else:
            median_s = mean_s = std_s = min_s = max_s = 0.0

        records.append({
            "system_a": sa,
            "system_b": sb,
            "pair_name": pair_name,
            "n_seeds": len(scores),
            "median_score": median_s,
            "mean_score": mean_s,
            "std_score": std_s,
            "min_score": min_s,
            "max_score": max_s,
        })

    df = pd.DataFrame(records)

    # Normalize median to [0, 1]
    med_min = df["median_score"].min()
    med_max = df["median_score"].max()
    if med_max > med_min:
        df["normalized_score"] = (df["median_score"] - med_min) / (med_max - med_min)
    else:
        df["normalized_score"] = 0.5

    # Save
    df.to_csv(output_dir / f"{task}_pair_prior.csv", index=False)

    # Save raw responses
    raw_path = output_dir / f"{task}_raw_responses.json"
    with open(raw_path, "w") as f:
        json.dump(seed_responses, f, indent=2)

    # Save generation provenance
    provenance = {
        "task": task,
        "model": model,
        "temperature": temperature,
        "seeds": seeds,
        "prompt_hash": prompt_hash,
        "n_pairs": len(all_pairs),
        "n_seeds_collected": len([s for s in seeds if str(s) in seed_responses]),
    }
    with open(output_dir / f"{task}_generation_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    return df


def create_control_priors(
    matched_df: pd.DataFrame,
    output_dir: str,
    shuffled_seed: int = 7301,
    random_seed: int = 7303,
) -> Dict[str, pd.DataFrame]:
    """Create shuffled, random, and cross-task control priors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = {}

    # Shuffled: permute scores across pair labels
    rng = np.random.RandomState(shuffled_seed)
    shuffled_scores = matched_df["normalized_score"].values.copy()
    rng.shuffle(shuffled_scores)
    shuffled_df = matched_df.copy()
    shuffled_df["normalized_score"] = shuffled_scores
    shuffled_df.to_csv(output_dir / "shuffled_pair_prior.csv", index=False)
    controls["shuffled"] = shuffled_df

    # Random: fixed random [0,1] vector
    rng2 = np.random.RandomState(random_seed)
    random_scores = rng2.uniform(0, 1, size=len(matched_df))
    random_df = matched_df.copy()
    random_df["normalized_score"] = random_scores
    random_df.to_csv(output_dir / "random_pair_prior.csv", index=False)
    controls["random"] = random_df

    # Cross-task: swap WM and FI priors
    cross_df = matched_df.copy()
    # Will be filled in by the pilot script when both tasks are available
    controls["cross_task"] = cross_df

    return controls


def compute_prior_diagnostics(wm_df: pd.DataFrame, fi_df: pd.DataFrame) -> dict:
    """Compute prior-quality diagnostics before HCP evaluation."""
    wm_scores = wm_df["normalized_score"].values
    fi_scores = fi_df["normalized_score"].values

    # Pearson correlation
    pearson = float(np.corrcoef(wm_scores, fi_scores)[0, 1])

    # Spearman correlation
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(wm_scores, fi_scores)

    # Top-5 and top-10 overlap
    wm_top5 = set(np.argsort(wm_scores)[-5:])
    fi_top5 = set(np.argsort(fi_scores)[-5:])
    wm_top10 = set(np.argsort(wm_scores)[-10:])
    fi_top10 = set(np.argsort(fi_scores)[-10:])

    top5_overlap = len(wm_top5 & fi_top5)
    top10_overlap = len(wm_top10 & fi_top10)

    return {
        "wm_fi_pearson": pearson,
        "wm_fi_spearman": float(spearman_corr),
        "wm_fi_spearman_p": float(spearman_p),
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "wm_top5_pairs": [wm_df.iloc[i]["pair_name"] for i in np.argsort(wm_scores)[-5:]],
        "fi_top5_pairs": [fi_df.iloc[i]["pair_name"] for i in np.argsort(fi_scores)[-5:]],
    }
