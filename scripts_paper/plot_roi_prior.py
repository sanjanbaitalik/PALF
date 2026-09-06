#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import plotting


def pick_column(
    df,
    names,
    required=True,
):

    for name in names:
        if name in df.columns:
            return name

    if required:
        raise KeyError(
            f"None of {names} found.\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atlas",
        required=True,
    )

    parser.add_argument(
        "--prior",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--title",
        default="",
    )

    parser.add_argument(
        "--index-column",
        default=None,
    )

    parser.add_argument(
        "--score-column",
        default=None,
    )

    parser.add_argument(
        "--index-base",
        choices=[
            "auto",
            "0",
            "1",
        ],
        default="auto",
    )

    args = parser.parse_args()

    atlas_path = Path(
        args.atlas
    )

    prior_path = Path(
        args.prior
    )

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas not found: {atlas_path}"
        )

    if not prior_path.exists():
        raise FileNotFoundError(
            f"Prior CSV not found: {prior_path}"
        )

    # ---------------------------------------------------------
    # Atlas
    # ---------------------------------------------------------

    atlas_img = nib.load(
        str(atlas_path)
    )

    atlas = np.asarray(
        atlas_img.dataobj
    )

    atlas_labels = np.unique(
        atlas[
            np.isfinite(atlas)
        ]
    )

    atlas_labels = (
        atlas_labels[
            atlas_labels > 0
        ]
        .astype(int)
    )

    print(
        "Atlas labels:"
    )

    print(
        f"  min   = {atlas_labels.min()}"
    )

    print(
        f"  max   = {atlas_labels.max()}"
    )

    print(
        f"  count = {len(atlas_labels)}"
    )

    # ---------------------------------------------------------
    # Prior
    # ---------------------------------------------------------

    df = pd.read_csv(
        prior_path
    )

    print(
        f"\nPrior CSV columns: "
        f"{list(df.columns)}"
    )

    score_col = (
        args.score_column
        or pick_column(
            df,
            [
                "score",
                "prior",
                "prior_score",
                "weight",
                "value",
                "relevance",
            ],
        )
    )

    index_col = (
        args.index_column
        or pick_column(
            df,
            [
                "roi_index",
                "index",
                "aal_index",
                "roi_id",
                "label_id",
                "roi",
            ],
            required=False,
        )
    )

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce",
    )

    if index_col is None:

        if len(df) != 116:
            raise ValueError(
                "No ROI index column found, and "
                f"the prior has {len(df)} rows "
                "rather than 116.\n"
                "Specify --index-column."
            )

        indices = np.arange(
            1,
            117,
        )

        print(
            "\nNo ROI index column found."
        )

        print(
            "Assuming CSV row order "
            "maps to AAL labels 1..116."
        )

    else:

        indices = pd.to_numeric(
            df[index_col],
            errors="coerce",
        ).to_numpy()

    valid = (
        np.isfinite(indices)
        & np.isfinite(
            scores.to_numpy()
        )
    )

    indices = (
        indices[valid]
        .astype(int)
    )

    scores = (
        scores
        .to_numpy()[valid]
        .astype(float)
    )

    # ---------------------------------------------------------
    # Handle 0-based CSV indices
    # ---------------------------------------------------------

    if args.index_base == "0":

        indices = (
            indices + 1
        )

    elif args.index_base == "auto":

        if (
            len(indices)
            and indices.min() == 0
            and indices.max() <= 115
            and atlas_labels.min() == 1
        ):

            indices = (
                indices + 1
            )

            print(
                "\nDetected 0-based "
                "ROI indices."
            )

            print(
                "Shifted them to "
                "AAL labels 1..116."
            )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    missing = sorted(
        set(indices)
        - set(atlas_labels)
    )

    if missing:
        raise ValueError(
            "Prior contains ROI labels "
            "not present in atlas:\n"
            f"{missing[:20]}"
        )

    print(
        f"\nLoaded {len(scores)} "
        "valid ROI scores."
    )

    print(
        f"score min  = {scores.min():.6f}"
    )

    print(
        f"score max  = {scores.max():.6f}"
    )

    print(
        f"score mean = {scores.mean():.6f}"
    )

    # ---------------------------------------------------------
    # Build statistical map
    # ---------------------------------------------------------

    stat = np.zeros_like(
        atlas,
        dtype=np.float32,
    )

    for idx, score in zip(
        indices,
        scores,
    ):

        stat[
            atlas == idx
        ] = score

    stat_img = nib.Nifti1Image(
        stat,
        atlas_img.affine,
        atlas_img.header,
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = plotting.plot_stat_map(
        stat_img,
        display_mode="z",
        cut_coords=7,
        threshold=0,
        colorbar=True,
    )

    display.savefig(
        str(output)
    )

    display.close()

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()