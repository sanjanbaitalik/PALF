#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import plotting


N_ROI = 116

N_EDGE = (
    N_ROI
    * (N_ROI - 1)
    // 2
)


def load_coefficients(
    path,
    npz_key=None,
    csv_column=None,
):

    path = Path(path)

    suffix = (
        path.suffix.lower()
    )

    # ---------------------------------------------------------
    # NPY
    # ---------------------------------------------------------

    if suffix == ".npy":
        return np.load(path)

    # ---------------------------------------------------------
    # NPZ
    # ---------------------------------------------------------

    if suffix == ".npz":

        archive = np.load(path)

        keys = list(
            archive.keys()
        )

        if npz_key is None:

            preferred = [
                "coefficients",
                "coef",
                "beta",
                "betas",
                "fc_coefficients",
            ]

            for key in preferred:

                if key in archive:

                    npz_key = key

                    break

        if npz_key is None:

            if len(keys) == 1:

                npz_key = keys[0]

            else:

                raise KeyError(
                    "NPZ contains multiple arrays:\n"
                    f"{keys}\n"
                    "Specify --npz-key."
                )

        print(
            f"Using NPZ key: "
            f"{npz_key}"
        )

        return archive[
            npz_key
        ]

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    if suffix == ".csv":

        df = pd.read_csv(path)

        print(
            f"CSV columns: "
            f"{list(df.columns)}"
        )

        if csv_column is None:

            preferred = [
                "coefficient",
                "coef",
                "beta",
                "weight",
                "value",
            ]

            for column in preferred:

                if column in df.columns:

                    csv_column = column

                    break

        if csv_column is not None:

            return (
                pd.to_numeric(
                    df[csv_column],
                    errors="coerce",
                )
                .dropna()
                .to_numpy()
            )

        numeric = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if (
            numeric
            .notna()
            .all()
            .all()
        ):

            return (
                numeric.to_numpy()
            )

        raise KeyError(
            "Could not infer a coefficient "
            "column.\n"
            f"Available columns: "
            f"{list(df.columns)}\n"
            "Specify --csv-column."
        )

    raise ValueError(
        "Unsupported coefficient "
        f"file type: {suffix}"
    )


def reduce_to_edge_vector(
    array,
    reduce,
):

    array = np.asarray(
        array
    )

    # ---------------------------------------------------------
    # One edge vector
    # ---------------------------------------------------------

    if (
        array.ndim == 1
        and array.size == N_EDGE
    ):

        return (
            array.astype(float)
        )

    # ---------------------------------------------------------
    # One 116 x 116 matrix
    # ---------------------------------------------------------

    if (
        array.ndim == 2
        and array.shape
        == (
            N_ROI,
            N_ROI,
        )
    ):

        iu = np.triu_indices(
            N_ROI,
            k=1,
        )

        return (
            array[iu]
            .astype(float)
        )

    # ---------------------------------------------------------
    # Collection of edge vectors
    # ---------------------------------------------------------

    if (
        array.ndim >= 2
        and array.shape[-1]
        == N_EDGE
    ):

        flat = array.reshape(
            -1,
            N_EDGE,
        )

        if reduce == "mean":

            return (
                flat.mean(
                    axis=0
                )
            )

        if reduce == "median":

            return np.median(
                flat,
                axis=0,
            )

        return flat[0]

    # ---------------------------------------------------------
    # Collection of matrices
    # ---------------------------------------------------------

    if (
        array.ndim >= 3
        and array.shape[-2:]
        == (
            N_ROI,
            N_ROI,
        )
    ):

        matrices = array.reshape(
            -1,
            N_ROI,
            N_ROI,
        )

        if reduce == "mean":

            matrix = (
                matrices.mean(
                    axis=0
                )
            )

        elif reduce == "median":

            matrix = np.median(
                matrices,
                axis=0,
            )

        else:

            matrix = matrices[0]

        iu = np.triu_indices(
            N_ROI,
            k=1,
        )

        return (
            matrix[iu]
            .astype(float)
        )

    raise ValueError(
        "Could not map coefficient "
        f"array of shape {array.shape} "
        f"to {N_EDGE} AAL116 FC edges."
    )


def edge_vector_to_matrix(
    vector,
):

    vector = np.asarray(
        vector
    ).reshape(-1)

    if vector.size != N_EDGE:

        raise ValueError(
            f"Expected {N_EDGE} edges, "
            f"got {vector.size}"
        )

    matrix = np.zeros(
        (
            N_ROI,
            N_ROI,
        ),
        dtype=float,
    )

    iu = np.triu_indices(
        N_ROI,
        k=1,
    )

    matrix[iu] = vector

    matrix[
        (
            iu[1],
            iu[0],
        )
    ] = vector

    return matrix


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atlas",
        required=True,
    )

    parser.add_argument(
        "--coeff",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--title",
        default="",
    )

    parser.add_argument(
        "--npz-key",
        default=None,
    )

    parser.add_argument(
        "--csv-column",
        default=None,
    )

    parser.add_argument(
        "--reduce",
        choices=[
            "mean",
            "median",
            "first",
        ],
        default="mean",
    )

    args = parser.parse_args()

    if not (
        1
        <= args.top_k
        <= N_EDGE
    ):

        raise ValueError(
            "--top-k must be "
            f"between 1 and {N_EDGE}"
        )

    atlas_path = Path(
        args.atlas
    )

    coeff_path = Path(
        args.coeff
    )

    if not atlas_path.exists():

        raise FileNotFoundError(
            f"Atlas not found: "
            f"{atlas_path}"
        )

    if not coeff_path.exists():

        raise FileNotFoundError(
            "Coefficient artifact "
            f"not found: {coeff_path}"
        )

    array = load_coefficients(
        coeff_path,
        npz_key=args.npz_key,
        csv_column=args.csv_column,
    )

    print(
        "\nLoaded coefficient "
        f"artifact shape: "
        f"{np.asarray(array).shape}"
    )

    coefficients = (
        reduce_to_edge_vector(
            array,
            args.reduce,
        )
    )

    if not np.isfinite(
        coefficients
    ).all():

        raise ValueError(
            "Coefficient vector "
            "contains NaN or infinity."
        )

    print(
        f"Reduced to "
        f"{len(coefficients)} FC edges."
    )

    print(
        f"Coefficient min = "
        f"{coefficients.min():+.6g}"
    )

    print(
        f"Coefficient max = "
        f"{coefficients.max():+.6g}"
    )

    # ---------------------------------------------------------
    # Select top absolute edges
    # ---------------------------------------------------------

    order = np.argsort(
        np.abs(coefficients)
    )[::-1]

    top_indices = order[
        : args.top_k
    ]

    sparse = np.zeros_like(
        coefficients
    )

    sparse[
        top_indices
    ] = coefficients[
        top_indices
    ]

    matrix = (
        edge_vector_to_matrix(
            sparse
        )
    )

    # ---------------------------------------------------------
    # Atlas coordinates
    # ---------------------------------------------------------

    atlas_img = nib.load(
        str(atlas_path)
    )

    coordinates = (
        plotting
        .find_parcellation_cut_coords(
            atlas_img
        )
    )

    if (
        coordinates.shape[0]
        != N_ROI
    ):

        raise ValueError(
            "Atlas yielded "
            f"{coordinates.shape[0]} "
            "parcel coordinates; "
            f"expected {N_ROI}."
        )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = (
        plotting.plot_connectome(
            matrix,
            coordinates,
            edge_threshold=None,
            node_size=8,
            title=args.title,
        )
    )

    display.savefig(
        str(output)
    )

    display.close()

    # ---------------------------------------------------------
    # Save exact edge indices
    # ---------------------------------------------------------

    iu = np.triu_indices(
        N_ROI,
        k=1,
    )

    rows = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        rows.append(
            {
                "rank": rank,

                "edge_vector_index":
                    int(index),

                "roi_i_1based":
                    int(
                        iu[0][index]
                        + 1
                    ),

                "roi_j_1based":
                    int(
                        iu[1][index]
                        + 1
                    ),

                "coefficient":
                    float(
                        coefficients[
                            index
                        ]
                    ),

                "abs_coefficient":
                    float(
                        abs(
                            coefficients[
                                index
                            ]
                        )
                    ),
            }
        )

    sidecar = (
        output.with_suffix(
            ".top_edges.tsv"
        )
    )

    pd.DataFrame(
        rows
    ).to_csv(
        sidecar,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved figure: {output}"
    )

    print(
        f"Saved edge list: {sidecar}"
    )


if __name__ == "__main__":
    main()