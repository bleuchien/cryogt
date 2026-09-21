#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import spearmanr


MEMBER = "member"
SPLIT = "split"
TAXID = "ncbiTaxID_new"
TARGET = "Temp_Duplicate_Average"
BIN = "bin_name"

ALEATORIC = "aleatoric_std_"
EPISTEMIC = "epistemic_std_"
EPS = 1e-12


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={MEMBER: str, TAXID: str},
        low_memory=False,
    )


def resolve_path(base, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def short_model_name(model):
    match = re.search(r"_(\d+M)_.*_head_([^_]+)$", model)
    return f"{match.group(1)} {match.group(2)}" if match else model


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()


def load_split_predictions(split_file, prediction_file, wanted_split):
    """Use the split file as the authoritative source of metadata."""

    splits = read_csv(split_file)
    predictions = read_csv(prediction_file)

    required = [MEMBER, SPLIT, TAXID, TARGET, BIN]

    if any(column not in splits.columns for column in required):
        raise ValueError(f"Missing required columns in {split_file}")

    if splits[MEMBER].duplicated().any():
        raise ValueError(f"Duplicate members in {split_file}")

    metadata = splits[required].copy()
    metadata[SPLIT] = (
        metadata[SPLIT]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"validation": "val", "valid": "val"})
    )

    # Replace prediction-file metadata with split-file metadata.
    predictions = predictions.drop(
        columns=[SPLIT, TAXID, TARGET, BIN],
        errors="ignore",
    )

    merged = predictions.merge(
        metadata,
        on=MEMBER,
        how="left",
        validate="one_to_one",
    )

    if merged[SPLIT].isna().any():
        raise ValueError(
            f"Some predictions in {prediction_file} are absent "
            f"from the split file."
        )

    selected = merged.loc[
        merged[SPLIT] == wanted_split
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No {wanted_split} predictions found in {prediction_file}"
        )

    return selected.reset_index(drop=True)


def find_models(df):
    models = []

    for column in df.columns:
        if not column.startswith(ALEATORIC):
            continue

        model = column[len(ALEATORIC):]

        if (
            model in df.columns
            and f"{EPISTEMIC}{model}" in df.columns
        ):
            models.append(model)

    return sorted(models)


def is_psychrophile(df):
    return (
        df[BIN]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("psychrophiles")
        .to_numpy()
    )


def domain_masks(df):
    bins = (
        df[BIN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    psych = bins.eq("psychrophiles").to_numpy()
    meso = bins.str.contains("mesoph", na=False).to_numpy()

    # Includes thermophiles and hyperthermophiles.
    thermo = bins.str.contains("thermoph", na=False).to_numpy()

    return {
        "Psychrophiles": psych,
        "Mesophiles": meso,
        "Thermophiles+": thermo,
        "Non-psychrophiles": ~psych,
    }


def get_values(df, model):
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(float)
    mu = pd.to_numeric(df[model], errors="coerce").to_numpy(float)

    aleatoric = pd.to_numeric(
        df[f"{ALEATORIC}{model}"],
        errors="coerce",
    ).to_numpy(float)

    epistemic = pd.to_numeric(
        df[f"{EPISTEMIC}{model}"],
        errors="coerce",
    ).to_numpy(float)

    valid = (
        np.isfinite(y)
        & np.isfinite(mu)
        & np.isfinite(aleatoric)
        & np.isfinite(epistemic)
        & (aleatoric >= 0)
        & (epistemic >= 0)
        & ((aleatoric**2 + epistemic**2) > EPS)
    )

    return y, mu, aleatoric, epistemic, valid


def fit_component_scales(df, model, psych_weight=0.5):
    """
    Jointly fit aleatoric and epistemic multipliers by minimizing
    Gaussian NLL.

    psych_weight=0.5 means psychrophiles and non-psychrophiles each
    contribute 50% of the calibration loss, regardless of sequence
    counts.
    """

    y, mu, aleatoric, epistemic, valid = get_values(df, model)

    y = y[valid]
    mu = mu[valid]
    aleatoric = aleatoric[valid]
    epistemic = epistemic[valid]
    psych = is_psychrophile(df)[valid]

    if not psych.any():
        raise ValueError("No valid psychrophile validation predictions.")

    if psych.all():
        weights = np.full(len(y), 1.0 / len(y))
    else:
        weights = np.where(
            psych,
            psych_weight / psych.sum(),
            (1.0 - psych_weight) / (~psych).sum(),
        )

    residual_squared = (mu - y) ** 2
    raw_variance = aleatoric**2 + epistemic**2

    initial_scale = np.sqrt(
        np.sum(weights * residual_squared / raw_variance)
    )

    initial_scale = np.clip(initial_scale, 0.05, 20.0)
    initial = np.log([initial_scale, initial_scale])

    def loss(log_scales):
        scale_a, scale_e = np.exp(log_scales)

        variance = (
            scale_a**2 * aleatoric**2
            + scale_e**2 * epistemic**2
        )

        variance = np.maximum(variance, EPS)

        return 0.5 * np.sum(
            weights
            * (
                np.log(variance)
                + residual_squared / variance
            )
        )

    bounds = [
        (np.log(0.05), np.log(20.0)),
        (np.log(0.05), np.log(20.0)),
    ]

    result = minimize(
        loss,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not result.success:
        raise RuntimeError(
            f"Calibration failed for {model}: {result.message}"
        )

    scale_a, scale_e = np.exp(result.x)

    return (
        float(scale_a),
        float(scale_e),
        int(psych.sum()),
        int((~psych).sum()),
    )


def add_calibrated_columns(
    df,
    model,
    scale_a,
    scale_e,
    threshold=None,
):
    out = df.copy()

    raw_a = pd.to_numeric(
        out[f"{ALEATORIC}{model}"],
        errors="coerce",
    )

    raw_e = pd.to_numeric(
        out[f"{EPISTEMIC}{model}"],
        errors="coerce",
    )

    cal_a = scale_a * raw_a
    cal_e = scale_e * raw_e
    cal_var = cal_a**2 + cal_e**2
    cal_std = np.sqrt(cal_var)

    out[f"calibrated_aleatoric_std_{model}"] = cal_a
    out[f"calibrated_epistemic_std_{model}"] = cal_e
    out[f"calibrated_var_{model}"] = cal_var
    out[f"calibrated_log_var_{model}"] = np.log(
        cal_var.clip(lower=EPS)
    )
    out[f"calibrated_std_{model}"] = cal_std

    if threshold is not None:
        out[f"high_uncertainty_{model}"] = (
            cal_std > threshold
        )

    return out


def gaussian_crps(y, mu, sigma):
    """Gaussian continuous ranked probability score."""

    sigma = np.maximum(sigma, np.sqrt(EPS))
    z = (y - mu) / sigma

    density = (
        np.exp(-0.5 * z**2)
        / np.sqrt(2.0 * np.pi)
    )

    return sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * density
        - 1.0 / np.sqrt(np.pi)
    )


def calculate_metrics(
    df,
    mask,
    plan,
    model,
    scale_a,
    scale_e,
    threshold,
    domain,
):
    y, mu, raw_a, raw_e, valid = get_values(df, model)
    selected = valid & np.asarray(mask)

    if not selected.any():
        return None

    y = y[selected]
    mu = mu[selected]
    raw_a = raw_a[selected]
    raw_e = raw_e[selected]

    error = mu - y
    absolute_error = np.abs(error)

    raw_std = np.sqrt(raw_a**2 + raw_e**2)

    cal_a = scale_a * raw_a
    cal_e = scale_e * raw_e
    cal_std = np.sqrt(cal_a**2 + cal_e**2)

    raw_z = error / raw_std
    cal_z = error / cal_std

    raw_crps = gaussian_crps(y, mu, raw_std)
    cal_crps = gaussian_crps(y, mu, cal_std)

    cal_nll = 0.5 * np.mean(
        np.log(2.0 * np.pi * cal_std**2)
        + error**2 / cal_std**2
    )

    if (
        len(error) > 1
        and np.std(absolute_error) > 0
        and np.std(cal_std) > 0
    ):
        correlation = float(
            spearmanr(absolute_error, cal_std)[0]
        )
    else:
        correlation = np.nan

    return {
        "Plan": plan,
        "Model": model,
        "Model_Short": short_model_name(model),
        "Domain": domain,
        "N_Sequences": int(selected.sum()),
        "N_Organisms": int(
            df.loc[selected, TAXID].nunique()
        ),
        "Bias": error.mean(),
        "MAE": absolute_error.mean(),
        "RMSE": np.sqrt(np.mean(error**2)),
        "Pct_Within_5C": 100.0 * np.mean(absolute_error <= 5),
        "Pct_Within_10C": 100.0 * np.mean(absolute_error <= 10),
        "Raw_TotalStd_Mean": raw_std.mean(),
        "Cal_AleatoricStd_Mean": cal_a.mean(),
        "Cal_EpistemicStd_Mean": cal_e.mean(),
        "Cal_TotalStd_Mean": cal_std.mean(),
        "Raw_Z_RMS": np.sqrt(np.mean(raw_z**2)),
        "Cal_Z_Mean": cal_z.mean(),
        "Cal_Z_RMS": np.sqrt(np.mean(cal_z**2)),
        "Raw_Coverage95": 100.0 * np.mean(
            np.abs(raw_z) <= 1.96
        ),
        "Cal_Coverage95": 100.0 * np.mean(
            np.abs(cal_z) <= 1.96
        ),
        "Raw_CRPS": raw_crps.mean(),
        "Cal_CRPS": cal_crps.mean(),
        "Cal_NLL": cal_nll,
        "Uncertainty_Error_Spearman": correlation,
        "High_Uncertainty_Pct": 100.0 * np.mean(
            cal_std > threshold
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--psych-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of psychrophiles in calibration loss. "
            "Default 0.5 gives psychrophiles and others equal weight."
        ),
    )

    parser.add_argument(
        "--flag-quantile",
        type=float,
        default=0.95,
        help=(
            "Psychrophile validation uncertainty quantile used "
            "for the high-uncertainty flag."
        ),
    )

    args = parser.parse_args()

    if not 0 < args.psych_weight < 1:
        raise ValueError("--psych-weight must be between 0 and 1.")

    if not 0 < args.flag_quantile < 1:
        raise ValueError("--flag-quantile must be between 0 and 1.")

    manifest_path = args.manifest.resolve()
    base = manifest_path.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    parameter_rows = []
    summary_rows = []

    for _, entry in manifest.iterrows():
        plan = str(entry["plan"])

        split_file = resolve_path(base, entry["split_file"])
        val_file = resolve_path(base, entry["val_predictions"])
        test_file = resolve_path(base, entry["test_predictions"])

        validation = load_split_predictions(
            split_file,
            val_file,
            "val",
        )

        test = load_split_predictions(
            split_file,
            test_file,
            "test",
        )

        models = sorted(
            set(find_models(validation))
            & set(find_models(test))
        )

        calibrated_test = test.copy()

        for model in models:
            scale_a, scale_e, n_psych, n_other = (
                fit_component_scales(
                    validation,
                    model,
                    args.psych_weight,
                )
            )

            calibrated_validation = add_calibrated_columns(
                validation,
                model,
                scale_a,
                scale_e,
            )

            cal_std_column = f"calibrated_std_{model}"

            psych_validation = is_psychrophile(
                calibrated_validation
            )

            threshold = calibrated_validation.loc[
                psych_validation,
                cal_std_column,
            ].quantile(args.flag_quantile)

            calibrated_test = add_calibrated_columns(
                calibrated_test,
                model,
                scale_a,
                scale_e,
                threshold,
            )

            at_bound = (
                scale_a <= 0.051
                or scale_a >= 19.9
                or scale_e <= 0.051
                or scale_e >= 19.9
            )

            parameter_rows.append({
                "Plan": plan,
                "Model": model,
                "Model_Short": short_model_name(model),
                "Aleatoric_Scale": scale_a,
                "Epistemic_Scale": scale_e,
                "High_Uncertainty_Threshold": threshold,
                "N_Val_Psychrophiles": n_psych,
                "N_Val_NonPsychrophiles": n_other,
                "Psych_Weight": args.psych_weight,
                "Scale_At_Bound": at_bound,
            })

            for domain, mask in domain_masks(test).items():
                row = calculate_metrics(
                    test,
                    mask,
                    plan,
                    model,
                    scale_a,
                    scale_e,
                    threshold,
                    domain,
                )

                if row is not None:
                    summary_rows.append(row)

        calibrated_test.to_csv(
            output_dir
            / f"{slug(plan)}_calibrated_test_predictions.csv",
            index=False,
        )

    parameters = pd.DataFrame(parameter_rows)
    summary = pd.DataFrame(summary_rows)

    parameters.to_csv(
        output_dir / "calibration_parameters.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "test_summary.csv",
        index=False,
    )

    # Compact psychrophile-centric ranking with OOD diagnostics.
    psych = summary.loc[
        summary["Domain"] == "Psychrophiles"
    ].copy()

    others = summary.loc[
        summary["Domain"] == "Non-psychrophiles",
        [
            "Plan",
            "Model",
            "Cal_TotalStd_Mean",
            "Cal_Coverage95",
            "High_Uncertainty_Pct",
        ],
    ].rename(columns={
        "Cal_TotalStd_Mean": "OOD_TotalStd_Mean",
        "Cal_Coverage95": "OOD_Coverage95",
        "High_Uncertainty_Pct": "OOD_HighUncertainty_Pct",
    })

    ranking = psych.merge(
        others,
        on=["Plan", "Model"],
        how="left",
    )

    ranking["Psych_Coverage95_Error"] = abs(
        ranking["Cal_Coverage95"] - 95.0
    )

    ranking["OOD_to_Psych_Std_Ratio"] = (
        ranking["OOD_TotalStd_Mean"]
        / ranking["Cal_TotalStd_Mean"]
    )

    ranking["OOD_Flag_Separation_PctPts"] = (
        ranking["OOD_HighUncertainty_Pct"]
        - ranking["High_Uncertainty_Pct"]
    )

    ranking = ranking.sort_values(
        ["Cal_CRPS", "MAE", "Psych_Coverage95_Error"]
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "Rank",
        np.arange(1, len(ranking) + 1),
    )

    ranking.to_csv(
        output_dir / "psychrophile_model_ranking.csv",
        index=False,
    )

    display_columns = [
        "Rank",
        "Plan",
        "Model_Short",
        "MAE",
        "RMSE",
        "Cal_CRPS",
        "Cal_Coverage95",
        "Cal_Z_RMS",
        "High_Uncertainty_Pct",
        "OOD_to_Psych_Std_Ratio",
        "OOD_HighUncertainty_Pct",
        "OOD_Flag_Separation_PctPts",
    ]

    print(
        ranking[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()