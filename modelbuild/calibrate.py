import argparse
from pathlib import Path
import numpy as np
import pandas as pd


EPS = 1e-12

MEMBER_COL = 'member'
TAXID_COL = 'ncbiTaxID_new'
TARGET_COL = 'Temp_Duplicate_Average'
BIN_COL = 'bin_name'


def clean_bin_names(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace('mesophiels', 'mesophiles', regex=False)
    )


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={
            MEMBER_COL: str,
            TAXID_COL: str,
        }
    )


def select_prediction_split(predictions, split_df, expected_split):
    """Attach the original split assignment and retain the requested split."""

    split_map = split_df[[MEMBER_COL, 'split']].copy()

    if split_map[MEMBER_COL].duplicated().any():
        raise ValueError(
            f'Duplicate members found in the split file.'
        )

    merged = predictions.merge(
        split_map,
        on=MEMBER_COL,
        how='left',
        validate='one_to_one',
    )

    missing = merged['split'].isna()

    if missing.any():
        examples = merged.loc[missing, MEMBER_COL].head().tolist()
        raise ValueError(
            f'{missing.sum()} prediction members were not found in the '
            f'split file. Examples: {examples}'
        )

    selected = merged.loc[
        merged['split'] == expected_split
    ].copy()

    if selected.empty:
        raise ValueError(
            f'No "{expected_split}" predictions found. '
            f'The prediction file must contain predictions for that split.'
        )

    selected.drop(columns='split', inplace=True)

    return selected


def discover_models(df):
    """Find model names from aleatoric_std_<model> columns."""

    prefix = 'aleatoric_std_'

    models = sorted(
        column[len(prefix):]
        for column in df.columns
        if column.startswith(prefix)
    )

    if not models:
        raise ValueError(
            'No columns beginning with "aleatoric_std_" were found.'
        )

    for model in models:
        required = [
            model,
            f'aleatoric_std_{model}',
            f'epistemic_std_{model}',
        ]

        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(
                f'Missing columns for model {model}: {missing}'
            )

    return models


def get_model_arrays(df, model):
    y = pd.to_numeric(df[TARGET_COL], errors='coerce').to_numpy(float)
    mu = pd.to_numeric(df[model], errors='coerce').to_numpy(float)

    aleatoric_std = pd.to_numeric(
        df[f'aleatoric_std_{model}'],
        errors='coerce',
    ).to_numpy(float)

    epistemic_std = pd.to_numeric(
        df[f'epistemic_std_{model}'],
        errors='coerce',
    ).to_numpy(float)

    total_var = (
        np.square(aleatoric_std) +
        np.square(epistemic_std)
    )

    valid = (
        np.isfinite(y) &
        np.isfinite(mu) &
        np.isfinite(total_var) &
        (total_var > 0)
    )

    return y, mu, aleatoric_std, epistemic_std, total_var, valid


def fit_total_scale(df, model, weighting='sequence'):
    """Fit one multiplicative scale for total predictive uncertainty."""

    y, mu, _, _, total_var, valid = get_model_arrays(df, model)

    residual = y[valid] - mu[valid]
    normalized_squared_error = (
        np.square(residual) /
        np.maximum(total_var[valid], EPS)
    )

    if weighting == 'sequence':
        mean_normalized_error = normalized_squared_error.mean()

    elif weighting == 'organism':
        taxids = (
            df.loc[valid, TAXID_COL]
            .astype(str)
            .to_numpy()
        )

        grouped = pd.DataFrame({
            TAXID_COL: taxids,
            'normalized_squared_error': normalized_squared_error,
        })

        # First average within each organism, then across organisms.
        mean_normalized_error = (
            grouped
            .groupby(TAXID_COL)['normalized_squared_error']
            .mean()
            .mean()
        )

    else:
        raise ValueError(f'Unknown weighting method: {weighting}')

    scale = np.sqrt(mean_normalized_error)

    return scale, int(valid.sum())


def apply_total_scale(df, model, scale):
    """Apply the common scale and add calibrated columns."""

    out = df.copy()

    aleatoric_std = pd.to_numeric(
        out[f'aleatoric_std_{model}'],
        errors='coerce',
    )

    epistemic_std = pd.to_numeric(
        out[f'epistemic_std_{model}'],
        errors='coerce',
    )

    raw_total_var = (
        aleatoric_std.pow(2) +
        epistemic_std.pow(2)
    )

    calibrated_var = scale**2 * raw_total_var

    out[f'calibrated_var_{model}'] = calibrated_var
    out[f'calibrated_log_var_{model}'] = np.log(
        calibrated_var.clip(lower=EPS)
    )
    out[f'calibrated_std_{model}'] = np.sqrt(calibrated_var)

    # These preserve the original aleatoric/epistemic proportions.
    # They are not independently calibrated components.
    out[f'scaled_aleatoric_std_{model}'] = (
        scale * aleatoric_std
    )
    out[f'scaled_epistemic_std_{model}'] = (
        scale * epistemic_std
    )

    return out


def calculate_metrics(df, model, dataset_name, state):
    y, mu, aleatoric_std, epistemic_std, raw_var, valid = (
        get_model_arrays(df, model)
    )

    if state == 'Raw':
        total_std = np.sqrt(raw_var)

    elif state == 'Calibrated':
        total_std = pd.to_numeric(
            df[f'calibrated_std_{model}'],
            errors='coerce',
        ).to_numpy(float)

        valid = (
            valid &
            np.isfinite(total_std) &
            (total_std > 0)
        )

    else:
        raise ValueError(f'Unknown state: {state}')

    clean_bins = clean_bin_names(df[BIN_COL]).to_numpy()

    subset_masks = {
        'Overall': np.ones(len(df), dtype=bool),
        'Psychrophiles': clean_bins == 'psychrophiles',
        'Others': clean_bins != 'psychrophiles',
    }

    rows = []

    for subset_name, subset_mask in subset_masks.items():
        mask = valid & subset_mask

        if not mask.any():
            continue

        # Positive residual means that OGT was overpredicted.
        residual = mu[mask] - y[mask]
        std = total_std[mask]
        var = np.square(std)
        z = residual / std

        gaussian_nll = 0.5 * np.mean(
            np.log(2.0 * np.pi * var) +
            np.square(residual) / var
        )

        taxid_count = (
            df.loc[mask, TAXID_COL]
            .astype(str)
            .nunique()
        )

        rows.append({
            'Dataset': dataset_name,
            'Model': model,
            'State': state,
            'Subset': subset_name,
            'N_Sequences': int(mask.sum()),
            'N_Organisms': int(taxid_count),
            'Mean_Residual': residual.mean(),
            'MAE': np.abs(residual).mean(),
            'RMSE': np.sqrt(np.mean(np.square(residual))),
            'Mean_Total_Std': std.mean(),
            'Median_Total_Std': np.median(std),
            'Z_Mean': z.mean(),
            'Z_Std': z.std(ddof=0),
            'Coverage_1SD': 100.0 * np.mean(np.abs(z) <= 1.0),
            'Coverage_1.96SD': 100.0 * np.mean(np.abs(z) <= 1.96),
            'Coverage_2SD': 100.0 * np.mean(np.abs(z) <= 2.0),
            'Gaussian_NLL': gaussian_nll,
        })

    return rows


def output_path(prefix, suffix):
    return prefix.parent / f'{prefix.name}_{suffix}.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate total ensemble uncertainty.'
    )

    parser.add_argument(
        '--split-file',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--val-predictions',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--test-predictions',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-prefix',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--weighting',
        choices=['sequence', 'organism'],
        default='sequence',
    )

    args = parser.parse_args()

    split_df = read_csv(args.split_file)
    val_predictions = read_csv(args.val_predictions)
    test_predictions = read_csv(args.test_predictions)

    val_df = select_prediction_split(
        val_predictions,
        split_df,
        expected_split='val',
    )

    test_df = select_prediction_split(
        test_predictions,
        split_df,
        expected_split='test',
    )

    models = discover_models(val_df)

    missing_test_models = [
        model for model in models
        if model not in discover_models(test_df)
    ]

    if missing_test_models:
        raise ValueError(
            f'Models missing from test predictions: '
            f'{missing_test_models}'
        )

    calibrated_val = val_df.copy()
    calibrated_test = test_df.copy()

    parameter_rows = []

    for model in models:
        scale, n_sequences = fit_total_scale(
            val_df,
            model,
            weighting=args.weighting,
        )

        parameter_rows.append({
            'Model': model,
            'Weighting': args.weighting,
            'N_Validation_Sequences': n_sequences,
            'N_Validation_Organisms': val_df[TAXID_COL].nunique(),
            'Std_Scale': scale,
            'Variance_Scale': scale**2,
        })

        calibrated_val = apply_total_scale(
            calibrated_val,
            model,
            scale,
        )

        calibrated_test = apply_total_scale(
            calibrated_test,
            model,
            scale,
        )

        print(
            f'{model}: '
            f'std scale = {scale:.4f}, '
            f'variance scale = {scale**2:.4f}'
        )

    metric_rows = []

    for dataset_name, data in [
        ('Validation', calibrated_val),
        ('Test', calibrated_test),
    ]:
        for model in models:
            metric_rows.extend(
                calculate_metrics(
                    data,
                    model,
                    dataset_name,
                    state='Raw',
                )
            )

            metric_rows.extend(
                calculate_metrics(
                    data,
                    model,
                    dataset_name,
                    state='Calibrated',
                )
            )

    parameters_df = pd.DataFrame(parameter_rows)
    metrics_df = pd.DataFrame(metric_rows)

    args.output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters_file = output_path(
        args.output_prefix,
        'calibration_parameters',
    )

    metrics_file = output_path(
        args.output_prefix,
        'calibration_metrics',
    )

    val_file = output_path(
        args.output_prefix,
        'val_predictions_calibrated',
    )

    test_file = output_path(
        args.output_prefix,
        'test_predictions_calibrated',
    )

    parameters_df.to_csv(parameters_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)
    calibrated_val.to_csv(val_file, index=False)
    calibrated_test.to_csv(test_file, index=False)

    print(f'Parameters saved to {parameters_file}')
    print(f'Metrics saved to {metrics_file}')
    print(f'Calibrated validation predictions saved to {val_file}')
    print(f'Calibrated test predictions saved to {test_file}')

    print('\nCalibration parameters')
    print(
        parameters_df.to_string(
            index=False,
            float_format=lambda x: f'{x:.4f}',
        )
    )

    print('\nTest calibration metrics')
    print(
        metrics_df[
            metrics_df['Dataset'] == 'Test'
        ].to_string(
            index=False,
            float_format=lambda x: f'{x:.3f}',
        )
    )


if __name__ == '__main__':
    main()