import sys
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from config import Config
from pathlib import Path
from datetime import datetime

# column name setup
target_col = 'Temp_Duplicate_Average'
bin_col = 'bin_name'

# current time stamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# parse command line arguments
parser = argparse.ArgumentParser(prog='CryOGT baseline calculation')
parser.add_argument('-c', '--config', help='Configuration file.', default='config.yaml')
parser.add_argument('-s', '--size', default=None, help='Head size parameter.')
args = parser.parse_args()

# sanity check for the config file
config_path = Path(args.config)
if not config_path.exists():
    print(f'Config file {config_path} does not exit!')
    sys.exit(1)

# read configuration
config = Config.from_yaml(config_path)

# data file path
data_file = Path(config.paths.data_dir) / 'baseline.csv'

if not data_file.exists():
    print(f'Baseline data file {data_file} not found.')
    sys.exit(1)

# load base line data
df = pd.read_csv(data_file)

# get the model names from the dataframe
models = [ c for c in df.columns if c.startswith('facebook/esm2') ]
# select a subset if given
if not args.size is None:
    models = [ c for c in models if c.endswith('head_' + str(args.size)) ]
# build tuple of full model name and shortened model name (ie 150M) (facebook/esm2_t30_150M_UR50D_head_S)
model_info = sorted([ (m, m.split('_')[2] + ' ' + m.split('_')[5], m.split('_')[2]) for m in models ], key=lambda x: int(x[2].removesuffix('M')))

models = [m for m, label, short in model_info]
model_labels = [label for m, label, short in model_info]

# convert OGT columns to a numeric type
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
for m in models:
    df[m] = pd.to_numeric(df[m], errors='coerce')

# fix bin name typo
df[bin_col] = df[bin_col].astype(str).str.replace('mesophiels', 'mesophiles')


# calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)                       # mean aboslute error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))              # root mean square error
    residuals = y_pred - y_true                                     # offset of the prediction to the true OGT
    abs_error = np.abs(residuals)                                   # absolute error
    mean_res = np.mean(residuals)                                   # mean residual
    median_res = np.median(residuals)                               # median residual
    median_abs_error = np.median(abs_error)                         # median absolute error
    
    r2 = r2_score(y_true, y_pred)                                   # R-square score
    
    pct_within_5 = np.mean(np.abs(residuals) <= 5.0) * 100          # % within 5°C
    pct_within_10 = np.mean(np.abs(residuals) <= 10.0) * 100        # % within 10°C
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Mean_Residual_Bias': mean_res,
        'Median_Residual': median_res,
        'Median_Abs_Error': median_abs_error,
        'R2': r2,
        'Pct_Within_5C': pct_within_5,
        'Pct_Within_10C': pct_within_10
    }

results_list = []

# loop metric calculation over all entries
for model, label in zip(models, model_labels):
    # overall metrics
    overall_metrics = calculate_metrics(df[target_col], df[model])
    overall_metrics.update({'Model': label, 'Subset': 'Overall'})
    results_list.append(overall_metrics)
    
    # psychrophile metrics
    psychro_df = df[df[bin_col] == 'psychrophiles']
    psychro_metrics = calculate_metrics(psychro_df[target_col], psychro_df[model])
    psychro_metrics.update({'Model': label, 'Subset': 'Psychrophiles'})
    results_list.append(psychro_metrics)

# assemble results into one dataframe
results_df = pd.DataFrame(results_list)
# reorder columns for readability
cols = ['Model', 'Subset', 'MAE', 'RMSE', 'Mean_Residual_Bias', 'Median_Residual', 'Median_Abs_Error', 'R2', 'Pct_Within_5C', 'Pct_Within_10C']
results_df = results_df[cols]
outfile = Path(config.paths.data_dir) / 'baseline_metrics_summary.csv'
results_df.to_csv(outfile, index=False)
print(f'Metrics saved to {outfile}')
print('Metrics Summary')
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))



# analyze significance of the difference between Psychrophile and Others predictions
test_rows = []

for model, label in zip(models, model_labels):
    test_df = df[[target_col, bin_col, model]].dropna().copy()

    test_df['residual'] = test_df[model] - test_df[target_col]
    test_df['abs_error'] = test_df['residual'].abs()

    psychro = test_df[test_df[bin_col] == 'psychrophiles']
    other = test_df[test_df[bin_col] != 'psychrophiles']

    for quantity in ['residual', 'abs_error']:
        a = psychro[quantity].to_numpy(dtype=float)
        b = other[quantity].to_numpy(dtype=float)

        if len(a) < 2 or len(b) < 2:
            continue

        # welch = stats.ttest_ind(a, b, equal_var=False)
        mann = stats.mannwhitneyu(a, b, alternative='two-sided')
        cliffs_delta = 2 * mann.statistic / (len(a) * len(b)) - 1           # size of the prediction difference [-1, 1]

        test_rows.append({
            'Model': label,
            'Quantity': quantity,
            'Psychrophile_Mean': np.mean(a),
            'Other_Mean': np.mean(b),
            'Difference_Psychro_Minus_Other': np.mean(a) - np.mean(b),
            # 'Welch_t_p': welch.pvalue,
            # 'MannWhitney_p': mann.pvalue,
            'Cliffs_Delta': cliffs_delta,
        })

tests_df = pd.DataFrame(test_rows)
outfile = Path(config.paths.data_dir) / 'baseline_psychro_vs_other_tests.csv'
tests_df.to_csv(outfile, index=False)
print(f'\nSignificance saved to {outfile}')
print('Significance Summary')
print(tests_df.to_string(index=False, float_format=lambda x: f'{x:.4g}'))



# visualizations
sns.set_theme(style='whitegrid')

# figure 1: Predicted vs True OGT (2x2 Grid)
rows = int(len(models) / 2)
fig, axes = plt.subplots(rows, 2, figsize=(16, 7 * rows))
axes = axes.flatten()

for i, (model, label) in enumerate(zip(models, model_labels)):
    sns.scatterplot(data=df, x=target_col, y=model, hue=bin_col, alpha=0.6, ax=axes[i])
    
    # Perfect prediction line and +/- 5C error bands
    axes[i].plot([0, 100], [0, 100], 'k--', label='Perfect Prediction')
    axes[i].fill_between([0, 100], [-5, 95], [5, 105], color='gray', alpha=0.2, label='±5°C Tolerance')
    
    axes[i].set_title(f'{label} Baseline: True vs Predicted')
    axes[i].set_xlabel('True OGT (°C)')
    axes[i].set_ylabel('Predicted OGT (°C)')
    
    if i == 1:
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[i].get_legend().remove()

plt.tight_layout()
outfile = Path(config.paths.data_dir) / 'figure_1_baseline_scatter_grid.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 2: Residual (Bias) Distribution by Bin (2x2 Grid)
# Define bin order
bin_order = [
    'psychrophiles', 
    'mesophiles bin 1', 
    'mesophiles bin 2', 
    'mesophiles bin 3',
    'mesophiles bin 4', 
    'thermophiles', 
    'hyperthermophiles'
]
fig, axes = plt.subplots(rows, 2, figsize=(16, 7 * rows))
axes = axes.flatten()

for i, (model, label) in enumerate(zip(models, model_labels)):
    df['temp_residual'] = df[model] - df[target_col]
    
    sns.violinplot(
        data=df, 
        x=bin_col, 
        y='temp_residual', 
        ax=axes[i], 
        inner='quartile', 
        order=bin_order
    )
    
    axes[i].axhline(0, color='k', linestyle='--', linewidth=1.5) # Zero bias line
    
    axes[i].set_title(f'{label} Baseline: Directional Bias by Bin')
    axes[i].set_xlabel('Temperature Bin')
    axes[i].set_ylabel('Residual (Predicted - True) °C')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
outfile = Path(config.paths.data_dir) / 'figure_2_baseline_bias_violin_grid.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close(fig)

# figure 3: Metric Comparison Bar Charts (Psychrophiles vs Overall)
fig, axes = plt.subplots(4, 1, figsize=(int(len(models)), 21))

# melt dataframe for easier plotting
melted_df = pd.melt(results_df, id_vars=['Model', 'Subset'], value_vars=['MAE', 'RMSE', 'R2', 'Pct_Within_10C'])

# plot MAE
sns.barplot(data=melted_df[melted_df['variable'] == 'MAE'], x='Model', y='value', hue='Subset', ax=axes[0])
axes[0].set_title('Mean Absolute Error (Lower is Better)')
axes[0].set_ylabel('MAE (°C)')

# plot RSME
sns.barplot(data=melted_df[melted_df['variable'] == 'RMSE'], x='Model', y='value', hue='Subset', ax=axes[1])
axes[1].set_title('Root Mean Square Error (Lower is Better)')
axes[1].set_ylabel('RMSE (°C)')

# plot R2
sns.barplot(data=melted_df[melted_df['variable'] == 'R2'], x='Model', y='value', hue='Subset', ax=axes[2])
axes[2].set_title('R² Score (Higher is Better, Max 1.0)')
axes[2].set_ylabel('R²')
axes[2].set_ylim(min(0, results_df['R2'].min() * 1.1), 1.0) # Ensure y-axis makes sense for R2

# plot % within 10°C
sns.barplot(data=melted_df[melted_df['variable'] == 'Pct_Within_10C'], x='Model', y='value', hue='Subset', ax=axes[3])
axes[3].set_title('% Predictions Within ±10°C (Higher is Better)')
axes[3].set_ylabel('Percentage (%)')
axes[3].set_ylim(0, 100)

plt.tight_layout()
outfile = Path(config.paths.data_dir) / 'figure_3_baseline_metrics_barcharts.png'
plt.savefig(outfile, dpi=300)
plt.close(fig)
