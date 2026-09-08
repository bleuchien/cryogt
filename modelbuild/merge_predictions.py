from pathlib import Path
import argparse
import sys
import pandas as pd
from config import Config

# parse command line arguments
parser = argparse.ArgumentParser(prog='CryOGT prediction merge.')
parser.add_argument('-c', '--config', default='config.yaml', help='Configuration file.')
parser.add_argument('--file1', default='prediction_pland.csv',
                     help='Primary file (point predictions, std, metadata).')
parser.add_argument('--file2', default='prediction_planc_with_pland_split.csv',
                     help='Secondary file (log_var, aleatoric/epistemic std).')
parser.add_argument('-o', '--output', default='prediction_merged.csv', help='Output file name.')
args = parser.parse_args()

# sanity check for the config file
config_path = Path(args.config)
if not config_path.exists():
    print(f'Config file {config_path} does not exist!')
    sys.exit(1)

config = Config.from_yaml(config_path)
data_dir = Path(config.paths.data_dir)

file1_path = data_dir / args.file1
file2_path = data_dir / args.file2
output_path = data_dir / args.output

for f in [file1_path, file2_path]:
    if not f.exists():
        print(f'Prediction data file {f} not found.')
        sys.exit(1)

# merge key
merge_key = 'member'

# metadata columns kept from file1 as-is
metadata_cols = ['ncbiTaxID_new', 'Temp_Duplicate_Average', 'bin_name']

# model names the per-model column templates apply to
model_names = [
    'facebook/esm2_t12_35M_UR50D_head_M',
    'facebook/esm2_t30_150M_UR50D_head_M',
]

# per-model column source template: True = from file1, False = from file2
model_col_template = {
    '{model}': True,
    'std_{model}': True,
    'log_var_{model}': False,
    'aleatoric_std_{model}': False,
    'epistemic_std_{model}': False,
}

# build the explicit (column_name -> from_file1) mapping
col_sources = {}
for model in model_names:
    for template, from_file1 in model_col_template.items():
        col_sources[template.format(model=model)] = from_file1

file1_cols = [merge_key] + metadata_cols + [c for c, f1 in col_sources.items() if f1]
file2_cols = [merge_key] + [c for c, f1 in col_sources.items() if not f1]

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# validate required columns exist before subsetting
for name, df, cols in [('file1', df1, file1_cols), ('file2', df2, file2_cols)]:
    missing = set(cols) - set(df.columns)
    if missing:
        print(f'{name} ({args.file1 if name == "file1" else args.file2}) is missing columns: {sorted(missing)}')
        sys.exit(1)

merged = df1[file1_cols].merge(
    df2[file2_cols], on=merge_key, how='inner', validate='one_to_one'
)

# sanity check: no rows lost/gained relative to file1
if len(merged) != len(df1):
    print(f'WARNING: merged row count ({len(merged)}) differs from {args.file1} ({len(df1)}) '
          f'— check for mismatched "{merge_key}" values between files.')

merged.to_csv(output_path, index=False)
print(f'Wrote merged file: {output_path} ({len(merged)} rows, {len(merged.columns)} columns)')
