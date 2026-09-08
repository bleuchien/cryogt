from pathlib import Path
import argparse
import sys
import pandas as pd
from config import Config

# parse command line arguments
parser = argparse.ArgumentParser(prog='CryOGT split overlap check.')
parser.add_argument('-c', '--config', default='config.yaml', help='Configuration file.')
parser.add_argument('--file1', default='ttv_splits.csv', help='First splits file.')
parser.add_argument('--file2', default='ttv_splits_planc.csv', help='Second splits file.')
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

for f in [file1_path, file2_path]:
    if not f.exists():
        print(f'Splits file {f} not found.')
        sys.exit(1)

valid_splits = {'train', 'val', 'test'}

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

for name, df in [(args.file1, df1), (args.file2, df2)]:
    unexpected = set(df['split'].dropna().unique()) - valid_splits
    if unexpected:
        print(f'WARNING: {name} contains unexpected split labels: {sorted(unexpected)}')

# map member -> split for each file (guards against duplicate members per file)
def member_split_map(df, name):
    dup = df['member'][df['member'].duplicated()]
    if not dup.empty:
        print(f'WARNING: {name} has {dup.nunique()} duplicated member IDs; keeping first occurrence.')
    return df.drop_duplicates(subset='member', keep='first').set_index('member')['split']

map1 = member_split_map(df1, args.file1)
map2 = member_split_map(df2, args.file2)

members1, members2 = set(map1.index), set(map2.index)
common = members1 & members2
only1 = members1 - members2
only2 = members2 - members1

print(f'{args.file1}: {len(members1)} members')
print(f'{args.file2}: {len(members2)} members')
print(f'Common members: {len(common)}')
print(f'Only in {args.file1}: {len(only1)}')
print(f'Only in {args.file2}: {len(only2)}\n')

# cross-tabulate split assignment for common members: rows = file1 split, cols = file2 split
common_df = pd.DataFrame({
    'split_file1': map1.loc[list(common)],
    'split_file2': map2.loc[list(common)],
})

crosstab = pd.crosstab(common_df['split_file1'], common_df['split_file2'])
# ensure full 3x3 shape even if a category has zero members
crosstab = crosstab.reindex(index=sorted(valid_splits), columns=sorted(valid_splits), fill_value=0)

print('Cross-tabulation of common members (rows: file1 split, cols: file2 split):')
print(crosstab)

# flag leakage: off-diagonal cells = member assigned to DIFFERENT splits across files
print('\nPotential leakage (members assigned to different splits across files):')
leak_found = False
for split_a in sorted(valid_splits):
    for split_b in sorted(valid_splits):
        if split_a == split_b:
            continue
        n = crosstab.loc[split_a, split_b]
        if n > 0:
            leak_found = True
            print(f'  {args.file1}:{split_a} <-> {args.file2}:{split_b}: {n} members')

if not leak_found:
    print('  None found — no cross-split member overlap.')

# same-split overlap counts (expected/benign, reported for completeness)
print('\nSame-split overlap (benign, expected if using shared organisms):')
for split in sorted(valid_splits):
    n = crosstab.loc[split, split]
    print(f'  {split}: {n} members in {split} in both files')
