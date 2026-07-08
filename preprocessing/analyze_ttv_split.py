from pathlib import Path
import pandas as pd
import sys

# this script analyzes the train/test/validation splits

# main data directory
data_dir = Path('../data')

# train/test/val split file
split_file = data_dir / 'ttv_splits.csv'

# diversity threshold: minimum organisms per bin required in val/test
min_organisms_threshold = 5

if not split_file.exists():
    print(f'{split_file} does not exist.')
    sys.exit(1)

# read splits file
splits = pd.read_csv(split_file)

# checking for missing values
n_missing_split = splits['split'].isna().sum()
if n_missing_split:
    print(f'{n_missing_split} rows have no split assignment and will be excluded below!')

n_missing_bin = splits['bin_name'].isna().sum()
if n_missing_bin:
    print(f'{n_missing_bin} rows have no bin_name and will be excluded below!')

print('Sequence Distribution')

# create a new dataframe analyzing the sequence distribution
seq_dist = (
    splits.groupby(['split', 'bin_name'], dropna=False)  # group by split and bin_name
    .size()                                              # count the rows of each grouping
    .unstack()                                           # pivot the table
    .fillna(0)                                           # replace missing values with 0
    .astype(int)                                         # convert to integer
)
# add the total as new column
seq_dist['total'] = seq_dist.sum(axis=1)
print(seq_dist)

assert seq_dist['total'].sum() == len(splits), \
    'Row count mismatch between seq_dist and splits — check groupby dropna behaviour.'

# verify organism count per bin_name
print('\nOrganism Count per bin name')

# create a new dataframe counting organisms per bin_name
org_dist = (
    splits.groupby(['split', 'bin_name'], dropna=False)['ncbiTaxID_new']  # group by split and bin_name and only select taxonomy ID
    .nunique()                                                            # count the number of unique organisms
    .unstack()                                                            # pivot the table
    .fillna(0)                                                            # replace missing values with 0
    .astype(int)                                                          # convert to integer 
)
print(org_dist)

print(f'\nChecking val/test split diversity (threshold: <{min_organisms_threshold} organisms)')
# checking val/test split for bins with fewer than 5 organisms
for split in ['val', 'test']:
    if split not in org_dist.index:
        print(f'split "{split}" not found in data!')
        continue
    # iterate over the bin_name
    for bin_name in org_dist.columns:
        num_organisms = int(org_dist.loc[split, bin_name])
        num_sequences = int(seq_dist.loc[split, bin_name])
        if num_organisms < min_organisms_threshold:
            print(f'{split} - {bin_name} has only {num_organisms} organisms ({num_sequences} sequences)!')

print('\nSplit proportions (% of total sequences)')
print((seq_dist['total'] / seq_dist['total'].sum() * 100).round(1))

print('\nOGT summary per Split')
# group the dataset by split but only select the OGT and print a summary
print(splits.groupby('split')['Temp_Duplicate_Average'].describe().round(2))
