from pathlib import Path
import pandas as pd
import numpy as np
import sys

# this script creates the train/test/validation splits
# it reads the mmseqs2 cluster file and the OGT samples file and merges both
# cluster sizes and the most frequend bin name are extracted
# large clusters are assigned to train/test/val in round robin fashion
# small clusters are grouped by bin name for a stratified split
# the split assignment is added as new column and saved to a new file

# split sizes
train_split_size = 0.80
val_split_size = 0.10
test_split_size = round(1 - train_split_size - val_split_size, 10)
assert test_split_size > 0, 'train_split_size + val_split_size must be < 1'

# large-vs-small cluster threshold (member count)
large_cluster_threshold = 50

# main data directory
data_dir = Path('../data')

# protein clusters file
cluster_file = data_dir / 'mmseqs/stage2/split_clusters.tsv'

# organism selection file
ogt_file = data_dir / 'growth_temp_dataset_selection.csv'

# train/test/val split file
split_file = data_dir / 'ttv_splits.csv'

# check if file exists
for f in [cluster_file, ogt_file]:
    if not f.exists():
        print(f'{f} does not exist.')
        sys.exit(1)

# read clusters file
clusters = pd.read_csv(cluster_file, sep='\t', header=None, names=['representative', 'member'])

# split taxonomy ID off from the FASTA ID which is taxonomy_ID|protein_ID
clusters['ncbiTaxID_new'] = clusters['member'].str.split('|').str[0].astype(int)

# load optimal growth temperature dataset
ogt_df = pd.read_csv(ogt_file)

# merge cluster and OGT dataset on taxonomy ID
clusters = clusters.merge(ogt_df[['ncbiTaxID_new', 'Temp_Duplicate_Average', 'bin_name']], on='ncbiTaxID_new', how='left')

# check for merge problems
n_unmatched = clusters['bin_name'].isna().sum()
if n_unmatched:
    print(f'WARNING: {n_unmatched} members ({n_unmatched / len(clusters):.2%}) had no matching OGT/bin_name entry.')

# custom mode function
def custom_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else 'unknown'

# build a dataframe containing the representatives, member count and the most common bin_name
cluster_sizes = (
    clusters.groupby('representative')
    .agg(
        # sum up all members
        n_members=('member', 'count'),
        # get the most frequent bin_name
        dominant_bin=('bin_name', custom_mode),
    )
    .reset_index()
    .sort_values(['n_members', 'representative'], ascending=[False, True], kind='mergesort')
)

# dictionary mapping representatives/clusters to a particular split
assignments = {}

# split the clusters into two depending on the number of members
large = cluster_sizes[cluster_sizes['n_members'] >= large_cluster_threshold].copy()
small = cluster_sizes[cluster_sizes['n_members'] < large_cluster_threshold].copy()

# for large clusters assign them to a split according to how far off the ideal split value it is
targets = {'train': train_split_size, 'val': val_split_size, 'test': test_split_size}
current_members = {'train': 0, 'val': 0, 'test': 0}

for rep, n_members in zip(large['representative'], large['n_members']):
    total_so_far = sum(current_members.values()) + n_members
    # calculate how far off each split si off its target
    deficits = { split: targets[split] * total_so_far - current_members[split] for split in targets }
    # choose the one split with the highest offset
    chosen = max(deficits, key=deficits.get)
    # add the split name for this representative
    assignments[rep] = chosen
    # update split member count
    current_members[chosen] += n_members

# initialize the random number generator
rng = np.random.default_rng(1202)

# for small clusters group them by the bin label to ensure a stratified split
# not only on the cluster boundary but also over the bin label
for bin_name, bin_group in small.groupby('dominant_bin'):
    # get the representatives from this cluster
    reps = bin_group['representative'].values.copy()
    # shuffel the series
    rng.shuffle(reps)
    # number of series entries
    n = len(reps)
    # calculate the split boundaries
    cut_train = int(train_split_size * n)
    cut_val = int((train_split_size + val_split_size) * n)
    # add the representative splits to assignments dictionary
    for rep in reps[:cut_train]: assignments[rep] = 'train'
    for rep in reps[cut_train:cut_val]: assignments[rep] = 'val'
    for rep in reps[cut_val:]: assignments[rep] = 'test'


# create a split column by mapping the assigment dictionary to the clusters dataframe
clusters['split'] = clusters['representative'].map(assignments)

# validate that every row got a split assigned
n_missing = clusters['split'].isna().sum()
if n_missing:
    print(f'WARNING: {n_missing} rows have no split assignment (unassigned representative clusters)!')

# save the relevant columns to a new file
clusters[['member', 'split', 'ncbiTaxID_new', 'Temp_Duplicate_Average', 'bin_name']].to_csv(split_file, index=False)

# print a summary
print('Split sequence counts:')
print(clusters['split'].value_counts())
print('\nSplit bin label distribution (% per split):')
print(
    clusters.groupby('split')['bin_name']
    .value_counts(normalize=True)
    .unstack()
    .round(2)
)
