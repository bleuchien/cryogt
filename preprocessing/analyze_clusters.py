from pathlib import Path
import sys
import pandas as pd
from Bio import SeqIO

# analyze the protein sequence clusters

# number of rows to show
examples = 40

# main data directory
data_dir = Path('../data')

# protein clusters file
cluster_file = data_dir / 'mmseqs/stage2/split_clusters.tsv'
# sequences file
fasta_file = data_dir / 'mmseqs/stage1/representatives.faa'

# check if file exists
for f in [cluster_file, fasta_file]:
    if not f.exists():
        print(f'{f} does not exist.')
        sys.exit(1)

print(f'Analyzing clusters from {cluster_file}.')
print(f'Using sequences from {fasta_file}.')

# read clusters file
clusters = pd.read_csv(cluster_file, sep='\t', header=None, names=['representative', 'member'])

# unique clusters
unique_clusters = clusters['representative'].unique()

# how many members does each cluster have, sorted large to small
cluster_sizes = (
    clusters.groupby("representative", sort=False)["member"]
    .size()
    .rename("n_members")
    .sort_values(ascending=False)
    .reset_index()
)

# member statistics
member_mean = float(cluster_sizes['n_members'].mean())
member_std = float(cluster_sizes['n_members'].std())
member_median = float(cluster_sizes['n_members'].median())

print('\nCluster member count distribution:')
print(f'Number of clusters n={len(cluster_sizes)},  Number of members mean={member_mean:.2f}, std={member_std:.2f}, median={member_median:.2f}')

print(f'\nTop {examples} clusters by size:')
print(cluster_sizes.head(examples).to_string(index=False))

# how long is the average sequence in each cluster and what is the standard deviation
lengths = {}
for rec in SeqIO.parse(fasta_file, 'fasta'):
    lengths[rec.id] = len(rec.seq)

# attach length to each member row
clusters['member_len'] = clusters['member'].map(lengths)

# sanity check
missing = int(clusters['member_len'].isna().sum())
if missing:
    print(f'{missing} members not found in FASTA length map.')

# sequence length statistics
cluster_len_stats = (
    clusters.groupby('representative')['member_len']
    .agg(mean_len='mean', std_len='std', n_members='size')
    .reset_index()
)

# std is NaN for singletons; optionally set to 0.0
cluster_len_stats['std_len'] = cluster_len_stats['std_len'].fillna(0.0)

# sort by largest clusters first
cluster_len_stats = cluster_len_stats.sort_values('n_members', ascending=False)

# over all stats
all_lengths = pd.Series(list(lengths.values()), name='length')
overall_mean = float(all_lengths.mean())
overall_std = float(all_lengths.std())

print('\nOverall sequence length stats:')
print(f'Number of sequnces n={len(all_lengths)}, Sequence length mean={overall_mean:.2f}, std={overall_std:.2f}')

print(f'{len(clusters)} sequences in {len(unique_clusters)} unique clusters.')
print(f'\nTop {examples} clusters (size + mean/std length):')
print(cluster_len_stats.head(examples).to_string(index=False))

# plot a graph?