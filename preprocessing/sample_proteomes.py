import sys
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO

# the task is to randomly sample sample_count proteins per organism and store the output in a new file
# - Bio Python to create a dictionary of the sequences per organism
# - filter for length
#   - cutoffs 50aa < sample < 1022
#     <50 is signal peptides, fragments, etc. -> cite?
#     ESM-2 has an input window of 1024 which includes <cls> and <eos> tokens

# main data directory
data_dir = Path('../data')

# path to the proteomes
proteomes_path = data_dir / 'proteomes'

# the results of the selection
proteomes_selected = data_dir / 'proteomes_sampled'

# create directories if they don't exist
proteomes_selected.mkdir(parents=True, exist_ok=True)

# organism selection CSV
selection_file = data_dir / 'growth_temp_dataset_selection.csv'

# sample length and count
min_length = 50
max_length = 1022
n_samples = 500

# single, reproducible RNG shared across all files
rng = np.random.default_rng(1202)

def process_proteome(fasta_file, rng, min_length=50, max_length=1022, n_samples=500):
    # read the gzipped fasta file and store as dictionary
    with gzip.open(fasta_file, 'rt') as handle:
        proteome = SeqIO.to_dict(SeqIO.parse(handle, 'fasta'))

    # filter by length
    filtered = { pid: record for pid, record in proteome.items() if min_length <= len(record.seq) <= max_length }

    n_available = len(filtered)

    if n_available < 50:
        print(f'WARNING: {fasta_file.name} has only {n_available} sequences after filtering -> IGNORED!')
        return None

    if n_available < n_samples:
        print(f'INFO: {fasta_file.name} has only {n_available} sequences after filtering -> ALL SAMPLED!')

    # sample randomly
    keys = sorted(filtered.keys())
    n_draw = min(n_samples, n_available)
    sampled_keys = rng.choice(keys, size=n_draw, replace=False)

    return [filtered[k] for k in sampled_keys]

def write_samples(records, output_path):
    SeqIO.write(records, output_path, 'fasta')

def scaled_sample_count(bin_name, total_samples=275000, organism_count=77):
    # bin scaling factors
    scaling_factors = {
        'psychrophiles': 4,
        'mesophiles bin 1': 0.2,
        'mesophiles bin 2': 0.2,
        'mesophiles bin 3': 0.1,
        'mesophiles bin 4': 0.1,
        'thermophiles': 0.2,
        'hyperthermophiles': 0.1
    }

    # get the sum of the scaling factors
    total_scaling = sum(scaling_factors.values())
    if total_scaling == 0:
        print('Sum of scaling factors is 0. Aborting!')
        sys.exit(1)

    # scale the bin counts to reach the over all sample goal
    bin_counts = { bin: int(total_samples * (scale / total_scaling)) for bin, scale in scaling_factors.items() }

    if bin_name not in bin_counts:
        print(f'Did not find the requested scaling factor for {bin_name}.')

    # return the scaled bin count or 0 by default
    return int(bin_counts.get(bin_name, 0) / organism_count)

count = 0
skipped = 0
failed = 0

# sanity check
if not selection_file.exists():
    print(f'Could not find selection file {selection_file}.')
    sys.exit(1)

print('Reading selection file.')
selection_df = pd.read_csv(selection_file)

# clean bin name typo
selection_df['bin_name'] = (
    selection_df['bin_name']
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace('mesophiels', 'mesophiles', regex=False)
)

# ensure string comparison works
selection_df['ncbiTaxID_new'] = selection_df['ncbiTaxID_new'].astype(str)

print('Sampling proteome fasta files.')

for file in sorted(proteomes_path.glob('*.faa.gz')):
    # the first part of the file name is the taxonomy ID
    taxid = file.name.split('_', 1)[0]

    # find what bin this organism belongs to
    matches = selection_df.loc[selection_df['ncbiTaxID_new'] == taxid, 'bin_name']

    if matches.empty:
        print(f'TaxID {taxid} not found in selection file {file} -> IGNORED!')
        skipped += 1
        continue

    bin_name = matches.iloc[0]

    # the the scaled bin sample count
    sample_count = scaled_sample_count(bin_name)

    try:
        selection = process_proteome(file, rng, min_length=min_length, max_length=max_length, n_samples=sample_count)
    except (OSError, EOFError) as e:
        print(f'FAILED to read {file.name}: {e}')
        failed += 1
        continue

    if selection is None:
        skipped += 1
        continue

    write_samples(selection, proteomes_selected / file.stem)
    count += 1

print(f'{count} files sampled, {skipped} skipped, {failed} failed.')
