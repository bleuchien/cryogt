from pathlib import Path
import gzip
import numpy as np
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

count = 0
skipped = 0
failed = 0

print('Sampling proteome fasta files.')

for file in sorted(proteomes_path.glob('*.faa.gz')):
    try:
        selection = process_proteome(file, rng, min_length=min_length, max_length=max_length, n_samples=n_samples)
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
