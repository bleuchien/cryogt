from pathlib import Path
import pandas as pd

from ncbi_utils import (
    download_file,
    ensure_refseq_assembly_summary,
    load_assembly_summary,
    pick_best_assemblies,
    proteome_url_from_ftp_path,
    download_proteomes
)

# download individual proteome files for the selected organisms from NCBI RefSeq

# data directory
data_dir = Path('../data')

# stratified organism selection
input_csv = data_dir / 'growth_temp_dataset_selection.csv'
taxid_col = 'ncbiTaxID_new'

# directory to store proteoms in
proteome_dir = data_dir / 'proteomes'

# log files
mapping_out = data_dir / 'selected_best_assemblies.csv'
download_log_out = data_dir / 'proteome_downloads.csv'
missing_out = data_dir / 'missing_refseq_assemblies.csv'

# 
# main script
#

# create directorie if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
proteome_dir.mkdir(parents=True, exist_ok=True)

# read selection file
df = pd.read_csv(input_csv)
taxids = set(df[taxid_col].astype(int).unique())

# load refseq assembly file
summary_path = ensure_refseq_assembly_summary(data_dir, max_age_days=30)
print(f'Using RefSeq assembly summary: {summary_path}')

print('Parsing assembly summary...')
asm = load_assembly_summary(summary_path)

print('Selecting best assembly per TaxID...')
best = pick_best_assemblies(asm, taxids)
best.to_csv(mapping_out, index=False)

print(f'Wrote mapping: {mapping_out}')
print(f'TaxIDs requested: {len(taxids)} | assemblies found: {len(best)}')

# separate found and missing
found_taxids = set(best['input_taxid'].astype(int))
missing_taxids = sorted(taxids - found_taxids)

missing_df = (
    df.assign(**{taxid_col: df[taxid_col].astype(int)})
      .loc[lambda x: x[taxid_col].isin(missing_taxids)]
      .copy()
)
missing_df.to_csv(missing_out, index=False)

if missing_taxids:
    print(f'WARNING: {len(missing_taxids)} TaxIDs had no RefSeq assembly.')
    print(f'Wrote missing-organisms log (original dataframe rows): {missing_out}')
    print(f'Example missing TaxIDs: {missing_taxids[:10]}')
else:
    print('All TaxIDs had at least one RefSeq assembly.')

print('Downloading proteomes...')
downloads = download_proteomes(best, proteome_dir)
downloads.to_csv(download_log_out, index=False)

print(f'Wrote download log: {download_log_out}')
print('Download summary:')
print(downloads['status'].value_counts(dropna=False).to_string())