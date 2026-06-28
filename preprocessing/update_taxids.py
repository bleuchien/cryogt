from pathlib import Path
import pandas as pd

from ncbi_utils import (
    ensure_taxdump_files,
    load_merged,
    load_delnodes,
    ensure_refseq_assembly_summary,
    load_assembly_summary,
    refseq_available_taxids,
    resolve_merge_chain,
)

# this script is the result of problems downloading proteoms for specific taxonomy IDs
# some have been merged while others do not exist any more
# NCBI provides information on what ID has been merged with which other and what has been dumped
# this information is used to update the growth temp dataset to only contain entries
# with updated taxonomy ID and available RefSeq entries

# main data directory
data_dir = Path('../data')

# in and output file names
data_in_file = data_dir / 'growth_temp_dataset_manual_cleanup_final_with_counts.csv'
data_out_file = data_in_file.with_name(data_in_file.stem + '_updated.csv')
dropped_refseq_missing_out = data_in_file.with_name(data_in_file.stem + '_refseq_missing_dropped.csv')

taxid_col = 'ncbiTaxID_new'

#
# main script
#  

# read growth temp dataset
df = pd.read_csv(data_in_file)

# get some statistical values for the summary
n_rows_before = len(df)
n_taxids_before = df[taxid_col].nunique()

# update the dataframe according to the merged and deleted data from NCBI

# load the NCBI dump file
_, merged_path, delnodes_path = ensure_taxdump_files(data_dir, max_age_days=30)
merged_df = load_merged(merged_path)
delnodes_set = load_delnodes(delnodes_path)
# build mapping from old to new taxonomy ID
merge_map = dict(zip(merged_df['old_taxid'].tolist(), merged_df['new_taxid'].tolist()))

# store original ID in a new column
df['taxid_original'] = df[taxid_col]
# add new columns if the ID was merged or deleted
df['taxid_was_merged'] = df[taxid_col].isin(merge_map)
df['taxid_was_deleted'] = df[taxid_col].isin(delnodes_set)

# update the ID according to the merge map
df[taxid_col] = df[taxid_col].map(lambda x: resolve_merge_chain(x, merge_map)).astype(int)

# add new column for delted IDs after the ID update
df['taxid_is_deleted_after_update'] = df[taxid_col].isin(delnodes_set)

deleted_rows = int(df['taxid_is_deleted_after_update'].sum())
df_updated = df.loc[~df['taxid_is_deleted_after_update']].copy()

# remove entries from the dataframe that do not have a RefSeq entry

# load the refseq assembly file
summary_path = ensure_refseq_assembly_summary(data_dir, max_age_days=30)
asm = load_assembly_summary(summary_path)

# create a list of all IDs in dataset
taxids_after_taxonomy_update = set(df_updated[taxid_col].astype(int).unique())
# create a list of all IDs from the dataset in the assembly list
taxids_with_refseq = refseq_available_taxids(asm, taxids_after_taxonomy_update)

# check if each dataset ID is available in the assembly
df_updated['has_refseq_assembly'] = df_updated[taxid_col].isin(taxids_with_refseq)

# get the number of missing and the missing IDs
refseq_missing_rows = int((~df_updated['has_refseq_assembly']).sum())
refseq_missing_taxids = sorted(set(df_updated.loc[~df_updated['has_refseq_assembly'], taxid_col].astype(int)))

# get a dataframe with only the missing
df_refseq_missing = df_updated.loc[~df_updated['has_refseq_assembly']].copy()
df_refseq_missing.to_csv(dropped_refseq_missing_out, index=False)

# only retain the entries that have a matching assembly entry
df_updated = df_updated.loc[df_updated['has_refseq_assembly']].copy()

# drop duplicate rows (exact duplicates)
df_updated = df_updated.drop_duplicates(keep='first')

# save the updated dataframe
df_updated.to_csv(data_out_file, index=False)

# summary
n_rows_after = len(df_updated)
n_taxids_after = df_updated[taxid_col].nunique()

merged_rows = int(df['taxid_was_merged'].sum())
merged_unique_old = int(df.loc[df['taxid_was_merged'], 'taxid_original'].nunique())
merged_unique_new = int(df.loc[df['taxid_was_merged'], taxid_col].nunique())

deleted_unique_before = int(df.loc[df['taxid_was_deleted'], 'taxid_original'].nunique())

print('Summary')
print('-------')
print(f'input file:   {data_in_file}')
print(f'output file:  {data_out_file}')
print(f'rows:         {n_rows_before} -> {n_rows_after} (dropped {n_rows_before - n_rows_after})')
print(f'unique taxids:{n_taxids_before} -> {n_taxids_after}')
print(f'merged rows:  {merged_rows}')
print(f'merged taxids (unique old -> unique new): {merged_unique_old} -> {merged_unique_new}')
print(f'deleted/obsolete taxids present in input (unique): {deleted_unique_before}')
print(f'rows dropped because updated taxid is deleted/obsolete: {deleted_rows}')
print(f'rows dropped because updated taxid has no refseq assembly: {refseq_missing_rows}')
print(f'unique taxids dropped because no refseq assembly: {len(refseq_missing_taxids)}')
print(f'wrote dropped refseq-missing rows to: {dropped_refseq_missing_out}')

# if refseq_missing_taxids:
#     print(f'example refseq-missing taxids: {refseq_missing_taxids[:10]}')