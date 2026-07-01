import pandas as pd
from pathlib import Path
from ncbi_utils import (
    ensure_refseq_assembly_summary,
    load_assembly_summary,
    pick_best_assemblies
)

#
# this program analyzes the OGT sample file with protein counts, selects samples and saves them to a new file for further processsing
#

# SCRIPT FUNCTION
# - load csv
# - remove entries without phyum_id
# - verify the number of psychrophiles Temp_Duplicate < 15 and protein_count > 20 -> make this the sample number for each bin
# - binning 
#     psychrophiles (-10, 15) -> take all
#     mesophiles
#       (15, 25)
#       (25, 35)
#       (35, 45)
#       (45, 60)
#     thermophiles (60, 80)
#     hyperthermophiles (80, 200)
# - within each bin group by phylum_id -
# - select proportionally to the max sample count per bin from each phylum
# - create a new dataframe

# temperature bin configuration list
# tuple of low and high temperature value and bin name
# low will be included, high will be excluded
bin_config = [
    (-10, 15, 'psychrophiles'),
    (15, 25, 'mesophiles bin 1'),
    (25, 35, 'mesophiles bin 2'),
    (35, 45, 'mesophiels bin 3'),
    (45, 60, 'mesophiles bin 4'),
    (60, 80, 'thermophiles'),
    (80, 120, 'hyperthermophiles')
]

# main data directory
data_dir = Path('../data')

# data file to read
data_in_file = data_dir / 'growth_temp_dataset_manual_cleanup_final_updated.csv'

# data file to write
data_out_file = data_dir / 'growth_temp_dataset_selection.csv'

# read data file
df = pd.read_csv(data_in_file)
print(f'{len(df)} entried read from {data_in_file}.')

# remove entries with no phylum_id
mask = df['phylum_id'].isna() | df['phylum_id'].astype(str).str.strip().eq('')
rem_count_no_phylum = int(mask.sum())
df = df.drop(index=df.index[mask])
print(f'Removed {rem_count_no_phylum} entries without phylum ID.')

print('Cross-referencing dataset taxonomy IDs with RefSeq Assembly Summary.')
taxids = set(df['ncbiTaxID_new'].astype(int).unique())

summary_path = ensure_refseq_assembly_summary(data_dir)
asm = load_assembly_summary(summary_path)
best_assemblies = pick_best_assemblies(asm, taxids)

valid_taxids = set(best_assemblies['input_taxid'].unique())
missing_count = len(taxids) - len(valid_taxids)

# filter dataset to only include organisms with downloadable RefSeq genomes
df = df[df['ncbiTaxID_new'].isin(valid_taxids)].copy()
print(f'Kept {len(valid_taxids)} organisms with valid RefSeq assemblies.')
print(f'Dropped {missing_count} organisms missing from RefSeq.')

# print('The first five rows of the dataset:')
# print(df.head())

# sample count set from the number of psychrophiles
s_count = len(df[(df['Temp_Duplicate_Average'] >= bin_config[0][0]) & (df['Temp_Duplicate_Average'] < bin_config[0][1])])

# all psychrophiles
# print(df[(df['Temp_Duplicate_Average'] >= bin_config[0][0]) & (df['Temp_Duplicate_Average'] < bin_config[0][1])])

print(f'Sample count set to {s_count}.')

# list of selected organisms
selected_parts = []

for b in bin_config:
    t_low, t_high, bin_name = b
    print(f'Working on bin {bin_name} ({t_low}°C <= X < {t_high}°C).')

    # select the given temperature bin
    df_bin = df[(df['Temp_Duplicate_Average'] >= t_low) & (df['Temp_Duplicate_Average'] < t_high)]

    if df_bin.empty: continue

    # group by phylum and proportionally to the sample count select random samples, also add the bin name
    selected = (df_bin
        .groupby("phylum", group_keys=False)
        .apply(lambda x: x.sample(
            min(
                len(x),
                max(1, round(s_count * len(x) / len(df_bin)))
            ),
            random_state=1202)
        )
        .head(s_count)
        .assign(bin_name=bin_name)
        )
    
    selected_parts.append(selected)

# combine the selection of all bins
df_selected = pd.concat(selected_parts, ignore_index=True)

# write to a new CSV file
df_selected.to_csv(data_out_file, index=False)

print(f'Stratified selection saved to {data_out_file}.')