import pandas as pd

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
#   - with all columns or just a selection?

# temperature bin configuration list
# tuple of low and high temperature value
# low will be included, high will be excluded
bin_config = [
    (-10, 15),    # psychrophiles
    (15, 25),     # mesophiles bin 1
    (25, 35),     # mesophiles bin 2
    (35, 45),     # mesophiles bin 3
    (45, 60),     # mesophiles bin 4
    (60, 80),     # thermophiles
    (80, 120)     # hyperthermophiles
]

# data file to read
data_in_file = '../data/growth_temp_dataset_manual_cleanup_final_with_counts.csv'

# data file to write
data_out_file = '../data/growth_temp_dataset_selection.csv'

# protein count threshold
prot_threshold = 20

# read data file
df = pd.read_csv(data_in_file)
print(f'{len(df)} entried read from {data_in_file}.')

# remove entries with 0 protein count
mask = df['protein_count'] < prot_threshold
rem_count_low_prot = int(mask.sum())
df = df.drop(index=df.index[mask])
print(f'Removed {rem_count_low_prot} entries with protein count smaller than {prot_threshold}.')

# remove entries with no phylum_id
mask = df['phylum_id'].isna() | df['phylum_id'].astype(str).str.strip().eq('')
rem_count_no_phylum = int(mask.sum())
df = df.drop(index=df.index[mask])
print(f'Removed {rem_count_no_phylum} entries without phylum ID.')

print('The first five rows of the dataset:')
print(df.head())

for b in bin_config:
    print(f'Working on bin {b}.')
