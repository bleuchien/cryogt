import pandas as pd
import requests
from tqdm import tqdm
import time

#
# (rough) script to query all entries for the number of associated protein sequences
#

# enable pandas progress bar
tqdm.pandas()

# read datasource
df = pd.read_csv('../data/growth_temp_dataset_manual_cleanup_final.csv')
#df = pd.read_csv('test.csv')

# query uniprot for the number of proteins
def get_protein_count(tax_id, max_retries=10):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=taxonomy_id:{tax_id}"
    for i in range(max_retries):
        try:
            response = requests.head(url, timeout=10)
            for header, value in response.headers.items():
                if header.lower() == 'x-total-results':
                    return int(value)
            return 0
        except Exception as e:
            print(f"Retrying {tax_id} due to {e}")
            time.sleep(2 * (i + 1))  # Wait before retrying
    return 0  # Give up after max_retries

# create a new column by running the method above with a progress bar
df['protein_count'] = df['ncbiTaxID_new'].progress_apply(get_protein_count)

# save output
df.to_csv('../data/growth_temp_dataset_manual_cleanup_final_with_counts.csv', index=False)
