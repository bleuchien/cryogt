# CryOGT - Optimal Growth Temperature Prediction with Confidence Value for Psychrophiles from Protein Sequence

## Clone the Repo

```
git clone git@github.com:bleuchien/cryogt.git
cd cryogt
```

## Python Environment

Prepare and activate the Python virtual environment.

```
python -m venv venv
source venv/bin/activate
```

## Pre-Processing

1. (Optional) Remove everyting from */data/* but the manually pre-processed OGT dataset */growth_temp_dataset_manual_cleanup_final.csv/*.
2. Change directory ```cd preprocessing```.
3. Update taxonomy IDs ```python update_taxids.py```.
4. (Optional) The previous update might fail due to a bug in the RefSeq file because of rows with 39 columns instead of 38 (due to a tab in the name of an organization). In this case fix the file by running ```./fix_refseq.sh``` (removing the additional tab). Then re-run ```python update_taxids.py```.
5. Analyze the available organisms and select samples by running ```python select_samples.py```.
6. Now select the "best" proteome per organism and download it ```python proteome_download.py```.
7. Update the sequence headers of the downloaded proteomes ```python update_fasta_files.py```.
8. Take protein sequences samples from the downloaded proteomes ```python sample_proteomes.py```.
9. Run a similarity search to remove closely related sequences and another one to group the sequences for the upcoming split ```./similarity_search.sh | tee simsearch.log```.
10. Check the sequence clusters ```python analyze_clusters.py```.
11. Then create the train/test/validation splits ```python ttv_split.py```.
12. And analyze the outcome ```python analyze_ttv_split.py```.