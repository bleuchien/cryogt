# CryOGT - OGT Prediction for Psychrophiles with Confidence Value

## Python Environment
Prepare and activate the Python virtual environment.
```
python -m venv venv
source venv/bin/activate
```

## Data Pre-Processing
1. (Optional) Remove everyting from */data/* but the manually pre-processed OGT dataset */growth_temp_dataset_manual_cleanup_final.csv/*.
3. Run ```python update_taxids.py``` in ```preprocessing``` to update taxonomy IDs.
4. (Optional) The previous update might fail due to a bug in the RefSeq file because of rows with 39 columns instead of 38 (due to a tab in the name of an organization). In this case fix the file by running ```./fix_refseq.sh``` (removing the additional tab). Then re-run ```python update_taxids.py```.
5. Analyze the available organisms and select samples by running ```python select_samples.py```.
6. Select the "best" proteome per organism and download it ```python proteome_download.py```.
7. Update the sequence headers of the downloaded proteomes ```python update_fasta_files.py```.
8. Take protein sequences samples from the downloaded proteomes ```python sample_proteomes.py```.
9. Run a similarity search to remove closely related sequences and another one to group the sequences for the upcoming split ```./similarity_search.sh | tee simsearch.log```.
10. Check the sequence clusters ```python analyze_clusters.py```.
11. Create the train/test/validation splits ```python ttv_split.py```.
12. Analyze the outcome ```python analyze_ttv_split.py```.

## Baseline Generation and Analysis
1. Setup ```config_something.yaml``` in ```modelbuild```.
2. Create the embeddings for the configured model ```python prep_embeddings.py -c config_something.yaml```.
3. Train the configured heads ```python baseline.py -c config_something.yaml -i ../data/ttv_splits_with_embeddings.csv```.
4. Generate the analysis and graphics ```python baseline_analysis.py -c config_something.yaml```.

## Model Training, Testing and Analysis
1. Setup ```config_something.yaml``` in ```modelbuild```.
2. Train the configured models ```python training.py -c config_something.yaml```.
3. Test the configured models ```python testing.py -c config_something.yaml```.
4. Generate the analysis and graphics ```python baseline_analysis.py -c config_something.yaml```.
5. Prepare the calibration config and run ```python calibrate2.py --manifest ../data/calibration_manifest.csv --output-dir ../data/calibration2_psych099 --psych-weight 0.99``` to calibrate the system and analyze it.

Different sequence selection strategies are different branches in git.
