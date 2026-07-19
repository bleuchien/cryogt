import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from huggingface_hub import snapshot_download
from typing import Literal
from pathlib import Path
import pandas as pd
from Bio import SeqIO

# dataset class for PyTorch ML input
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class PsychrophileDataset(Dataset):
    def __init__(
            self,
            sequences: list[str],           # protein sequences as list of strings
            ogt_values: list[float],        # OGT values as floats
            tokenizer: AutoTokenizer,       # model tokenizer
            max_length: int = 1024,         # maximum input length
        ):
        self.ogt_values = ogt_values

        # tokenizing the sequence
        # no padding yet to safe space
        # but build special token mask for amio acid centric mean pooling later
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_special_tokens_mask=True,
        )

        # assign the tokenizer output
        self.input_ids = tokenized['input_ids']
        self.special_tokens_mask = tokenized['special_tokens_mask']

    def __len__(self):
        return len(self.ogt_values)

    def __getitem__(self, idx):
        # return one specific item from the list
        return {
            'input_ids': self.input_ids[idx],
            'special_tokens_mask': self.special_tokens_mask[idx],
            'label': self.ogt_values[idx],
        }

# customized data collator for dynamic padding and mask creation
class PsychrophileCollator:
    def __init__(self, tokenizer):
        # use HuggingFace collator for padding
        self.hf_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors='pt',
        )

    def __call__(self, examples):
        batch = self.hf_collator(examples)

        # ensure the datatype of the OGT (label) is float
        batch['labels'] = batch['labels'].float()

        # create a "residue mask" for mean pooling of amino acids only
        batch['residue_mask'] = (
            batch['attention_mask'].bool()
            & ~batch['special_tokens_mask'].bool()
        )

        return batch

# download the specified model from huggingface to the specified directory
def download_model(
        model: str,                         # model name
        local_dir: Path                     # path to store the model in
    ):

    print(f'Downloading {model} to {local_dir}.')
    snapshot_download(
        repo_id=model,
        local_dir=local_dir,
        # ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.safetensors"],  # Optional: skip large files if needed
        force_download=False,
    )
    print(f'Finished downloading {model}.')

# prepare the sequences and OGT values for the given split
def prepare_split_data(
        df: pd.DataFrame,                           # train/test/val split dataframe
        split: Literal['train', 'test', 'val'],     # choice of split from the list
        proteomes_dir: Path,                        # directory of the proteome files
    ) -> tuple[list[str], list[float]]:             # returns a list of sequences and corresponding OGTs

    # only access the required split
    df = df[df['split'] == split].copy()

    # sanity check of the dataframe
    if df.empty:
        raise ValueError(f'No rows found for split "{split}"')
    
    # group the dataframe by taxonomy ID
    # only select the members and convert the to a set (there should not be duplicates but this would remove them)
    # and the create a dictionary from the result { taxonomy ID: { set of members } }
    needed_by_taxid = df.groupby('ncbiTaxID_new')['member'].apply(set).to_dict()

    # initizlize empty dictionary for the found sequences per member (key: string and value: string)
    sequence_lookup: dict[str, str] = {}

    # loop over all dictionary entries
    for taxid, members in needed_by_taxid.items():
        # proteome filenames start with the taxonomy ID and ends with _protein.faa
        matches = list(Path(proteomes_dir).glob(f'{str(taxid)}_*_protein.faa'))

        # sanity check for file existence
        if not matches:
            raise FileNotFoundError(f'No proteome file found for taxid {taxid} in {proteomes_dir}')

        # sanity check in case there are multiple files starting with this taxonomy ID
        if len(matches) > 1:
            print(f'Warning: {len(matches)} proteome files found for taxid {taxid} using {matches[0].name}')

        found = 0
        # parse FASTA file
        for record in SeqIO.parse(str(matches[0]), 'fasta'):
            # check each sequence record if it's in the members set
            # record.id is ie 1092|WP_012467398.1
            if record.id in members:
                # store the found sequence 
                sequence_lookup[record.id] = str(record.seq)
                found += 1
                # early stop if all members are found
                if found == len(members):
                    break

    # map sequences to dataframe rows preserving the original order
    mapped = df['member'].map(sequence_lookup)

    # sanity check for entries without sequence (missing values)
    missing = mapped.isna().sum()
    if missing > 0:
        raise ValueError(f'{missing} member sequences not found in proteome files.')

    # produce lists of the found sequences and OGT values
    sequences = mapped.tolist()
    ogts = df['Temp_Duplicate_Average'].astype(float).tolist()

    return sequences, ogts

