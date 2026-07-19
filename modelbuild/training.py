import sys
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from config import Config
from build_utils import (
    PsychrophileDataset, 
    prepare_split_data, 
    download_model,
)

# this script fine-tunes ESM-2 and trains the regression head

# read configuration
config = Config.from_yaml('config.yaml')

print('Performing some sanity checks.')

# check proteomes directory existence
proteomes_dir = Path(config.paths.proteomes_dir)
if not proteomes_dir.exists():
    print(f'{proteomes_dir} does not exist!')
    sys.exit(1)

# check split file existence
split_file = Path(config.paths.split_file)
if not split_file.exists():
    print(f'{split_file} does not exist!')
    sys.exit(1)

# check model directory
full_model_dir = Path(config.paths.model_dir) / config.model.name
if not full_model_dir.exists():
    download_model(config.model.name, full_model_dir)

print(f'Using model: {config.model.name}.')

# read split file
print(f'Reading splits file: {split_file}.')
df = pd.read_csv(split_file)

# create the tokenizer from the configured model
tokenizer = AutoTokenizer.from_pretrained(Path(config.paths.model_dir) / config.model.name)

# prepare datasets
print('Preparing training dataset.')
train_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'train', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

print(f'Training dataset has {len(train_dataset)} entries.')

print(train_dataset[0])

print('Preparing validation dataset.')
val_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'val', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

print(f'Validation dataset has {len(val_dataset)} entries.')

print(val_dataset[0])

# use the HuggingFace collator with the created tokenizer
# using the collator is more efficient than pre-processing the whole dataset in one step
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# create the dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size, 
    shuffle=True, 
    collate_fn=collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size, 
    shuffle=True, 
    collate_fn=collator
)