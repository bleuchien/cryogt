import sys
import argparse
import pandas as pd
import logging
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from datetime import datetime
from config import Config
from build_utils import (
    PsychrophileDataset,
    PsychrophileCollator,
    prepare_split_data,
    download_model,
)

# this script uses the vanilla ESM without head to create an embedding
# then uses the embedding to train the MLP to predigt the OGT

# mean pooling ovcer the amino acid residues only
def pool_mean(self, last_hidden_state, residue_mask):
    # mean pooling with higher precision
    mask = residue_mask.unsqueeze(-1).float()
    summed = (last_hidden_state.float() * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)

    return (summed / denom).to(last_hidden_state.dtype)

# current time stamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# setup logging to console and file
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, 'file_only', False)

file_only = {'file_only': True}
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(f'training_{timestamp}.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.addFilter(ConsoleFilter())
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# parse command line arguments
parser = argparse.ArgumentParser(prog='CryOGT model training')
parser.add_argument('-c', '--config', help='Configuration file.', default='config.yaml')
args = parser.parse_args()

# sanity check for the config file
config_path = Path(args.config)
if not config_path.exists():
    logger.error(f'Config file {config_path} does not exit!')
    sys.exit(1)

# read configuration
config = Config.from_yaml(config_path)

# check proteomes directory existence
proteomes_dir = Path(config.paths.proteomes_dir)
if not proteomes_dir.exists():
    logger.error(f'{proteomes_dir} does not exist!')
    sys.exit(1)

# check split file existence
split_file = Path(config.paths.split_file)
if not split_file.exists():
    logger.error(f'{split_file} does not exist!')
    sys.exit(1)

# check model directory
full_model_path = Path(config.paths.model_dir) / config.model.name
if not full_model_path.exists():
    download_model(config.model.name, full_model_path)

logger.info(f'Using model: {config.model.name}.')

# read split file
logger.info(f'Reading splits file: {split_file}.')
df = pd.read_csv(split_file)

# create the tokenizer from the configured model
tokenizer = AutoTokenizer.from_pretrained(full_model_path)

train_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'train', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

logger.info(f'Training dataset has {len(train_dataset)} entries.')

logger.info('Preparing validation dataset.')
val_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'val', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

logger.info(f'Validation dataset has {len(val_dataset)} entries.')

# custom collator for dynamic batch padding and mask creation
collator = PsychrophileCollator(tokenizer=tokenizer)

# create the dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size, 
    shuffle=False, 
    collate_fn=collator,
    # num_workers=4,                                  # parallel dataloading (more efficient but bad for debugging)
    # pin_memory=True,                                # PyTorch recommendation for parallel dataloading
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size, 
    shuffle=False, 
    collate_fn=collator,
    # num_workers=4,                                  # parallel dataloading (more efficient but bad for debugging)
    # pin_memory=True,                                # PyTorch recommendation for parallel dataloading
)

# PyTorch accelerator device setup
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
logger.info(f'Using {device} device for tensor calculation acceleration.')


# load the HuggingFace ESM model
esm = AutoModel.from_pretrained(config.model.name, dtype=torch.bfloat16)
# send model to accelerator
esm = esm.to(device)
# turn on evaluation mode
esm.eval()

all_embeddings = []
all_ogts = []

with torch.inference_mode():
    for batch in tqdm(train_loader, desc='Generating ESM embeddings'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        residue_mask = batch['residue_mask'].to(device)
        labels = batch['labels']

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pooled = pool_mean(
            outputs.last_hidden_state,
            residue_mask,
        )

        all_embeddings.append(pooled.cpu())
        all_ogts.append(labels.float().cpu())

embeddings = torch.cat(all_embeddings, dim=0)
ogts_tensor = torch.cat(all_ogts, dim=0)