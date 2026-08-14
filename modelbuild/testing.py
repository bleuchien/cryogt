import sys
import torch
import statistics
import argparse
import pandas as pd
import logging
import random
import numpy as np
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm.auto import tqdm
from config import Config
from build_utils import (
    PsychrophileDataset,
    PsychrophileCollator,
    ESMDoRA,
    train_one_epoch,
    evaluate,
    prepare_split_data, 
    download_model,
)

# test a deep ensemble of adapters and heads and save the output

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
file_handler = logging.FileHandler(f'testing_{timestamp}.log')
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

logger.info('Performing some sanity checks.')

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

# prepare datasets
logger.info('Preparing testing dataset.')
sequences, ogts = prepare_split_data(df, 'test', config.paths.proteomes_dir)
# mean_ogt = statistics.mean(ogts)
# logger.info(f'Testing set mean OGT: {mean_ogt:.1f}°C.')
test_dataset = PsychrophileDataset(
    sequences,
    ogts,
    tokenizer,
    config.training.max_length,
)

logger.info(f'Testing dataset has {len(test_dataset)} entries.')

# custom collator for dynamic batch padding and mask creation
collator = PsychrophileCollator(tokenizer=tokenizer)

# create the dataloaders
test_loader = DataLoader(
    test_dataset,
    batch_size=config.training.batch_size, 
    shuffle=False, 
    collate_fn=collator,
    # num_workers=4,                                  # parallel dataloading (more efficient but bad for debugging)
    # pin_memory=True,                                # PyTorch recommendation for parallel dataloading
)

# PyTorch accelerator device setup
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
logger.info(f'Using {device} device for tensor calculation acceleration.')

# sanity check
if len(config.testing.adapters) != len(config.testing.heads):
    logger.error('Adapter and head configuration list length differ. Aborting!')
    sys.exit(1)

for adapter in config.testing.adapters:
    adapter_path = Path(config.paths.adapter_dir) / f'adapter_{adapter}'
    if not adapter_path.exists():
        logger.error(f'Adapter {adapter_path} does not exist. Aborting!')
        sys.exit(1)

for head in config.testing.heads:
    head_path = Path(config.paths.model_dir) / f'head_{head}'
    if not head_path.exists():
        logger.error(f'Head {head_path} does not exist. Aborting!')
        sys.exit(1)

ensemble_ogts = []
ensemble_log_vars = []

# for each adapter
for adapter, head in zip(config.testing.adapters, config.testing.heads):
    adapter_path = Path(config.paths.adapter_dir) / f'adapter_{adapter}'
    head_path = Path(config.paths.model_dir) / f'head_{head}'
    logger.info(f'Processing adapter {adapter_path} and head {head_path}.')

    # model setup
    logger.info('Setting up model.')
    model = ESMDoRA(
        esm_model_name=full_model_path,
        head_hidden_dims=config.head.hidden_layers,
        head_dropout=config.head.dropout,
        layer_norm=config.head.layer_norm,
        log_var_min=config.head.log_var_min,
        log_var_max=config.head.log_var_max,
        # mean_out_bias_init=mean_ogt,
        dora_r=config.esmdora.dora_r,
        dora_alpha=config.esmdora.dora_alpha,
        dora_dropout=config.esmdora.dora_dropout,
        target_modules=config.esmdora.target_modules,
        gradient_checkpointing=False,
        adapter_path=adapter_path,
        adapter_trainable=False,
        head_path=head_path
    )

    # move model to the accelerator
    model.to(device)

    # evaluation mode
    model.eval()

    all_ogts = []
    all_log_vars = []

    # predict OGTs for all testing entries
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            residue_mask = batch['residue_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
            )

            all_ogts.append(outputs['mu'].float().cpu())
            all_log_vars.append(outputs['log_var'].float().cpu())

    ogts = torch.cat(all_ogts, dim=0)
    log_vars = torch.cat(all_log_vars, dim=0)

    ensemble_ogts.append(ogts)
    ensemble_log_vars.append(log_vars)

    # clean up memory
    del model
    gc.collect()

    if device == 'cuda':
        torch.cuda.empty_cache()

# combine the ensemble output
ogts = torch.stack(ensemble_ogts, dim=0)
log_vars = torch.stack(ensemble_log_vars, dim=0)

vars_exp = torch.exp(log_vars)

ensemble_ogts = ogts.mean(dim=0)

aleatoric_var = vars_exp.mean(dim=0)
epistemic_var = ogts.var(dim=0, unbiased=False)

ensemble_var = aleatoric_var + epistemic_var
ensemble_log_var = torch.log(ensemble_var.clamp_min(1e-12))
ensemble_std = torch.sqrt(ensemble_var)
aleatoric_std = torch.sqrt(aleatoric_var)
epistemic_std = torch.sqrt(epistemic_var)


# output file path and name
outfile = Path(config.paths.data_dir) / 'prediction.csv'

# build a small dataframe of (member, prediction) using the current test order
full_model_name = config.model.name + '_head_' + config.head.name
test_df = df[df['split'] == 'test'].reset_index(drop=True)
pred_df = pd.DataFrame({
    'member': test_df['member'], 
    full_model_name: ensemble_ogts.numpy(),
    'log_var_' + full_model_name: ensemble_log_var.numpy(),
    'std_' + full_model_name: ensemble_std.numpy(),
    'aleatoric_std_' + full_model_name: aleatoric_std.numpy(),
    'epistemic_std_' + full_model_name: epistemic_std.numpy()
    })

if outfile.exists():
    logger.info(f'Output file already exists. Updating {full_model_name} column.')
    out_df = pd.read_csv(outfile)
    # drop the column if it already exists
    out_df = out_df.drop(columns=[
        full_model_name, 
        'log_var_' + full_model_name, 
        'std_' + full_model_name, 
        'aleatoric_std_' + full_model_name, 
        'epistemic_std_' + full_model_name
        ], errors='ignore')
else:
    # pepare new dataframe
    column_list = ['member', 'ncbiTaxID_new', 'Temp_Duplicate_Average', 'bin_name']
    out_df = test_df[column_list]

# merge on 'member' in case the row order is different
out_df = out_df.merge(pred_df, on='member', how='left', validate='one_to_one')

# sanity check for missing values
n_missing = out_df[full_model_name].isna().sum()
if n_missing:
    logger.error(f'{n_missing} rows failed to align on "member" — check split file consistency!')
    sys.exit(1)

logger.info(f'Saving dataframe to {outfile}.')
out_df.to_csv(outfile, index=False)