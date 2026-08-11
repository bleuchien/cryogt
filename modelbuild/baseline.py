import sys
import argparse
import pandas as pd
import logging
import torch
import random
import numpy as np
import statistics
import math
import ast
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from config import Config
from build_utils import (
    RegressionHead,
)

# this script uses the vanilla ESM embeddings created with prep_embedding.py
# then uses the embedding to train the MLP to predigt the OGT

# convert embedding stored in the CSV file into tensor
def parse_embedding(x):
    if isinstance(x, torch.Tensor):
        return x.float()

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()

    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)

    if isinstance(x, str):
        try:
            return torch.tensor(ast.literal_eval(x), dtype=torch.float32)
        except Exception:
            arr = np.fromstring(x.strip("[]").replace(",", " "), sep=" ")
            return torch.from_numpy(arr).float()

    raise TypeError(f'Unsupported embedding type: {type(x)}')

# dataset class for MLP input
class PsychrophileEmbeddingDataset(Dataset):
    def __init__(
            self,
            embeddings: list[str],          # protein sequences as list of strings
            ogt_values: list[float],        # OGT values as floats
        ):
        self.embeddings = torch.stack([ parse_embedding(e) for e in embeddings ])
        self.labels = torch.tensor(ogt_values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return one specific item from the list
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx],
        }

# Gaussian NLL function (based on the PyTorch source with (almost) the same name)
# using mu and log_var instead of var
def gaussian_nll_loss(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        y: torch.Tensor,
        full: bool = True,
    ):

    # make sure tensors are floats
    mu = mu.float()
    log_var = log_var.float()
    y = y.float()

    # loss calculation
    loss = 0.5 * (log_var + (y - mu).pow(2) * torch.exp(-log_var))

    # add the static term if necessary
    if full:
        loss += 0.5 * math.log(2.0 * math.pi)

    return loss.mean()

# one training run
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(
        training_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        model: torch.nn,
        epoch_index: int,
        tb_writer: SummaryWriter,
        device,
        log_every: int = 100
    ):
    model.train()

    total_loss = 0.0
    
    progbar = tqdm(
        training_loader,
        desc=f'Epoch {epoch_index + 1} train',
        position=1,
        leave=False,
        dynamic_ncols=True
    )

    # enumerate the training loader for more detailed reporting
    for i, batch in enumerate(progbar, start=1):
        # get data from batch and transfer it to accelerator
        x = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        # zero your gradients for every batch
        optimizer.zero_grad()

        # run input through the model
        mu, log_var = model(x)

        # calculate loss
        loss = gaussian_nll_loss(mu, log_var, labels)

        # compute gradients
        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # adjust learning weights
        optimizer.step()

        # reduce learning rate
        scheduler.step()

        loss_value = loss.item()
        total_loss += loss_value
        avg_loss = total_loss / i

        if i % log_every == 0 or i == 1:
            progbar.set_postfix({
                'batch_loss': f'{loss_value:.5f}',
                'avg_loss': f'{avg_loss:.5f}'
            })

            global_step = epoch_index * len(training_loader) + i
            tb_writer.add_scalar('Loss/train_batch', loss_value, global_step)

    # return average loss
    return total_loss / len(training_loader)

def evaluate(
        validation_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        epoch_index: int,
        device,
        overall_progbar=None
    ):
    model.eval()

    total_loss = 0.0

    progbar = tqdm(
        validation_loader,
        desc=f'Epoch {epoch_index + 1} val',
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    all_mu: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for i, batch in enumerate(progbar, start=1):
            x = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            # make predictions for this batch
            mu, log_var = model(x)
            loss = gaussian_nll_loss(mu, log_var, labels)
            loss_value = loss.item()

            all_mu.append(mu.detach().cpu())
            all_labels.append(labels.detach().cpu())

            total_loss += loss_value
            avg_loss = total_loss / i

            progbar.set_postfix({
                'val_loss': f'{avg_loss:.4f}',
            })

    # compute RMSE and MAE from accumulated predictions
    all_mu_t = torch.cat(all_mu).float()
    all_labels_t = torch.cat(all_labels).float()

    rmse = torch.sqrt(torch.mean((all_mu_t - all_labels_t) ** 2)).item()
    mae  = torch.mean(torch.abs(all_mu_t - all_labels_t)).item()

    return total_loss / len(validation_loader), rmse, mae

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
file_handler = logging.FileHandler(f'baseline_{timestamp}.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.addFilter(ConsoleFilter())
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# parse command line arguments
parser = argparse.ArgumentParser(prog='CryOGT baseline calculation')
parser.add_argument('-c', '--config', help='Configuration file.', default='config.yaml')
parser.add_argument('-i', '--input', default=None, help='TTV splits file with embeddings.')
parser.add_argument('-r', '--resume', metavar='STATE_FILE', help='Resume training with the given state file.')
parser.add_argument('-m', '--model', default=None, help='Model file to load for testing.')
args = parser.parse_args()

# sanity check for the config file
config_path = Path(args.config)
if not config_path.exists():
    logger.error(f'Config file {config_path} does not exit!')
    sys.exit(1)

# read configuration
config = Config.from_yaml(config_path)

# check split file existence
if args.input is not None:
    split_file = Path(args.input)
else:
    split_file = Path(config.paths.embedding_file)
if not split_file.exists():
    logger.error(f'{split_file} does not exist!')
    sys.exit(1)

# read split file
logger.info(f'Reading splits file: {split_file}.')
df = pd.read_csv(split_file)

# checking if the desired embedding column exists in the file
if not config.model.name in df.columns:
    logger.error(f'Column with embeddings for {config.model.name} not found in {split_file}. Try --input.')
    sys.exit(1)

logger.info('Preparing training dataset.')
ogts = df[df['split'] == 'train']['Temp_Duplicate_Average'].astype(float).to_list()
mean_ogt = statistics.mean(ogts)
logger.info(f'Training set mean OGT: {mean_ogt:.1f}°C.')
train_dataset = PsychrophileEmbeddingDataset(
    df[df['split'] == 'train'][config.model.name].to_list(),
    ogts
)
logger.info(f'Training dataset has {len(train_dataset)} entries.')

logger.info('Preparing validation dataset.')
val_dataset = PsychrophileEmbeddingDataset(
    df[df['split'] == 'val'][config.model.name].to_list(),
    df[df['split'] == 'val']['Temp_Duplicate_Average'].astype(float).to_list()
)
logger.info(f'Validation dataset has {len(val_dataset)} entries.')

logger.info('Preparing test dataset.')
test_dataset = PsychrophileEmbeddingDataset(
    df[df['split'] == 'test'][config.model.name].to_list(),
    df[df['split'] == 'test']['Temp_Duplicate_Average'].astype(float).to_list()
)
logger.info(f'Test dataset has {len(test_dataset)} entries.')

logger.info('Preparing dataloaders.')
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.training.batch_size,
    shuffle=False
)

# PyTorch accelerator device setup
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
logger.info(f'Using {device} device for tensor calculation acceleration.')

logger.info('Initializing random number generators.')
# initialize the random number generators
# to be on the save side not only for forch but also Python and numpy
run_seed = config.training.base_seed
random.seed(run_seed)
np.random.seed(run_seed)
torch.manual_seed(run_seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(run_seed)

logger.info('Setting up model.')
model = RegressionHead(
    input_dim=train_dataset.embeddings.shape[1],
    hidden_dims=config.head.hidden_layers,
    dropout=config.head.dropout,
    layer_norm=config.head.layer_norm,
    log_var_min=config.head.log_var_min,
    log_var_max=config.head.log_var_max,
    mean_out_bias_init=mean_ogt
)

# move model to the accelerator
model.to(device)

# setup optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.head_learning_rate,
    weight_decay=config.training.weight_decay
)

# setup scheduler (reducing the learning rate over time)
total_steps = config.training.epochs * len(train_loader)
warmup_steps = int(0.05 * total_steps)  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# initialize parameters for this loop
checkpoint = None
best_vloss = 1_000_000.                 # randomly high validation loss
patience = config.training.patience     # how long to run before early stopping is triggered
epochs_no_improve = 0                   # counting epoch without improvment
start_epoch = 0                         # starting the loop on this epoch
run_id = timestamp
best_model_path = Path(config.paths.model_dir) / f'mlponly_{run_id}.pt'
checkpoint_path = Path(config.paths.model_dir) / f'mlponly_training_state_{run_id}.pt'

# train when no model is given
if args.model is None:
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f'Training resume file {resume_path} does not exist!')
            sys.exit(1)

        # load saved checkpoint
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_vloss = checkpoint['best_vloss']
        epochs_no_improve = checkpoint['epochs_no_improve']

        run_id = checkpoint.get('timestamp', timestamp)
        best_model_path = Path(checkpoint.get('best_model_path', str(Path(config.paths.model_dir) / f'mlponly_{run_id}.pt')))
        checkpoint_path = Path(config.paths.model_dir) / f'mlponly_training_state_{run_id}.pt'

        logger.info(f'Resume training {start_epoch + 1}/{config.training.epochs}.')
    else:
        logger.info(f'Begin training MLP for {config.training.epochs} epochs.')

    # logger setup
    writer = SummaryWriter(Path(config.paths.data_dir) / f'summarywriter/mlponly_{run_id}')

    # start training loop
    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f'Training epoch {epoch + 1}.', extra=file_only)

        # train the model with the training set
        avg_loss = train_one_epoch(
            training_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            epoch_index=epoch,
            tb_writer=writer,
            device=device
        )

        # evaluate with the validation set
        avg_vloss, vrmse, vmae = evaluate(
            validation_loader=val_loader,
            model=model,
            device=device,
            epoch_index=epoch
        )

        tqdm.write(
            f'Epoch {epoch + 1}: '
            f'train_loss={avg_loss:.5f}, '
            f'val_loss={avg_vloss:.5f}, '
            f'RMSE={vrmse:.2f}°C, '
            f'MAE={vmae:.2f}°C'
        )

        logger.info(f'Epoch {epoch + 1}: train_loss={avg_loss:.5f}, val_loss={avg_vloss:.5f}, RMSE={vrmse:.2f}°C, MAE={vmae:.2f}°C', extra=file_only)

        # log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.add_scalar('RMSE/val', vrmse, epoch + 1)
        writer.add_scalar('MAE/val', vmae, epoch + 1)
        writer.flush()

        # track best performance, and save the model's adapter and regression head
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            epochs_no_improve = 0
            # save model
            torch.save(model.state_dict(), best_model_path)

            logger.info(f'Saved best performing MLP, epoch {epoch + 1}, file extension {run_id}.', extra=file_only)
        else:
            epochs_no_improve += 1

        # save training checkpoint
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_vloss': best_vloss,
                'epochs_no_improve': epochs_no_improve,
                'timestamp': timestamp,
                'best_model_path': str(best_model_path)
            },
            checkpoint_path
        )

        # stop training if there is no improvement
        if epochs_no_improve >= patience:
            tqdm.write(f'Early stopping triggered after {epochs_no_improve} epochs without improvement in epoch {epoch + 1}.')
            logger.info(f'Early stopping triggered after {epochs_no_improve} epochs without improvement in epoch {epoch + 1}.', extra=file_only)
            break

    logger.info('Training done.')
    writer.close()

logger.info('Testing the model.')
if args.model is not None:
    logger.info(f'Loading model from file {args.model}.')
    best_model_path = Path(args.model)
else:
    logger.info('Using best model from this training run.')

if best_model_path is None or not best_model_path.exists():
    logger.error(f'Model file {best_model_path} does not exist!')
    sys.exit(1)

# load best/given model
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
model.to(device)

# evaluation mode
model.eval()

all_preds = []

logger.info('Predicting OGTs.')
with torch.inference_mode():
    for batch in tqdm(test_loader, desc='Predicting'):
        x = batch['embeddings'].to(device)

        # predict OGT
        mu, log_var = model(x)

        all_preds.append(mu.float().cpu())
        
ogts = torch.cat(all_preds, dim=0).tolist()

# output file path and name
outfile = Path(config.paths.data_dir) / 'baseline.csv'

# build a small dataframe of (member, prediction) using the current test order
test_df = df[df['split'] == 'test'].reset_index(drop=True)
pred_df = pd.DataFrame({'member': test_df['member'], config.model.name: ogts})

if outfile.exists():
    logger.info(f'Output file already exists. Updating {config.model.name} column.')
    out_df = pd.read_csv(outfile)
    # drop the column if it already exists
    out_df = out_df.drop(columns=config.model.name, errors='ignore')
else:
    # pepare new dataframe
    column_list = ['member', 'ncbiTaxID_new', 'Temp_Duplicate_Average', 'bin_name']
    out_df = test_df[column_list]

# merge on 'member' in case the row order is different
out_df = out_df.merge(pred_df, on='member', how='left', validate='one_to_one')

# sanity check for missing values
n_missing = out_df[config.model.name].isna().sum()
if n_missing:
    logger.error(f'{n_missing} rows failed to align on "member" — check split file consistency!')
    sys.exit(1)

logger.info(f'Saving dataframe to {outfile}.')
out_df.to_csv(outfile, index=False)
