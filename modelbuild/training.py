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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
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

# this script fine-tunes ESM-2 and trains the regression head

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
group = parser.add_mutually_exclusive_group()
group.add_argument('-r', '--resume', metavar='STATE_FILE', help='Resume training with the given state file.')
group.add_argument('-a', '--adapter', type=int, metavar='N', help='Train only this specific adapter.')
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
logger.info('Preparing training dataset.')
sequences, ogts = prepare_split_data(df, 'train', config.paths.proteomes_dir)
mean_ogt = statistics.mean(ogts)
logger.info(f'Training set mean OGT: {mean_ogt:.1f}°C.')
train_dataset = PsychrophileDataset(
    sequences,
    ogts,
    tokenizer,
    config.training.max_length,
)

logger.info(f'Training dataset has {len(train_dataset)} entries.')

# print(train_dataset[0])

logger.info('Preparing validation dataset.')
val_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'val', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

logger.info(f'Validation dataset has {len(val_dataset)} entries.')

# print(val_dataset[0])

# custom collator for dynamic batch padding and mask creation
collator = PsychrophileCollator(tokenizer=tokenizer)

# create the dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size, 
    shuffle=True, 
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

# initialize parameters for this run
checkpoint = None                           # PyTorch checkpoint if training is resumed
start_adapter = 0                           # default adapter start
stop_adapter = config.training.adapters     # number of adapters to train

# if resume is given load state
if args.resume is not None:
    resume_path = Path(args.resume)
    if not resume_path.exists():
        logger.error(f'Training resume file {resume_path} does not exist!')
        sys.exit(1)

    # load saved checkpoint
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    # only get the adapter to train and timestamp, the rest later when everything is in place
    start_adapter = checkpoint['adapter']
    timestamp = checkpoint['timestamp']

    logger.info(f'Resume training for adapter {start_adapter + 1}/{config.training.adapters}.')
elif args.adapter is not None:
    # only train one specific adapter
    if args.adapter < 1 or args.adapter > config.training.adapters:
        logger.error(f'Adapter must be between 1 and {config.training.adapters} got {args.adapter}.')
        sys.exit(1)

    start_adapter = args.adapter - 1
    stop_adapter = args.adapter
else:
    logger.info(f'Beginning to train {config.training.adapters} adapters for {config.training.epochs} epochs.')

# loop over the number of adapter to train
for adapter in range(start_adapter, stop_adapter):
    # filename postfix
    filename_postfix = f'{timestamp}_adapter_{adapter + 1}'

    # initialize the random number generators
    # to be on the save side not only for forch but also Python and numpy
    run_seed = config.training.base_seed + (adapter + 1) ** 5 * 97
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(run_seed)

    logger.info(f'Traning adapter {adapter + 1}/{stop_adapter} with seed {run_seed}.')

    # model setup
    logger.info('Setting up model.')
    model = ESMDoRA(
        esm_model_name=full_model_path,
        head_hidden_dims=config.head.hidden_layers,
        head_dropout=config.head.dropout,
        layer_norm=config.head.layer_norm,
        log_var_min=config.head.log_var_min,
        log_var_max=config.head.log_var_max,
        dora_r=config.esmdora.dora_r,
        dora_alpha=config.esmdora.dora_alpha,
        dora_dropout=config.esmdora.dora_dropout,
        target_modules=config.esmdora.target_modules,
        gradient_checkpointing=True
    )

    # move model to the accelerator
    model.to(device)

    # check tunable parameters of the head and adapter
    # model.esm.print_trainable_parameters()

    total_trainable = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.numel())
            total_trainable += param.numel()

    logger.info(f'Total trainable parameters: {total_trainable:,}')

    # separate MLP head and ESM parameters
    adapter_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith('head.'):
            head_params.append(param)
        else:
            adapter_params.append(param)

    # optimizer setup
    optimizer = torch.optim.AdamW(
        [
            {
                'params': adapter_params,
                'lr': config.training.adapter_learning_rate,
            },
            {
                'params': head_params,
                'lr': config.training.head_learning_rate,
            },
        ],
        weight_decay=config.training.weight_decay,
    )

    # setup scheduler (reducing the learning rate over time)
    total_steps = config.training.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # training loop setup
    # https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    writer = SummaryWriter(Path(config.paths.data_dir) / f'summarywriter/esmdora_{filename_postfix}')

    # initialize parameters for this loop
    best_vloss = 1_000_000.                 # randomly high validation loss
    patience = config.training.patience     # how long to run before early stopping is triggered
    epochs_no_improve = 0                   # counting epoch without improvment
    start_epoch = 0                         # starting the loop on this epoch

    # restore the rest of the checkpoint parameters (only for this adapter)
    if checkpoint is not None and adapter == start_adapter:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_vloss = checkpoint['best_vloss']
        epochs_no_improve = checkpoint['epochs_no_improve']

        logger.info(f'Loaded remaining checkpoint parameters for adapter {adapter + 1}, epoch {start_epoch + 1}.')

    # continue to next adapter if the resumed training has already reached the max number of epochs
    if start_epoch >= config.training.epochs:
        logger.info(f'Adapter {adapter + 1} training completed {config.training.epochs}. Continuing with the next adapter.')
        continue

    # also continue if early stopping conditions are met upon restart
    if epochs_no_improve >= patience:
        logger.info(f'Adapter {adapter + 1} reached early stopping. Continuing with the next adapter.')
        continue

    # setup TQDM output
    total_steps = config.training.epochs * (len(train_loader) + len(val_loader))
    overall_progbar = tqdm(
        total=total_steps,
        desc='Overall training',
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    # update tqdm progress if training is resumed
    if start_epoch > 0:
        # update progress bar
        steps_per_epoch = len(train_loader) + len(val_loader)
        overall_progbar.update(start_epoch * steps_per_epoch)

    # start training loop
    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f'Training epoch {epoch + 1}, adapter {adapter + 1}.', extra=file_only)

        # train the model with the training set
        avg_loss = train_one_epoch(
            training_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            epoch_index=epoch,
            tb_writer=writer,
            device=device,
            overall_progbar=overall_progbar
        )

        # evaluate with the validation set
        avg_vloss, vrmse, vmae = evaluate(
            validation_loader=val_loader,
            model=model,
            device=device,
            epoch_index=epoch,
            overall_progbar=overall_progbar,
        )

        tqdm.write(
            f'Epoch {epoch + 1}: '
            f'train_loss={avg_loss:.5f}, '
            f'val_loss={avg_vloss:.5f}, '
            f'RMSE={vrmse:.2f}°C, '
            f'MAE={vmae:.2f}°C'
        )

        logger.info(f'Adapter {adapter + 1}, epoch {epoch + 1}: train_loss={avg_loss:.5f}, val_loss={avg_vloss:.5f}, RMSE={vrmse:.2f}°C, MAE={vmae:.2f}°C', extra=file_only)

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
            # save adapter
            model.esm.save_pretrained(Path(config.paths.adapter_dir) / f'adapter_{filename_postfix}')
            # save head
            torch.save(model.head.state_dict(), Path(config.paths.model_dir) / f'head_{filename_postfix}.pt')

            logger.info(f'Saved best performing adapter and head, adapter {adapter + 1}, epoch {epoch + 1}, file extension {filename_postfix}.', extra=file_only)
        else:
            epochs_no_improve += 1

        # save training checkpoint
        torch.save(
            {
                'adapter': adapter,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_vloss': best_vloss,
                'epochs_no_improve': epochs_no_improve,
                'timestamp': timestamp
            },
            Path(config.paths.model_dir) / f'training_state_{timestamp}.pt'
        )

        # stop training if there is no improvement
        if epochs_no_improve >= patience:
            tqdm.write(f'Early stopping triggered after {epochs_no_improve} epochs without improvement in epoch {epoch + 1}.')
            logger.info(f'Early stopping triggered after {epochs_no_improve} epochs without improvement in epoch {epoch + 1}.', extra=file_only)
            break

    overall_progbar.close()
    writer.close()

    # clean up memory
    del model, optimizer, scheduler
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    logger.info(f'Finished training adapter {adapter + 1}.')

logger.info('Finished training adapter ensemble.')
