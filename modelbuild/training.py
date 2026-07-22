import sys
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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
full_model_path = Path(config.paths.model_dir) / config.model.name
if not full_model_path.exists():
    download_model(config.model.name, full_model_path)

print(f'Using model: {config.model.name}.')

# read split file
print(f'Reading splits file: {split_file}.')
df = pd.read_csv(split_file)

# create the tokenizer from the configured model
tokenizer = AutoTokenizer.from_pretrained(full_model_path)

# prepare datasets
print('Preparing training dataset.')
train_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'train', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

print(f'Training dataset has {len(train_dataset)} entries.')

# print(train_dataset[0])

print('Preparing validation dataset.')
val_dataset = PsychrophileDataset(
    *prepare_split_data(df, 'val', config.paths.proteomes_dir),
    tokenizer,
    config.training.max_length,
)

print(f'Validation dataset has {len(val_dataset)} entries.')

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
print(f'Using {device} device for tensor calculation acceleration.')

# model setup
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
model.esm.print_trainable_parameters()

total_trainable = 0

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
        total_trainable += param.numel()

print(f'Total trainable parameters: {total_trainable:,}')

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

# training loop setup
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(Path(config.paths.data_dir) / f'summarywriter/esmdora_{timestamp}')

total_steps = config.training.epochs * (len(train_loader) + len(val_loader))

overall_progbar = tqdm(
    total=total_steps,
    desc='Overall training',
    position=0,
    leave=True,
    dynamic_ncols=True,
)

best_vloss = 1_000_000.

for epoch in range(config.training.epochs):
    # train the model with the training set
    avg_loss = train_one_epoch(
        training_loader=train_loader,
        optimizer=optimizer,
        model=model,
        epoch_index=epoch,
        tb_writer=writer,
        device=device,
        overall_progbar=overall_progbar
    )

    # evaluate with the validation set
    avg_vloss = evaluate(
        validation_loader=val_loader,
        model=model,
        device=device,
        epoch_index=epoch,
        overall_progbar=overall_progbar,
    )

    tqdm.write(
        f'Epoch {epoch + 1}: '
        f'train_loss={avg_loss:.5f}, '
        f'val_loss={avg_vloss:.5f}'
    )

    # log the running loss averaged per batch for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # save adapter
        model.esm.save_pretrained(Path(config.paths.adapter_dir) / f'{timestamp}_{epoch}')
        # save head
        torch.save(model.head.state_dict(), Path(config.paths.model_dir) / f'head_{timestamp}_{epoch}.pt')
        # save training state
        torch.save(
            {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_vloss': best_vloss,
            },
            Path(config.paths.model_dir) / f'training_state_{timestamp}_{epoch}.pt'
        )

overall_progbar.close()