import torch
import math
import pandas as pd
import logging
from typing import Union, List, Tuple
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from typing import Literal
from pathlib import Path
from tqdm.auto import tqdm
from Bio import SeqIO

logger = logging.getLogger(__name__)

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

# MLP regression head
class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,                                                 # dimensionality of the input
        hidden_dims: Union[List[int], Tuple[int, ...]] = (512, 128),    # dimesions of the MLP layers
        dropout: float = 0.1,                                           # dropout value
        layer_norm: bool = True,                                        # should normalization be applied
        log_var_min: float = -10.0,                                     # log_var clamping min value
        log_var_max: float = 5.0,                                       # log_var clamping max value
        mean_out_bias_init: float = 30.0                                # mean_out bias initialization
    ):
        super().__init__()

        # log_var clamping values
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # layer configuration
        layers = []

        # add a normalization layer
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        # store the current layer dimension temporarely
        prev_dim = int(input_dim)
        # build the MLP from the hidden_dims given
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            # add a linear layer (reducing the dimensionality)
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # add a GELU layer (should perform better than ReLU)
            layers.append(nn.GELU())
            # add a dropout layer to improve learning and reduce overfitting
            layers.append(nn.Dropout(dropout))
            # update the stored previous layer dimension
            prev_dim = hidden_dim

        # build shared neural net for both output heads
        self.shared_net = nn.Sequential(*layers)

        # output heads for OGT and log_var
        self.mean_out = nn.Linear(prev_dim, 1)
        self.logvar_out = nn.Linear(prev_dim, 1)
        
        # initialize log_var close to zero (for added stability)
        nn.init.zeros_(self.logvar_out.weight)
        nn.init.zeros_(self.logvar_out.bias)

        # initialize mean_out bias to approximate training set mean OGT
        nn.init.constant_(self.mean_out.bias, mean_out_bias_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pass through the shared network
        shared_features = self.shared_net(x)
        
        # predict OGT and log_var
        mu = self.mean_out(shared_features).squeeze(-1)
        log_var = self.logvar_out(shared_features).squeeze(-1)
        
        # clamp log_var for stability
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        
        return mu, log_var

# ESM model setup
# https://github.com/ProteinVision/ESM2-Tutorial/blob/main/ESM2.ipynb
class ESMDoRA(nn.Module):
    def __init__(
        self,
        esm_model_name: str,                                                            # ESM model name
        head_hidden_dims: Union[List[int], Tuple[int, ...]] = (512, 128),               # head MLP setup
        head_dropout: float = 0.1,                                                      # head dropout
        layer_norm: bool = True,                                                        # head should normalization be applied
        log_var_min: float = -10.0,                                                     # head log_var clamping min value
        log_var_max: float = 5.0,                                                       # head log_var clamping max value
        mean_out_bias_init: float = 30.0,                                               # head mean_out bias initialization
        dora_r: int = 16,                                                               # DoRA rank
        dora_alpha: int = 32,                                                           # DoRA alpha value
        dora_dropout: float = 0.05,                                                     # DoRA dropout
        target_modules: Union[List[str], Tuple[str, ...]] = ('query', 'key', 'value'),  # base model fine-tune targets
        gradient_checkpointing: bool = False,                                           # option to reduce GPU memory footprint
        adapter_path: Union[str, Path, None] = None,                                    # path to the adapter to load
        adapter_trainable: bool = True,                                                 # set the adapter to trainable or not
        head_path: Union[str, Path, None] = None                                        # path to the head to load
        ):
        super().__init__()

        # load the HuggingFace base model
        base = AutoModel.from_pretrained(esm_model_name, dtype=torch.bfloat16)

        # option to reduce the GPU memory footprint
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            if hasattr(base, 'enable_input_require_grads'):
                base.enable_input_require_grads()

        if adapter_path is None:
            # setup the DoRA fine-tuning configuration
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=dora_r,
                lora_alpha=dora_alpha,
                lora_dropout=dora_dropout,
                use_dora=True,
                bias='none',
                target_modules=list(target_modules),
            )

            # wrap the base model and DoRA configuration
            self.esm = get_peft_model(base, peft_config)
        else:
            self.esm = PeftModel.from_pretrained(base, adapter_path, is_trainable=adapter_trainable)

        # get the output dimensionality of the base model
        input_dim = base.config.hidden_size

        # prepare the regression head
        self.head = RegressionHead(
            input_dim=input_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
            layer_norm=layer_norm,
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            mean_out_bias_init=mean_out_bias_init
        )

        if head_path is not None:
            checkpoint = torch.load(head_path, map_location='cpu')
            self.head.load_state_dict(checkpoint)

    # mean pooling ovcer the amino acid residues only
    def pool_mean(self, last_hidden_state, residue_mask):
        # mean pooling with higher precision
        mask = residue_mask.unsqueeze(-1).float()
        summed = (last_hidden_state.float() * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)

        return (summed / denom).to(last_hidden_state.dtype)
    
    def forward(self, input_ids, attention_mask, residue_mask, labels=None):
        # run the input through ESM
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # retrieve the output
        hidden = outputs.last_hidden_state

        # mean pooling the aa residues only
        pooled = self.pool_mean(hidden, residue_mask)

        # run small regression head and NLL in FP32
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            pooled = pooled.float()
            # run the pooled output through the regression head
            mu, log_var = self.head(pooled)

            # prepare the results for return
            result = {
                'mu': mu,
                'log_var': log_var,
                'var': torch.exp(log_var),
                'std': torch.exp(0.5 * log_var),
            }

            # calculate the Gaussian NLL loss
            if labels is not None:
                result['loss'] = gaussian_nll_loss(mu, log_var, labels)

        return result

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
        log_every: int = 100,
        overall_progbar=None
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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        residue_mask = batch['residue_mask'].to(device)
        labels = batch['labels'].to(device)

        # zero your gradients for every batch
        optimizer.zero_grad()

        # make predictions for this batch
        # autocast enables mix precision (necessary to overcome OOM problems)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
                labels=labels,
            )

            # compute loss
            loss = outputs['loss']

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

            if overall_progbar is not None:
                overall_progbar.set_postfix({
                    'epoch': epoch_index + 1,
                    'train_loss': f'{avg_loss:.5f}',
                })

            global_step = epoch_index * len(training_loader) + i
            tb_writer.add_scalar('Loss/train_batch', loss_value, global_step)

        if overall_progbar is not None:
            overall_progbar.update(1)

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            residue_mask = batch['residue_mask'].to(device)
            labels = batch['labels'].to(device)

            # make predictions for this batch
            # autocast enables mix precision (necessary to overcome OOM problems)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    residue_mask=residue_mask,
                    labels=labels,
                )

                # compute loss
                loss_value = outputs['loss'].item()

                all_mu.append(outputs['mu'].detach().cpu())
                all_labels.append(labels.detach().cpu())

            total_loss += loss_value
            avg_loss = total_loss / i

            progbar.set_postfix({
                'val_loss': f'{avg_loss:.4f}',
            })

            if overall_progbar is not None:
                overall_progbar.set_postfix({
                    'epoch': epoch_index + 1,
                    'val_loss': f'{avg_loss:.4f}',
                })
                overall_progbar.update(1)

    # compute RMSE and MAE from accumulated predictions
    all_mu_t = torch.cat(all_mu).float()
    all_labels_t = torch.cat(all_labels).float()

    rmse = torch.sqrt(torch.mean((all_mu_t - all_labels_t) ** 2)).item()
    mae  = torch.mean(torch.abs(all_mu_t - all_labels_t)).item()

    return total_loss / len(validation_loader), rmse, mae

# download the specified model from huggingface to the specified directory
def download_model(
        model: str,                         # model name
        local_dir: Path                     # path to store the model in
    ):

    logger.info(f'Downloading {model} to {local_dir}.')
    snapshot_download(
        repo_id=model,
        local_dir=local_dir,
        # ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.safetensors"],  # Optional: skip large files if needed
        force_download=False,
    )
    logger.info(f'Finished downloading {model}.')

# prepare the sequences and OGT values for the given split
def prepare_split_data(
        df: pd.DataFrame,                                       # train/test/val split dataframe
        split: Literal['train', 'test', 'val', 'all'],          # choice of split from the list
        proteomes_dir: Path,                                    # directory of the proteome files
        as_dataframe: bool = False                              # return as dataframe instaed of tuple of lists
    ) -> Union[Tuple[List[str], List[float]], pd.DataFrame]:    # returns a list of sequences and corresponding OGTs

    # only access the required split
    if not split == 'all':
        df = df[df['split'] == split]

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
        matches = list(Path(proteomes_dir).glob(f'{str(int(taxid))}_*_protein.faa'))

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

    if as_dataframe:
        # add the sequences to the dataframe
        df['sequence'] = sequences

        return df
    else:
        return sequences, ogts

