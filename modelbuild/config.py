from dataclasses import dataclass, field
from typing import List, Tuple, Union
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str

@dataclass
class TrainingConfig:
    base_seed: int
    adapters: int
    epochs: int
    batch_size: int
    max_length: int
    head_learning_rate: float
    adapter_learning_rate: float
    weight_decay: float
    patience: int

@dataclass
class TestingConfig:
    adapters: List[str]
    heads: List[str]

@dataclass
class RegressionHeadConfig:
    name: str
    hidden_layers: Union[List[int], Tuple[int, ...]]
    dropout: float
    layer_norm: bool
    log_var_min: float
    log_var_max: float

@dataclass
class ESMDoRAConfig:
    dora_r: int
    dora_alpha: int
    dora_dropout: float
    target_modules: Union[List[str], Tuple[str, ...]]

@dataclass
class PathsConfig:
    data_dir: str
    split_file: str
    embedding_file: str
    proteomes_dir: str
    model_dir: str
    adapter_dir: str

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    head: RegressionHeadConfig = field(default_factory=RegressionHeadConfig)
    esmdora: ESMDoRAConfig = field(default_factory=ESMDoRAConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, file: str = 'config.yaml') -> 'Config':
        # create Path object
        f = Path(file)

        # check if the config file exists
        if not f.exists():
            logger.warning(f'WARNING: {file} not found!')

        # read the config and return the content as dictionary or an empty dictionary
        with open(f) as c:
            logger.info(f'Reading configuration from {file}.')
            raw = yaml.safe_load(c) or {}

        # build the config object from the YAML read dictionary
        return cls(
            model=ModelConfig(**raw.get('model', {})),
            training=TrainingConfig(**raw.get('training', {})),
            testing=TestingConfig(**raw.get('testing', {})),
            head=RegressionHeadConfig(**raw.get('head', {})),
            esmdora=ESMDoRAConfig(**raw.get('esmdora', {})),
            paths=PathsConfig(**raw.get('paths', {})),
        )