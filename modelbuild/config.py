from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    name: str

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    max_length: int

@dataclass
class PathsConfig:
    data_dir: str
    split_file: str
    proteomes_dir: str
    model_dir: str
    adapter_dir: str

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, file: str = 'config.yaml') -> 'Config':
        # create Path object
        f = Path(file)

        # check if the config file exists
        if not f.exists():
            print(f'WARNING: {file} not found!')

        # read the config and return the content as dictionary or an empty dictionary
        with open(f) as c:
            print(f'Reading configuration from {file}.')
            raw = yaml.safe_load(c) or {}

        # build the config object from the YAML read dictionary
        return cls(
            model=ModelConfig(**raw.get('model', {})),
            training=TrainingConfig(**raw.get('training', {})),
            paths=PathsConfig(**raw.get('paths', {})),
        )