from dataclasses import dataclass
from typing import List
from hydra.core.config_store import ConfigStore

@dataclass
class Paths:
    image_train: str
    label_train: str
    image_test: str
    label_test: str

@dataclass
class Image:
    height: int
    width: int
    channels: int

@dataclass
class MaxPool:
    kernel_size: int
    stride: int
    padding: int

@dataclass
class ConvLayer(MaxPool):
    out_channels: int

@dataclass
class Model:
    lr: float
    batch_size: int
    dropout: float
    classes: int


@dataclass
class MNISTConfig:
    image: Image
    model: Model
    conv_layers: List[ConvLayer]
    maxpool: MaxPool
    paths: Paths

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="mnist_config",
        node=MNISTConfig
    )