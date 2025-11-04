import yaml
from dataclasses import dataclass

@dataclass
class ModelParams:
    BATCH_SIZE: int
    LR: float
    EPOCHS: int
    PATCH_SIZE: int
    NUM_CLASSES: int
    IMAGE_SIZE: int
    IN_CHANNELS: int
    EMBED_DIMENSION: int
    NUM_HEADS: int
    DEPTH: int
    DROPOUT_RATE: float
    MLP_RATIO: float
    PATIENCE: int