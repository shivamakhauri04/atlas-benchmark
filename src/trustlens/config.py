"""TrustLens experiment configuration."""

from research_utils.config import BaseExperimentConfig


class TrainConfig(BaseExperimentConfig):
    """Hyperparameters for the TrustLens trace encoder."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 50
    dataset: str = "toolbench"
    use_args: bool = True
    alpha: float = 0.05
    aci_gamma: float = 0.01
