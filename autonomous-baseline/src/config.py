"""Global configuration using Pydantic for type-safe settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def compute_seed_size(
    train_pool_size: int,
    num_classes: int,
    min_per_class: int = 10
) -> int:
    """
    Compute seed set size per NeurIPS/ICLR/JMLR protocol.
    
    Formula: max(0.02·|D_train|, 10·|C|) capped at 0.05·|D_train|
    
    Args:
        train_pool_size: Total size of training pool
        num_classes: Number of classes (or 1 for regression)
        min_per_class: Minimum samples per class (default: 10)
    
    Returns:
        Seed set size
    
    Example:
        >>> compute_seed_size(train_pool_size=1000, num_classes=5)
        50  # max(20, 50) = 50, capped at 50
    """
    if train_pool_size <= 0:
        raise ValueError(f"train_pool_size must be positive, got {train_pool_size}")
    
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    
    # Minimum: larger of 2% of training pool or 10 per class
    min_size = max(
        int(0.02 * train_pool_size),  # 2% of training pool
        min_per_class * num_classes    # 10 samples per class
    )
    
    # Maximum: 5% of training pool
    max_size = int(0.05 * train_pool_size)
    
    # Return minimum, capped at maximum
    seed_size = min(min_size, max_size)
    
    return seed_size


class PathConfig(BaseModel):
    """File path configuration."""

    data_root: Path = Path("data")
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    contracts: Path = Path("data/contracts")
    evidence: Path = Path("evidence")
    models: Path = Path("models")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v: Path) -> Path:
        """Ensure paths are resolved."""
        return Path(v).resolve() if isinstance(v, (str, Path)) else v


class DataConfig(BaseModel):
    """Data splitting and processing configuration (NeurIPS/ICLR compliant)."""

    test_size: float = Field(0.15, ge=0.0, le=1.0)  # FIXED: was 0.20, now 70/15/15 split
    val_size: float = Field(0.15, ge=0.0, le=1.0)   # FIXED: was 0.10, now 70/15/15 split
    # NOTE: seed_labeled_size removed - computed dynamically via compute_seed_size()
    min_samples_per_class: int = Field(10, ge=1)  # NEW: for seed size formula
    stratify_bins: int = Field(5, ge=2)
    near_dup_threshold: float = Field(0.99, ge=0.0, le=1.0)  # Lower=stricter, tune per dataset


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    use_matminer: bool = True
    featurizer_type: Literal["magpie", "light"] = "magpie"
    n_components_pca: Optional[int] = None  # None = no PCA
    scale_method: Literal["standard", "robust", "minmax"] = "standard"


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_type: Literal["rf_qrf", "mlp_mc", "ngboost", "gpr"] = "rf_qrf"
    random_state: int = 42
    n_jobs: int = -1


class RFQRFConfig(ModelConfig):
    """Random Forest with Quantile Regression Forest configuration."""

    model_type: Literal["rf_qrf"] = "rf_qrf"
    n_estimators: int = 200
    max_depth: Optional[int] = 30
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: Literal["sqrt", "log2"] = "sqrt"
    bootstrap: bool = True
    quantiles: list[float] = [0.025, 0.975]  # 95% PI


class MLPMCConfig(ModelConfig):
    """MLP with MC Dropout configuration."""

    model_type: Literal["mlp_mc"] = "mlp_mc"
    hidden_dims: list[int] = [256, 128, 64]
    dropout_p: float = Field(0.2, ge=0.0, lt=1.0)
    mc_samples: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    early_stopping_patience: int = 20


class NGBoostConfig(ModelConfig):
    """NGBoost configuration for aleatoric uncertainty."""

    model_type: Literal["ngboost"] = "ngboost"
    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 1.0
    base_learner_depth: int = 5


class UncertaintyConfig(BaseModel):
    """Uncertainty calibration configuration."""

    conformal_method: Literal["split", "mondrian"] = "mondrian"
    conformal_alpha: float = Field(0.05, ge=0.0, le=1.0)  # 95% coverage
    ece_n_bins: int = 15
    calibration_size: float = Field(0.3, ge=0.1, le=0.5)


class OODConfig(BaseModel):
    """Out-of-distribution detection configuration."""

    mahalanobis_quantile: float = Field(0.95, ge=0.0, le=1.0)
    kde_bandwidth: Optional[float] = None  # None = auto
    kde_quantile: float = Field(0.05, ge=0.0, le=1.0)
    conformal_risk_threshold: float = Field(0.90, ge=0.0, le=1.0)


class ActiveLearningConfig(BaseModel):
    """Active learning configuration (NeurIPS/ICLR compliant)."""

    acquisition_fn: Literal["ucb", "ei", "max_var", "eig_proxy"] = "ucb"
    diversity_method: Optional[Literal["kmedoids", "dpp"]] = "kmedoids"
    diversity_weight: float = Field(0.3, ge=0.0, le=1.0)
    budget_total: int = 200
    k_per_round: int = 10  # Used only if use_dynamic_batch_size=False
    ucb_beta: float = 2.0
    cost_weight: float = Field(0.0, ge=0.0)  # 0 = no cost penalty
    risk_gate_sigma_max: float = Field(10.0, gt=0.0)  # Max uncertainty allowed
    ood_block: bool = True  # Block OOD candidates
    
    # NEW: Dynamic batch sizing (Protocol compliance)
    use_dynamic_batch_size: bool = True  # Use 5% of remaining pool
    fixed_batch_size: Optional[int] = None  # Override if not None (ignores use_dynamic_batch_size)


class Config(BaseModel):
    """Global configuration container."""

    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    ood: OODConfig = OODConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()

    seed: int = 42
    deterministic: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_config() -> Config:
    """Get global configuration instance."""
    return Config()

