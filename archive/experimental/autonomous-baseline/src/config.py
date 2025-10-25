"""Global configuration using Pydantic for type-safe settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    """Data splitting and processing configuration."""

    test_size: float = Field(0.20, ge=0.0, le=1.0)
    val_size: float = Field(0.10, ge=0.0, le=1.0)
    seed_labeled_size: int = Field(50, ge=10)
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
    """Active learning configuration."""

    acquisition_fn: Literal["ucb", "ei", "max_var", "eig_proxy"] = "ucb"
    diversity_method: Optional[Literal["kmedoids", "dpp"]] = "kmedoids"
    diversity_weight: float = Field(0.3, ge=0.0, le=1.0)
    budget_total: int = 200
    k_per_round: int = 10
    ucb_beta: float = 2.0
    cost_weight: float = Field(0.0, ge=0.0)  # 0 = no cost penalty
    risk_gate_sigma_max: float = Field(10.0, gt=0.0)  # Max uncertainty allowed
    ood_block: bool = True  # Block OOD candidates


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

