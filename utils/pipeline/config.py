import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Dict, Any, Optional, Union
import yaml

class _ConfigBase:
    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item, default=None):
        return getattr(self, item, default) if hasattr(self, item) else default

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

def _from_dict_helper(cls, d: Any) -> Any:
    if not isinstance(d, dict):
        return d
    
    from typing import get_args, get_origin
    kwargs = {}
    for f in fields(cls):
        field_type = f.type
        origin = get_origin(field_type)
        
        is_optional = False
        if origin is Union:
            args = get_args(field_type)
            is_optional = type(None) in args
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                field_type = non_none[0]
        
        if f.name in d:
            val = d[f.name]
            if val is None and not is_optional:
                raise ValueError(f"Field '{f.name}' in {cls.__name__} cannot be None (expected {field_type})")
        else:
            if f.default is not MISSING:
                val = f.default
            elif f.default_factory is not MISSING:
                val = f.default_factory()
            else:
                raise ValueError(f"Missing required field '{f.name}' in {cls.__name__}")
        
        if is_dataclass(field_type) and isinstance(val, dict):
            kwargs[f.name] = _from_dict_helper(field_type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)

# Below are the config schema classes
# These classes will be used to store the config values
# and will be used to create a Config object
@dataclass
class DataConfig(_ConfigBase):
    data_root: str
    source_data_type: str
    is_cmdonly: bool
    eval_split_ratio: float
    num_workers: int
    description_level: str
    batch_size: int
    sample_ratio: Optional[float] = None
    max_samples: Optional[int] = None
    train_csv_path: Optional[str] = None
    val_data_root: Optional[str] = None
    val_csv_path: Optional[str] = None

@dataclass
class TokenizerConfig(_ConfigBase):
    source: str
    model_name: str

@dataclass
class ModelConfig(_ConfigBase):
    is_pretrained: bool
    max_new_cmds: int
    encoder_type: str = "t5-small"
    cmd_decoder_type: str = "t5-small"
    args_decoder_type: Optional[str] = None
    is_cmd_only: bool = False
    d_model: Optional[int] = 512
    moe_type: Optional[str] = None
    moe_conf: Optional[str] = None
    freeze_encoder: bool = False
    freeze_cmd_decoder: bool = False
    freeze_args_decoder: bool = False
    adaptive_layer: str = "none"
    cmd_embedding_type: str = "standard"
    args_embedding_type: str = "standard"
    use_drop_out: bool = True
    drop_out_p: float = 0.1
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CriterionConfig(_ConfigBase):
    source: str
    cls: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchedulerConfig(_ConfigBase):
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = None

@dataclass
class TrainerConfig(_ConfigBase):
    optimizer: str
    criterion: CriterionConfig
    epochs: int
    max_new_cmds: int
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_steps: int = 1000
    scheduler: Optional[SchedulerConfig] = None

@dataclass
class FineTuningConfig(_ConfigBase):
    run_name: str
    type: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    random_seed: int
    pretrained_path: Optional[str] = None

@dataclass
class PretrainConfig(_ConfigBase):
    run_name: str
    type: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    random_seed: int
    pretrained_path: Optional[str] = None

@dataclass
class GRPOKwargsConfig(_ConfigBase):
    n_rollouts: int = 8
    clip_eps: float = 0.2
    min_cd: float = 1e-5
    max_cd: float = 0.5
    eval_fraction: float = 0.1
    temperature: float = 1.0

@dataclass
class GRPOConfig(_ConfigBase):
    run_name: str
    type: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    grpo: GRPOKwargsConfig
    random_seed: int
    pretrained_path: Optional[str] = None

class Config:
    # Namespace mappings for backwards compatibility and referencing nested classes
    Data = DataConfig
    Tokenizer = TokenizerConfig
    Model = ModelConfig
    Criterion = CriterionConfig
    Trainer = TrainerConfig
    Scheduler = SchedulerConfig
    GRPOKwargs = GRPOKwargsConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Union[FineTuningConfig, PretrainConfig, GRPOConfig]:
        config_type = d.get("type", "fine_tuning")
        if config_type == "pretrain":
            if "type" not in d:
                d = dict(d)
                d["type"] = "pretrain"
            return _from_dict_helper(PretrainConfig, d)
        elif config_type == "fine_tuning":
            if "type" not in d:
                d = dict(d)
                d["type"] = "fine_tuning"
            return _from_dict_helper(FineTuningConfig, d)
        elif config_type == "grpo":
            if "type" not in d:
                d = dict(d)
                d["type"] = "grpo"
            return _from_dict_helper(GRPOConfig, d)
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    @classmethod
    def from_yaml(cls, path: str) -> Union[FineTuningConfig, PretrainConfig, GRPOConfig]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)