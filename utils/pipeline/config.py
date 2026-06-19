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

@dataclass
class TokenizerConfig(_ConfigBase):
    source: str
    model_name: str

@dataclass
class ModelConfig(_ConfigBase):
    source: str
    cls: str
    is_pretrained: bool
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CriterionConfig(_ConfigBase):
    source: str
    cls: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainerConfig(_ConfigBase):
    optimizer: str
    criterion: CriterionConfig
    epochs: int
    max_new_cmds: int
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config(_ConfigBase):
    run_name: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    random_seed: int
    pretrained_path: Optional[str] = None

    # Namespace mappings for backwards compatibility and referencing nested classes
    Data = DataConfig
    Tokenizer = TokenizerConfig
    Model = ModelConfig
    Criterion = CriterionConfig
    Trainer = TrainerConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return _from_dict_helper(cls, d)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
