from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_heads: int = 8
    num_sampling_points: int = 8
    num_queries: int = 2
    arcface_centers: int = 20
    arcface_scale: float = 20.0
    arcface_margin: float = 0.5
    dropout: float = 0.1
    pretrained_backbone: bool = False

    def __post_init__(self):
        if self.embed_dim != 256 or self.num_queries != 2:
            raise ValueError('This architecture requires embed_dim=256 and num_queries=2')
        if self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError('num_heads must divide embed_dim')
        if self.num_sampling_points < 1 or self.arcface_centers < 1:
            raise ValueError('Sampling points and ArcFace centers must be positive')
        if not 0 <= self.dropout < 1 or self.arcface_scale <= 0:
            raise ValueError('Invalid dropout or ArcFace scale')
        if not 0 <= self.arcface_margin < 1.5707963267948966:
            raise ValueError('arcface_margin must be in [0, pi/2)')


@dataclass
class DataConfig:
    root: str = 'data'
    train_split: str = 'train'
    val_split: str = 'val'
    scene_glob: str = 'Town*/scene_*'
    image_dir: str = 'front_cam_pv_rgb'
    label_dir: str = 'info'
    sequence_length: int = 10
    sample_stride: int = 1
    skip_first: int = 10
    image_height: int = 512
    image_width: int = 960
    light_key: str = 'traffic_light_20'
    distance_key: str = 'traffic_light_distance'

    def __post_init__(self):
        if self.sequence_length < 1 or self.sample_stride < 1 or self.skip_first < 0:
            raise ValueError('Invalid sequence settings')
        if min(self.image_height, self.image_width) < 129:
            raise ValueError('Unpadded projection convolutions require image dimensions >=129')


@dataclass
class TrainConfig:
    output_dir: str = 'runs/tlcformer'
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    scheduler_step_size: int = 12
    scheduler_gamma: float = 0.4
    max_grad_norm: float = 5.0
    light_loss_weight: float = 1.0
    distance_loss_weight: float = 1.0
    device: str = 'auto'
    seed: int = 42

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.scheduler_step_size) < 1 or self.num_workers < 0:
            raise ValueError('Invalid training count')
        if min(self.learning_rate, self.scheduler_gamma, self.max_grad_norm) <= 0:
            raise ValueError('Learning rate, scheduler gamma and gradient norm must be positive')
        if min(self.weight_decay, self.light_loss_weight, self.distance_loss_weight) < 0:
            raise ValueError('Loss weights and weight decay must be nonnegative')
        if self.light_loss_weight + self.distance_loss_weight == 0:
            raise ValueError('At least one loss weight must be positive')


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {'model', 'data', 'train'}:
            raise ValueError('Config must contain only model, data and train sections')
        sections = {}
        for name, typ in [('model', ModelConfig), ('data', DataConfig), ('train', TrainConfig)]:
            options = value.get(name, {})
            if not isinstance(options, dict):
                raise ValueError(f'{name} must be a mapping')
            unknown = set(options) - {f.name for f in fields(typ)}
            if unknown:
                raise ValueError(f'Unknown {name} options: {sorted(unknown)}')
            sections[name] = typ(**options)
        return cls(**sections)


def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        return Config.from_dict(yaml.safe_load(stream))
