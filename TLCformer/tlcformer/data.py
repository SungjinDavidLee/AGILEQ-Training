import json
import math
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


LIGHT_CLASSES = ('GO', 'STOP')
MAX_DISTANCE = 20
FAR_CLASS = 21
DISTANCE_CLASSES = 22


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower()) for part in re.split(r'(\d+)', str(path)))


def distance_to_class(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('traffic_light_distance must be a number')
    if not math.isfinite(value) or value < 0:
        raise ValueError('traffic_light_distance must be finite and nonnegative')
    return FAR_CLASS if value == 0 or value > MAX_DISTANCE else round(value)


def read_targets(path, light_key='traffic_light_20', distance_key='traffic_light_distance'):
    with Path(path).open(encoding='utf-8') as stream:
        labels = json.load(stream)
    light = labels[light_key]
    if not isinstance(light, (bool, int)) or light not in (0, 1):
        raise ValueError(f'{path}: {light_key} must be boolean or 0/1')
    try:
        distance = distance_to_class(labels[distance_key])
    except (ValueError, KeyError) as error:
        raise ValueError(f'{path}: invalid {distance_key}') from error
    return int(light), distance


def image_transform(height, width):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_image(path, transform):
    with Image.open(path) as image:
        return transform(image.convert('RGB'))


class TrafficLightDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.transform = image_transform(config.image_height, config.image_width)
        self.scenes = []
        self.index = []
        split_root = Path(config.root) / split
        for scene in sorted(split_root.glob(config.scene_glob), key=natural_key):
            images = sorted((scene / config.image_dir).glob('*.png'), key=natural_key)[config.skip_first:]
            if len(images) < config.sequence_length:
                continue
            scene_index = len(self.scenes)
            self.scenes.append((scene, images))
            for end in range(config.sequence_length - 1, len(images), config.sample_stride):
                label = scene / config.label_dir / f'{images[end].stem}.json'
                if not label.is_file():
                    raise FileNotFoundError(label)
                self.index.append((scene_index, end))
        if not self.index:
            raise ValueError(f'No complete sequences under {split_root}/{config.scene_glob}; check sequence_length and skip_first')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        scene_index, end = self.index[index]
        scene, paths = self.scenes[scene_index]
        sequence = paths[end - self.config.sequence_length + 1:end + 1]
        label_path = scene / self.config.label_dir / f'{paths[end].stem}.json'
        light, distance = read_targets(label_path, self.config.light_key, self.config.distance_key)
        return {'images': torch.stack([load_image(path, self.transform) for path in sequence]),
                'light_target': torch.tensor(light, dtype=torch.long),
                'distance_target': torch.tensor(distance, dtype=torch.long),
                'frame_path': str(paths[end])}
