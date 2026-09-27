import copy
import math
import re
from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = copy.deepcopy(config)
    count = config.get("data_num")
    if isinstance(count, bool) or not re.fullmatch(r"[0-9]+", str(count)) or int(count) < 1:
        raise ValueError("data_num must be a positive integer")
    config["data_num"] = int(count)
    towns = config.get("towns")
    if not isinstance(towns, list) or not towns or any(not isinstance(town, str) or not re.fullmatch(r"[A-Za-z0-9_]+", town) for town in towns):
        raise ValueError("towns must contain safe CARLA map names")
    if len(set(towns)) != len(towns):
        raise ValueError("towns must not contain duplicates")
    sensors = config.get("sensors")
    if not isinstance(sensors, dict) or not any(name != "lidar" for name in sensors):
        raise ValueError("sensors must contain at least one camera")
    for name, settings in sensors.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid sensor name: {name}")
        if not isinstance(settings, dict):
            raise ValueError(f"{name}: settings must be a mapping")
        for key in ("x", "y", "z", "pitch", "roll", "yaw"):
            value = settings.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be a finite number")
        if name == "lidar":
            for key in ("channels", "range", "points_per_second", "rotation_frequency"):
                if key in settings:
                    value = settings[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"lidar.{key} must be positive")
            continue
        for key in ("width", "height"):
            value = settings.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}.{key} must be a positive integer")
        fov = settings.get("fov")
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 0 < fov < 180:
            raise ValueError(f"{name}.fov must be between 0 and 180 degrees")
        for key in ("seg", "depth", "od"):
            settings.setdefault(key, False)
            if not isinstance(settings[key], bool):
                raise ValueError(f"{name}.{key} must be true or false, without quotes")
    return config
