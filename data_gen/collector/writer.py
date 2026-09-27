import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from collector.bev import OUT_SIZE, PIXELS_PER_METER, ANGLE_BIAS_DEG


MASK_FOLDERS = ("das", "lane", "walker_mask", "vehicle_mask", "stopline", "route", "road_network")


def depth_meters(image):
    weights = np.array([65536.0, 256.0, 1.0], dtype=np.float32)
    return (image.astype(np.float32) @ weights) * np.float32(1000.0 / 16777215.0)


def new_scene(output, town, config, options):
    root = Path(output) / town
    root.mkdir(parents=True, exist_ok=True)
    index = max((int(path.name[6:]) for path in root.glob("scene_*") if path.is_dir() and path.name[6:].isdigit()), default=0) + 1
    while True:
        scene = root / f"scene_{index:04d}"
        try:
            scene.mkdir()
            break
        except FileExistsError:
            index += 1
    folders = list(MASK_FOLDERS) + ["info"]
    for name, settings in config["sensors"].items():
        if name == "lidar":
            folders.append("lidar")
            continue
        folders.append(f"{name}_rgb")
        for flag, suffix in (("seg", "seg"), ("seg", "seg_ids"), ("depth", "depth"), ("od", "bbs")):
            if settings[flag]:
                folders.append(f"{name}_{suffix}")
    for folder in folders:
        (scene / folder).mkdir()
    metadata = {"config": config, "collection": options}
    (scene / "collection.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return scene


def location_to_dict(location):
    return {key: getattr(location, key) for key in ("x", "y", "z")} if location is not None else None


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_sample(scene, stem, world, agent, info, sensor_config, frame, timestamp, sensor_frames, bev_map):
    if not sensor_frames or any(value != frame for value in sensor_frames.values()):
        raise ValueError("All sensor frames must match the world frame")
    outputs = []

    def write_png(folder, value):
        path = scene / folder / f"{stem}.png"
        outputs.append(path)
        if not cv2.imwrite(str(path), value, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Failed to save {path}")

    def write_npy(folder, value):
        path = scene / folder / f"{stem}.npy"
        outputs.append(path)
        np.save(path, value, allow_pickle=False)

    try:
        masks, footprints = bev_map.render(world.world, world.player, info, agent.future_waypoints)
        for folder, mask in masks.items():
            write_png(folder, mask)
        boxes = world._get_3d_bbs(max_distance=50) if any(settings.get("od", False) for settings in sensor_config.values()) else None
        for name, settings in sensor_config.items():
            if name == "lidar":
                write_npy("lidar", world.lidar_sensor.point_cloud)
                continue
            write_png(f"{name}_rgb", getattr(world, f"{name}_rgb").img)
            if settings["seg"]:
                segmentation = getattr(world, f"{name}_seg")
                write_png(f"{name}_seg", segmentation.color_img)
                write_png(f"{name}_seg_ids", segmentation.img[:, :, 2])
            if settings["depth"]:
                depth = depth_meters(getattr(world, f"{name}_depth").img)
                write_npy(f"{name}_depth", depth)
                write_png(f"{name}_depth", np.rint(np.clip(depth, 0, 65.535) * 1000).astype(np.uint16))
            if settings["od"]:
                world._sensor_data = {key: settings[key] for key in ("width", "height", "fov")}
                world._sensor_data["calibration"] = world._get_camera_to_car_calibration(world._sensor_data)
                segmentation = getattr(world, f"{name}_seg")
                bbs = world._get_2d_bbs(segmentation.sensor.get_transform(), boxes, segmentation.img[:, :, 2])
                write_npy(f"{name}_bbs", np.asarray(bbs, dtype=np.float32).reshape(-1, 5))
        metadata = dict(info)
        for key in ("current_waypoint", "target_waypoint"):
            metadata[key] = location_to_dict(metadata[key])
        metadata["future_waypoints"] = [location_to_dict(point) for point in metadata["future_waypoints"]]
        metadata.update({
            "frame": frame,
            "timestamp": timestamp,
            "sensor_frames": sensor_frames,
            "imu": world.imu_sensor.imu_data,
            "bev_size": OUT_SIZE,
            "bev_meters_per_pixel": 1.0 / PIXELS_PER_METER,
            "bev_angle_bias_deg": ANGLE_BIAS_DEG,
            "bev_world_offset": bev_map.offset.tolist(),
            "bev_source": "data5",
            "bev_dynamic_footprints": footprints,
            "camera_angles_unit": "degrees",
        })
        path = scene / "info" / f"{stem}.json"
        temp = path.with_suffix(".json.tmp")
        outputs.extend((temp, path))
        temp.write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")
        temp.replace(path)
    except BaseException:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
