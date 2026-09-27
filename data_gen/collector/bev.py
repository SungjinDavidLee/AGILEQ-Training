import math
from pathlib import Path

import cv2
import numpy as np


OUT_SIZE = 200
PIXELS_PER_METER = 5.0
ANGLE_BIAS_DEG = 90.0
HIGH_ROAD_THRESHOLDS = {"Town04": 5.0, "Town05": 7.0}
STATIC_FILES = {"road_network": None, "das": "das_full.png", "lane": "lane_full.png", "stopline": "stoplines.png"}


def read_map(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Cannot read BEV map: {path}")
    image = image.astype(np.uint8, copy=False)
    if image.ndim == 2 and image.max() <= 1:
        image = image * 255
    return image


class BEVMap:
    def __init__(self, root, town):
        directory = Path(root) / town
        self.town = town
        self.offset = np.asarray(np.load(directory / "world_offset.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
        if self.offset.size != 2 or not np.isfinite(self.offset).all():
            raise ValueError(f"Invalid world_offset.npy: {directory}")
        self.images = {name: read_map(directory / (filename or f"{town}.png"), name != "road_network") for name, filename in STATIC_FILES.items()}
        self.high_images = {}
        if town in HIGH_ROAD_THRESHOLDS:
            self.high_images = {name: read_map(directory / f"{name}_full_high.png", True) for name in ("das", "lane")}
        self.center = (OUT_SIZE - 1) / 2.0
        uu, vv = np.meshgrid(np.arange(OUT_SIZE, dtype=np.float32), np.arange(OUT_SIZE, dtype=np.float32))
        self.du, self.dv = uu - self.center, vv - self.center

    def pose(self, info):
        location = info["ego_vehicle_loc"]
        values = (location["x"], location["y"], location["z"], info["ego_vehicle_rot"]["yaw"])
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Ego pose must be finite")
        x, y, z, yaw = values
        cx = int((x - self.offset[0]) * PIXELS_PER_METER)
        cy = int((y - self.offset[1]) * PIXELS_PER_METER)
        theta = math.radians(yaw + ANGLE_BIAS_DEG)
        return cx, cy, math.cos(theta), math.sin(theta), z

    def sampling_grid(self, info):
        cx, cy, ct, st, z = self.pose(info)
        xg = cx + (self.du * ct - self.dv * st)
        yg = cy + (self.du * st + self.dv * ct)
        return np.rint(xg).astype(np.int32), np.rint(yg).astype(np.int32)

    @staticmethod
    def sample(image, xi, yi):
        height, width = image.shape[:2]
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        output = np.zeros((OUT_SIZE, OUT_SIZE) + image.shape[2:], dtype=np.uint8)
        output[inside] = image[yi[inside], xi[inside]]
        return output

    def static_masks(self, info):
        xi, yi = self.sampling_grid(info)
        sources = dict(self.images)
        if self.town in HIGH_ROAD_THRESHOLDS and info["ego_vehicle_loc"]["z"] > HIGH_ROAD_THRESHOLDS[self.town]:
            sources.update(self.high_images)
        return {name: self.sample(image, xi, yi) for name, image in sources.items()}

    def world_to_local(self, points, info):
        cx, cy, ct, st, z = self.pose(info)
        points = (np.asarray(points, dtype=np.float64) - self.offset) * PIXELS_PER_METER
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        return np.column_stack((dx * ct + dy * st, -dx * st + dy * ct)) + self.center

    def dynamic_masks(self, world, ego, info, future_waypoints):
        masks = {name: np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8) for name in ("vehicle_mask", "walker_mask", "route")}
        footprints = []
        actors = world.get_actors()
        for pattern, name in (("vehicle.*", "vehicle_mask"), ("walker.pedestrian.*", "walker_mask")):
            for actor in actors.filter(pattern):
                if actor.id == ego.id:
                    continue
                transform = actor.get_transform()
                if transform.location.distance(ego.get_location()) > 60.0:
                    continue
                if name == "vehicle_mask":
                    extent = actor.bounding_box.extent
                    half_x = int(max(extent.x * 2, 1.0) * PIXELS_PER_METER) / (2 * PIXELS_PER_METER)
                    half_y = int(max(extent.y * 2, 1.0) * PIXELS_PER_METER) / (2 * PIXELS_PER_METER)
                else:
                    half_x = half_y = 0.7
                theta = math.radians(transform.rotation.yaw)
                rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
                local = np.array([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]])
                points = local @ rotation.T + [transform.location.x, transform.location.y]
                polygon = np.rint(self.world_to_local(points, info)).astype(np.int32)
                cv2.fillConvexPoly(masks[name], polygon, 255)
                footprints.append({"actor_id": actor.id, "mask": name, "world_corners": points.tolist(), "z": transform.location.z})
        if future_waypoints:
            origin = [info["ego_vehicle_loc"]["x"], info["ego_vehicle_loc"]["y"]]
            points = self.world_to_local([origin] + list(future_waypoints[:5]), info)
            cv2.polylines(masks["route"], [np.rint(points).astype(np.int32)], False, 255, thickness=16)
        return masks, footprints

    def render(self, world, ego, info, future_waypoints):
        masks = self.static_masks(info)
        dynamic, footprints = self.dynamic_masks(world, ego, info, future_waypoints)
        masks.update(dynamic)
        return masks, footprints


def validate_maps(root, towns):
    for town in towns:
        maps = BEVMap(root, town)
        del maps
