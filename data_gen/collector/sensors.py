import math
import weakref

import carla
import numpy as np

from collector.frames import FrameBuffer


def transform(config):
    return carla.Transform(
        carla.Location(**{key: config[key] for key in ("x", "y", "z")}),
        carla.Rotation(**{key: config[key] for key in ("pitch", "roll", "yaw")}),
    )


class CameraSensor:
    blueprint_id = "sensor.camera.rgb"

    def __init__(self, parent_actor, config):
        self.frames = FrameBuffer()
        self.frame = None
        self.img = None
        self.color_img = None
        world = parent_actor.get_world()
        blueprint = world.get_blueprint_library().find(self.blueprint_id)
        for source, target in (("width", "image_size_x"), ("height", "image_size_y"), ("fov", "fov")):
            blueprint.set_attribute(target, str(config[source]))
        blueprint.set_attribute("sensor_tick", "0.0")
        self.sensor = world.spawn_actor(blueprint, transform(config), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CameraSensor.on_image(weak_self, event))

    @staticmethod
    def on_image(weak_self, event):
        sensor = weak_self()
        if sensor is None:
            return
        raw = np.frombuffer(event.raw_data, dtype=np.uint8).reshape(event.height, event.width, 4)[:, :, :3].copy()
        color = None
        if sensor.blueprint_id == "sensor.camera.semantic_segmentation":
            event.convert(carla.ColorConverter.CityScapesPalette)
            color = np.frombuffer(event.raw_data, dtype=np.uint8).reshape(event.height, event.width, 4)[:, :, :3].copy()
        sensor.frames.publish(event.frame, (raw, color))

    def wait_for_frame(self, frame, timeout):
        self.img, self.color_img = self.frames.wait(frame, timeout)
        self.frame = frame


class CameraSensor_RGB(CameraSensor):
    blueprint_id = "sensor.camera.rgb"


class CameraSensor_Seg(CameraSensor):
    blueprint_id = "sensor.camera.semantic_segmentation"


class CameraSensor_Depth(CameraSensor):
    blueprint_id = "sensor.camera.depth"


class LidarSensor:
    def __init__(self, parent_actor, config):
        self.frames = FrameBuffer()
        self.frame = None
        self.point_cloud = None
        world = parent_actor.get_world()
        blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for key, default in (("channels", 64), ("range", 100), ("points_per_second", 1200000), ("rotation_frequency", 20)):
            blueprint.set_attribute(key, str(config.get(key, default)))
        blueprint.set_attribute("sensor_tick", "0.0")
        self.sensor = world.spawn_actor(blueprint, transform(config), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LidarSensor.on_data(weak_self, event))

    @staticmethod
    def on_data(weak_self, event):
        sensor = weak_self()
        if sensor is not None:
            points = np.frombuffer(event.raw_data, dtype=np.float32).reshape(-1, 4).copy()
            sensor.frames.publish(event.frame, points)

    def wait_for_frame(self, frame, timeout):
        self.point_cloud = self.frames.wait(frame, timeout)
        self.frame = frame


class ImuSensor:
    def __init__(self, parent_actor):
        self.frames = FrameBuffer()
        self.frame = None
        self.imu_data = None
        world = parent_actor.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.imu")
        blueprint.set_attribute("sensor_tick", "0.0")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(z=2.0)), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ImuSensor.on_data(weak_self, event))

    @staticmethod
    def on_data(weak_self, event):
        sensor = weak_self()
        if sensor is not None:
            sensor.frames.publish(event.frame, {
                "accelerometer": {key: getattr(event.accelerometer, key) for key in ("x", "y", "z")},
                "gyroscope": {key: getattr(event.gyroscope, key) for key in ("x", "y", "z")},
                "compass": event.compass,
            })

    def wait_for_frame(self, frame, timeout):
        self.imu_data = self.frames.wait(frame, timeout)
        self.frame = frame


class CollisionSensor:
    def __init__(self, parent_actor):
        self.history = []
        world = parent_actor.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor.on_data(weak_self, event))

    @staticmethod
    def on_data(weak_self, event):
        sensor = weak_self()
        if sensor is not None:
            impulse = event.normal_impulse
            sensor.history.append((event.frame, math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)))
            if len(sensor.history) > 4000:
                sensor.history.pop(0)

    def get_collision_history(self):
        history = {}
        for frame, intensity in self.history:
            history[frame] = history.get(frame, 0) + intensity
        return history


def synchronized_sensors(world, config):
    sensors = {"imu": world.imu_sensor}
    for name, settings in config.items():
        if name == "lidar":
            sensors["lidar"] = world.lidar_sensor
            continue
        sensors[f"{name}_rgb"] = getattr(world, f"{name}_rgb")
        if settings["seg"] or settings["od"]:
            sensors[f"{name}_seg"] = getattr(world, f"{name}_seg")
        if settings["depth"]:
            sensors[f"{name}_depth"] = getattr(world, f"{name}_depth")
    return sensors
