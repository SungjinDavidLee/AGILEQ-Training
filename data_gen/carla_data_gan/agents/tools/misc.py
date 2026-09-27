import math
import numpy as np
import carla

def draw_waypoints(world, waypoints, z=0.5):
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_trafficlight_trigger_location(traffic_light):

    def rotate_point(point, radians):
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y
        return carla.Vector3D(rotated_x, rotated_y, point.z)
    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent
    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)
    return carla.Location(point_location.x, point_location.y, point_location.z)

def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    target_vector = np.array([target_transform.location.x - reference_transform.location.x, target_transform.location.y - reference_transform.location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target < 0.001:
        return True
    if norm_target > max_distance:
        return False
    if not angle_interval:
        return True
    min_angle = angle_interval[0]
    max_angle = angle_interval[1]
    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1.0, 1.0)))
    return min_angle < angle < max_angle

def compute_magnitude_angle(target_location, current_location, orientation):
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1.0, 1.0)))
    return (norm_target, d_angle)

def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y
    return math.sqrt(x * x + y * y)

def vector(location_1, location_2):
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return [x / norm, y / norm, z / norm]

def compute_distance(location_1, location_2):
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def positive(num):
    return num if num > 0.0 else 0.0
