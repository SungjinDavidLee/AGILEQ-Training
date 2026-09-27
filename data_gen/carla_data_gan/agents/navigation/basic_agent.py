import carla
from shapely.geometry import Polygon
from carla_data_gan.agents.navigation.local_planner import LocalPlanner
from carla_data_gan.agents.navigation.global_route_planner import GlobalRoutePlanner
from carla_data_gan.agents.tools.misc import get_speed, is_within_distance, get_trafficlight_trigger_location, compute_distance
import numpy as np

class BasicAgent(object):

    def __init__(self, vehicle, target_speed=20, opt_dict={}):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None
        self._last_traffic_light2 = None
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._target_speed = target_speed
        self._sampling_resolution = 1.0
        self._base_tlight_threshold = 2.0
        self._base_vehicle_threshold = 5.0
        self._max_brake = 0.5
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_brake' in opt_dict:
            self._max_steering = opt_dict['max_brake']
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

    def waypoint_ahead_on_lane(self, meters: float, step: float=2.0) -> carla.Waypoint:
        origin = self._vehicle.get_transform().location
        wp = self._map.get_waypoint(origin, project_to_road=True, lane_type=carla.LaneType.Driving)
        dist = 0.0
        while dist + step < meters:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            dist += step
        rem = meters - dist
        if rem > 0:
            nxt = wp.next(rem)
            if nxt:
                wp = nxt[0]
        return wp

    def waypoint_on_route_distance(self, destination: carla.Location, meters: float):
        origin = self._vehicle.get_transform().location
        route_trace = self._global_planner.trace_route(origin, destination)
        if not route_trace:
            return None
        locs = [wp.transform.location for wp, _ in route_trace]
        acc = 0.0
        for i in range(1, len(locs)):
            p0, p1 = (locs[i - 1], locs[i])
            seg = p0.distance(p1)
            if acc + seg >= meters:
                t = (meters - acc) / max(seg, 1e-06)
                x = p0.x + t * (p1.x - p0.x)
                y = p0.y + t * (p1.y - p0.y)
                z = p0.z + t * (p1.z - p0.z)
                target_loc = carla.Location(x=x, y=y, z=z)
                return self._map.get_waypoint(target_loc, project_to_road=True)
            acc += seg
        return route_trace[-1][0]

    def waypoint_after_meters(self, meters: float, destination: carla.Location=None):
        if destination is None:
            return self.waypoint_ahead_on_lane(meters)
        return self.waypoint_on_route_distance(destination, meters)

    def add_emergency_stop(self, control):
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        return self._local_planner

    def get_global_planner(self):
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False
        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)
        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        self._local_planner.set_global_plan(plan, stop_waypoint_creation=stop_waypoint_creation, clean_queue=clean_queue)

    def trace_route(self, start_waypoint, end_waypoint):
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        hazard_detected = False
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter('*vehicle*')
        lights_list = actor_list.filter('*traffic_light*')
        vehicle_speed = get_speed(self._vehicle) / 3.6
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True
        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)
        return control

    def done(self):
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        self._ignore_vehicles = active

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        if self._ignore_traffic_lights:
            return (False, None)
        if not lights_list:
            lights_list = self._world.get_actors().filter('*traffic_light*')
        if not max_distance:
            max_distance = self._base_tlight_threshold
        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue
            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)
        return (False, None)

    def _affected_by_traffic_light2(self, lights_list=None, max_distance=20):
        if self._ignore_traffic_lights:
            return (False, None)
        if not lights_list:
            lights_list = self._world.get_actors().filter('*traffic_light*')
        if not max_distance:
            max_distance = self._base_tlight_threshold
        if self._last_traffic_light2:
            if self._last_traffic_light2.state != carla.TrafficLightState.Red:
                self._last_traffic_light2 = None
            else:
                return (True, self._last_traffic_light2)
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue
            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light2 = traffic_light
                return (True, traffic_light)
        return (False, None)

    def traffic_light_distance(self, lights_list=None):
        if self._ignore_traffic_lights:
            return 0.0
        if not lights_list:
            lights_list = self._world.get_actors().filter('*traffic_light*')
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0:
                continue
            target_vector = np.array([object_waypoint.transform.location.x - self._vehicle.get_transform().location.x, object_waypoint.transform.location.y - self._vehicle.get_transform().location.y])
            norm_target = np.linalg.norm(target_vector)
            return norm_target
        return 0.0

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        if self._ignore_vehicles:
            return (False, None, -1)
        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter('*vehicle*')
        if not max_distance:
            max_distance = self._base_vehicle_threshold
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(x=ego_extent * ego_forward_vector.x, y=ego_extent * ego_forward_vector.y)
        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue
                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(x=target_extent * target_forward_vector.x, y=target_extent * target_forward_vector.y)
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])
                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break
                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])
                if len(route_bb) < 3:
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue
                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)
                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                return (False, None, -1)
        return (False, None, -1)
