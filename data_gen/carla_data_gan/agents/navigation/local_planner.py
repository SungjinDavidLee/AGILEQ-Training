from enum import Enum
from collections import deque
import random
import carla
from carla_data_gan.agents.navigation.controller import VehiclePIDController
from carla_data_gan.agents.tools.misc import draw_waypoints, get_speed

class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class LocalPlanner(object):

    def __init__(self, vehicle, opt_dict={}):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None
        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0
        self._sampling_radius = 1.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 1.5
        self._follow_speed_limits = False
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = opt_dict['sampling_radius']
            if 'lateral_control_dict' in opt_dict:
                self._args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                self._args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']
            if 'base_min_distance' in opt_dict:
                self._base_min_distance = opt_dict['base_min_distance']
            if 'follow_speed_limits' in opt_dict:
                self._follow_speed_limits = opt_dict['follow_speed_limits']
        self._init_controller()

    def reset_vehicle(self):
        self._vehicle = None

    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self._vehicle, args_lateral=self._args_lateral_dict, args_longitudinal=self._args_longitudinal_dict, offset=self._offset, max_throttle=self._max_throt, max_brake=self._max_brake, max_steering=self._max_steer)
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed):
        if self._follow_speed_limits:
            print("WARNING: The max speed is currently set to follow the speed limits. Use 'follow_speed_limits' to deactivate this")
        self._target_speed = speed

    def follow_speed_limits(self, value=True):
        self._follow_speed_limits = value

    def _compute_next_waypoints(self, k=1):
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)
        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))
            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]
            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        if clean_queue:
            self._waypoints_queue.clear()
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._stop_waypoint_creation = stop_waypoint_creation

    def run_step(self, debug=False):
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()
        if not self._stop_waypoint_creation and len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed
        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1
            else:
                min_distance = self._min_distance
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)
        return control

    def get_incoming_waypoint_and_direction(self, steps=3):
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]
        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return (wpt, direction)
            except IndexError:
                return (None, RoadOption.VOID)

    def get_plan(self):
        return self._waypoints_queue

    def done(self):
        return len(self._waypoints_queue) == 0

def _retrieve_options(list_waypoints, current_waypoint):
    options = []
    for next_waypoint in list_waypoints:
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)
    return options

def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0
    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0
    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > 180 - threshold:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
