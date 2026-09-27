import os
import subprocess
import sys
import glob
import time
import gym
import gym.spaces
import pygame
import cv2
from pygame.locals import *
import csv
import random
import torch
from pyquaternion import Quaternion

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.navigation.local_planner import _compute_connection
from carla_env.wrappers import *
from carla_env.envs.run_red_light import RunRedLight
from carla_env.envs.traffic_light import TrafficLightHandler

from BEV.way_utils_before import LocalPlanner
from BEV.map_utils import MapImage, PIXELS_PER_METER
from BEV.lts_rendering import Renderer
from torchvision import transforms
import torch.nn.functional as F

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from collections import deque
import itertools

from .model import *
import torchvision.transforms.functional as TF

def pad_or_trim_to_np(x, shape, pad_val=0):
  shape = np.asarray(shape)
  pad = shape - np.minimum(np.shape(x), shape)
  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:shape[0], :shape[1]]


normalize_img = transforms.Compose((
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))
transform_pv = transforms.Compose([
    transforms.Resize((1280, 720)),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])
transform_tensor = transforms.Compose([
    transforms.ToTensor()
])

def vector(v):

    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = cv2.resize(img, resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran

def angle_diff(v0, v1):

    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0

    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    dot_product = np.dot(v0_xy_u, v1_xy_u)
    angle = np.arccos(dot_product)


    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    if abs(angle) >= 2.3:
        return 0
    return round(angle, 2)

def to_u8(img):
    if img is None:
        return None
    if img.dtype == np.bool_:
        return (img.astype(np.uint8) * 255)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2 and img.max() <= 1:
        img = img * 255
    return img

class CarlaRouteEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }
    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1120, 560), obs_res=(160, 80),
                 reward_fn=None,
                 encode_state_fn=None,
                 fps=15, action_smoothing=0.0, action_space_type="continuous",
                 activate_spectator=True,
                 activate_lidar=False,
                 activate_model=False,
                 start_carla=True,
                 eval=False,
                 activate_render=True,
                 observation_space=None,
                 train_town="town05",
                 eval_town="town02"
                 ):

        self.das_04 = to_u8(cv2.imread('BEV/town04/das_full_high.png', cv2.IMREAD_GRAYSCALE))
        self.das_05 = to_u8(cv2.imread('BEV/town05/das_full_high.png', cv2.IMREAD_GRAYSCALE))

        self.lane_04=to_u8(cv2.imread('BEV/town04/lane_full_high.png', cv2.IMREAD_GRAYSCALE))
        self.lane_05=to_u8(cv2.imread('BEV/town05/lane_full_high.png', cv2.IMREAD_GRAYSCALE))

        self.lane_global = to_u8(cv2.imread(f'BEV/{train_town}/lane_full.png', cv2.IMREAD_GRAYSCALE))
        self.das_global  = to_u8(cv2.imread(f'BEV/{train_town}/das_full.png', cv2.IMREAD_GRAYSCALE))
        self.stop_global = to_u8(cv2.imread(f'BEV/{train_town}/stoplines.png', cv2.IMREAD_GRAYSCALE))
        world_offset = np.load(f"BEV/{train_town}/world_offset.npy")
        self.OFFSET_X = float(world_offset[0])
        self.OFFSET_Y = float(world_offset[1])

        self.limit=1000
        if train_town=='town04':
            self.limit=5
        elif train_town=='town05':
            self.limit=7

        self.observation_space=observation_space
        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")
            time.sleep(8)
        self.detection_radius = 60.0
        self.light_radius = 50.0

        width, height = viewer_res
        self.activate_render = activate_render

        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)


        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2
        self.angle=0
        self.speed=0
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000
        self.activate_spectator = activate_spectator
        self.activate_model=activate_model
        self.activate_lidar = activate_lidar
        self.eval = eval
        self.display_seg = None

        self.terminal_reason_counter = {
            "Vehicle stopped" : 0,
            "Off-track" : 0,
            "Too fast": 0,
            "Red light violation" : 0,
            "Success": 0,
            "Collision": 0,
            "Hit walker" : 0,
            "Other": 0,
            "Safe time obstacle": 0
        }

        self.world = None
        self.data_box = []
        self.sensor_list=[]


        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)

        if not self.eval:
            self.world = World(self.client, train_town)
        else:
            self.eval_routes = self._get_eval_routes(eval_town)
            self.world = World(self.client, eval_town)
            print("world set : ", eval_town)

        town_name = self.world.get_map().name.split('/')[-1]
        xodr_path = os.path.join(os.environ.get("CARLA_ROOT", "./"), "CarlaUE4/Content/Carla/Maps/OpenDrive", f"{town_name}.xodr")
        with open(xodr_path, "r") as f:
            xodr_str = f.read()
        world_map_for_bev = carla.Map("BEVMap", xodr_str)

        map_image = MapImage(self.world, world_map_for_bev, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        self.global_map = np.zeros((1, 8,) + road.shape)

        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device='cuda', dtype=torch.float32)

        world_offset = torch.tensor(map_image._world_offset, device='cuda', dtype=torch.float32)
        map_dims = self.global_map.shape[2:4]
        self.renderer = Renderer(world_offset, map_dims, data_generation=True)

        self.bev_image_vis = None
        self.bev_image = None

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1 / self.fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")

        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")

        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.world.map.get_spawn_points()[0])

        self.sensor_list.append(self.vehicle)
        self.traffic_model = TrafficModel()

        self.traffic_model.eval().cuda()
        self.ways_bev=[]


        self.camera_order = [
            "left_cam",
            "front_cam",
            "right_cam",
            "rear_cam",
            "rear_left_cam",
            "rear_cam",
            "rear_right_cam"
        ]

        self.sensor_cfg = {
            "front_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": 1.7, "y": 0.0, "yaw": 0.0, "z": 1.54
            },
            "front_cam_pv": {
                "depth": False, "fov": 100.0, "height": 720, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 1280,
                "x": 1.7, "y": 0.0, "yaw": 0.0, "z": 1.54
            },
            "left_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": 1.53, "y": -0.78, "yaw": -45.0, "z": 1.52
            },
            "lidar": {
                "depth": False, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False,
                "x": 0.0, "y": 0.0, "yaw": 0.0, "z": 1.73
            },
            "rear_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": -1.0, "y": 0.0, "yaw": 180.0, "z": 1.54
            },
            "rear_left_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": -0.78, "y": -0.78, "yaw": -135.0, "z": 1.51
            },
            "rear_right_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": -0.78, "y": 0.78, "yaw": 135.0, "z": 1.51
            },
            "right_cam": {
                "depth": False, "fov": 64.6, "height": 300, "od": False,
                "pitch": 0.0, "roll": 0.0, "seg": False, "width": 400,
                "x": 1.53, "y": 0.78, "yaw": 45.0, "z": 1.52}}

        self.sensor_data = {
            'width': int(self.sensor_cfg['front_cam']['width']),
            'height': int(self.sensor_cfg['front_cam']['height']),
            'fov': int(self.sensor_cfg['front_cam']['fov']),
        }

        self.collision_sensor = CollisionSensor(self.vehicle, call_back=self._on_collision)


        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle)
        self.gnss_sensor = GnssSensor(self.vehicle)
        if self.activate_model:
            self.bev_model = bev_SEG(self.activate_lidar)
            self.bev_model.load_state_dict(torch.load('./run/cls/best_loss.pt'))
            self.bev_model.eval().cuda()
            for name in list(self.sensor_cfg.keys()):
                print(name)
                if name == 'lidar':
                    continue

                rgb_sensor = CameraSensor_RGB(self.vehicle, self.sensor_cfg[name])
                setattr(self, f"{name}_rgb", rgb_sensor)
                self.sensor_list.append(rgb_sensor.sensor)
        else:

            rgb_sensor = CameraSensor_RGB(self.vehicle, self.sensor_cfg['front_cam_pv'])
            setattr(self, f"front_cam_pv_rgb", rgb_sensor)
            self.sensor_list.append(rgb_sensor.sensor)


        if 'lidar' in self.sensor_cfg:
            self.lidar_sensor = LidarSensor(self.vehicle,self.sensor_cfg['lidar'])

        self.traffic_light_handler=TrafficLightHandler()
        self.traffic_light_handler.reset(self.world)
        self.run_red_light = RunRedLight(self.world.get_map(), distance_light=30)

        self.previous_location = self.vehicle.get_transform().location
        self.limit_speed = 0
        self.old_steer= 0

        self.traffic_manager = self.client.get_trafficmanager(8000)

        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)


        self.world.set_pedestrians_cross_factor(1.0)

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)


        self.ignore_lights_prob = 15.0

        self.high_speeding_vehicle_prob = 25.0

        self.high_speeding_amount = -20.0

        self.low_speeding_vehilce_prob = 40.0
        self.low_speeding_amount = 20.0


        self.vehicle_spawn_points = self.world.get_map().get_spawn_points()
        all_vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
        self.vehicle_bp_lib = [bp for bp in all_vehicle_bps if self._is_valid_blueprint(bp)]
        if len(self.vehicle_bp_lib) == 0:
            print("Warning: No valid vehicle blueprints found after filtering! Using default.")
            self.vehicle_bp_lib = self.world.get_blueprint_library().filter('vehicle.tesla.model3')
        self.walker_bp_lib = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        self.controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        self.controller_actors = []
        self.npc_vehicle_list = list()
        self.num_npc_vehicles = 35
        self.num_walkers = 15
        self.successful_spawn_location = []
        self.walker_destinations = {}

        self.i = 0
        self.full_step = 0

        if len(self.vehicle_spawn_points) < self.num_npc_vehicles:
            print(f"Warning: requested {self.num_npc_vehicles} vehicles, but only {len(self.vehicle_spawn_points)} spawn points are available.")
            self.num_npc_vehicles = len(self.vehicle_spawn_points)

        self.vehicle_bp = random.choices(self.vehicle_bp_lib, k=self.num_npc_vehicles)
        self.walker_bp = random.choices(self.walker_bp_lib, k=self.num_walkers)


        self.previous_speed_ms = 0.0
        self.previous_acceleration = 0.0
        self.current_jerk = 0.0

        self.final_vector = None
        self.dist_to_stop = 20.0
        self.right_turn_indicator = False


        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_jerk = 0.0
        self.prev_traffic_light_state = None
        self.prev_speed = 0.0

        self.prev_road_limit = 0.0
        self.prev_dist_to_stop = 20.0

        self.step_measurement = 0
        self.last_walker_reset_step = 0

        self.viewer_image=None
        self.vis_info=None
        self.img_dir = 'pred/img'
        os.makedirs(self.img_dir, exist_ok=True)

        print("Caching traffic light actors...")
        self.traffic_light_list = self.world.get_actors().filter('*traffic_light*')
        print(f"Found {len(self.traffic_light_list)} traffic lights.")

        time.sleep(2)
        self.reset()


    def _get_eval_routes(self, town):
        route_dict = {
            "town01": itertools.cycle([(88, 184), (28, 83), (0, 72), (203, 109)]),
            "town02": itertools.cycle([(61, 39), (28, 83), (48, 21), (0, 72)]),
            "town03": itertools.cycle([(58, 53), (141, 213), (88, 207), (255, 170)]),
            "town04": itertools.cycle([(176, 369), (263, 206)]),
            "town05": itertools.cycle([(198, 45), (227,170), (9,286), (211, 17)]),
        }
        if town not in route_dict:
            raise ValueError(f"Unknown town '{town}' — eval route not defined!")
        return route_dict[town]

    def calculate_route_length(self, route_waypoints):
        route_length = 0.0
        for i in range(1, len(route_waypoints)):
            prev_wp = route_waypoints[i - 1][0].transform.location
            curr_wp = route_waypoints[i][0].transform.location
            route_length += prev_wp.distance(curr_wp)
        return route_length

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def reset(self, is_training=False):

        should_reset_pedestrians = (self.step_measurement > 50000)
        self.time_out = 0.0

        if os.path.exists(self.img_dir):
            for filename in os.listdir(self.img_dir):
                file_path = os.path.join(self.img_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        time.sleep(0.05)


        self._cleanup_vehicles()

        if should_reset_pedestrians:
            self._cleanup_walkers_v4()
            self._verify_and_force_cleanup_walkers()


        self.last_visited_wp_idx = 0
        self.distance_traveled_wp = 0.0

        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        self.local_planner = LocalPlanner(self.vehicle, opt_dict={'max_steering': 1.0})
        self.local_planner.set_global_plan(self.route_waypoints)
        self.pid_controller = self.local_planner._vehicle_controller._lat_controller

        self.route_length = self.calculate_route_length(self.route_waypoints)

        self.terminal_state = False
        self.success_state = False
        self.safe_timeout = False

        self.closed = False
        self.extra_info = []


        self.step_count = 0
        self.step_count2 = 0


        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.more_previous_location = None
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0

        self.limit_speed = 0
        self.collision_true = False
        self.hit_true = False
        self.previous_speed_ms = 0.0
        self.previous_acceleration = 0.0
        self.current_jerk = 0.0
        self.speed = 0.0

        self.final_vector = None
        self.dist_to_stop = 20.0
        self.right_turn_indicator = False


        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_jerk = 0.0
        self.prev_traffic_light_state = None
        self.prev_speed = 0.0

        self.prev_road_limit = 0.0
        self.prev_dist_to_stop = 20.0


        if self.i == 0:
            self.walkers_position_save()
            self.i += 1

        self._spawn_vehicles()

        if should_reset_pedestrians:
            self._spawn_walkers_v2()


        time.sleep(0.5)
        obs = self.step(None)[0]
        time.sleep(0.5)

        return obs

    def new_route(self):

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 0.0
        self.vehicle.apply_control(self.control)
        self.vehicle.set_simulate_physics(False)

        if not self.eval:
            spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(self.eval_routes)]

        route_length = 1
        while route_length <= 1:
            if len(spawn_points_list) == 2:
                self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                                spawn_points_list]
                self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)

            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

    def close(self):
        print("--- [DEBUG] Starting env.close() ---")

        if self.world is not None:
            print("Destroying all tracked actors...")

            self._cleanup_vehicles()
            self._cleanup_walkers_v4()
            self._verify_and_force_cleanup_walkers()

            for actor in self.sensor_list:
                if hasattr(actor, '_destroy'):
                    actor._destroy()
                else:
                    actor.destroy()

            self.sensor_list.clear()
        if self.carla_process:
            print("Terminating CARLA process...")
            self.carla_process.terminate()
            self.carla_process.wait()

        if self.world is not None:
            pass
        self.closed = True


    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        if self.bev_image_vis is not None and self.front_cam_pv_rgb.img is not None:

            self.viewer_image = self.front_cam_pv_rgb.img

            if isinstance(self.ways_bev, torch.Tensor):
                points = self.ways_bev.squeeze(1).cpu().numpy().astype(int)
            else:
                points = self.ways_bev


            bev_crop = self.bev_image_vis[150:350, 150:350]
            bev_resized = cv2.resize(bev_crop, (300, 300), interpolation=cv2.INTER_LINEAR)

            line_height = 35
            num_lines = len(self.vis_info) if self.vis_info else 0
            text_canvas_height = 40 + num_lines * line_height
            text_canvas = np.zeros((text_canvas_height, 300, 3), dtype=np.uint8)

            if self.vis_info:
                for i, (k, v) in enumerate(self.vis_info.items()):
                    text = f"{k}: {v}"
                    y = 40 + i * line_height
                    cv2.putText(text_canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            bev_and_text = np.vstack((bev_resized, text_canvas))

            viewer_h, viewer_w, _ = self.viewer_image.shape
            bev_h = bev_and_text.shape[0]
            target_height = max(viewer_h, bev_h)

            if viewer_h < target_height:
                pad_h = target_height - viewer_h
                viewer_image_padded = cv2.copyMakeBorder(self.viewer_image, 0, pad_h, 0, 0,
                                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                viewer_image_padded = self.viewer_image


            if bev_h < target_height:
                pad_h = target_height - bev_h
                bev_and_text_padded = cv2.copyMakeBorder(bev_and_text, 0, pad_h, 0, 0,
                                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                bev_and_text_padded = bev_and_text

            combined_image = np.hstack((viewer_image_padded, bev_and_text_padded))
            self.lender_image = combined_image
            cv2.imwrite(f'pred/img/image_{self.step_count2}.png', combined_image)

            remove_index = self.step_count2 - 80
            if remove_index >= 0:
                old_image_path = f'pred/img/image_{remove_index}.png'
                if os.path.exists(old_image_path):
                    os.remove(old_image_path)
            self.step_count2 += 1

        if action is not None:
            if self.action_space_type == "continuous":
                steer, dw = [float(a) for a in action]


            if dw > 0:
                throttle = dw
                brake = 0.0
            elif dw < 0:
                throttle = 0.0
                brake = -1 * dw
            else:
                throttle = 0.0
                brake = 0.0

            self.control = carla.VehicleControl()
            self.control.steer = self.local_planner._vehicle_controller.smooth_steering(steer)
            self.control.throttle = throttle
            self.control.brake = brake
            self.vehicle.apply_control(self.control)

        try:
            self.world.tick()
        except RuntimeError as e:
            print(f"World step tick filed: {e}")
            self._tick_safe()

        self.local_planner.run_step()
        self.right_turn_indicator = self.get_right_turn_indicator()

        if self.step_count % 10 == 0:
            self._retask_walkers()


        if self.local_planner.done():

            self.success_state = True

            if self.bev_image is None:
                self.bev_image = torch.zeros((1, 7, 196, 196), device='cuda')
        else:
            if self.activate_model:
                if self.activate_lidar:
                    self.lidar_data = self.lidar_sensor.point_cloud
                    points = self.lidar_data[:, :3]
                    num_points = points.shape[0]
                    lidar_data = pad_or_trim_to_np(points, [81920, 5]).astype('float32')
                    lidar_mask = np.ones(81920).astype('float32')
                    lidar_mask[num_points:] *= 0.0

                imgs, post_rots, post_trans, intrinsic, extrinsic, rotation, translation = self.get_model_input()
                DAS, Lane, Vehicle, walker, stopline, route = self.bev_model(imgs, translation, rotation, intrinsic,
                                                                          post_trans, post_rots, lidar_data, lidar_mask)
                DAS_img = self.sigmoid_to_img(DAS)
                Lane_img = self.sigmoid_to_img(Lane)
                Vehicle_img = self.sigmoid_to_img(Vehicle)
                Walker_img = self.sigmoid_to_img(walker)
                stopline_img = self.sigmoid_to_img(stopline)
                route_img = self.sigmoid_to_img(route)

                h, w = DAS_img.shape
                ego_img = np.zeros((h, w), dtype=np.uint8)
                rect_w, rect_h = 12, 25
                center_x, center_y = w // 2, h // 2
                top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
                bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)
                cv2.rectangle(ego_img, top_left, bottom_right, 255, thickness=-1)

                merged_img = np.zeros((h, w, 3), dtype=np.uint8)
                merged_img2 = np.zeros((h, w, 3), dtype=np.uint8)

                merged_img[DAS_img > 128] = (60, 60, 60)
                merged_img[Lane_img > 128] = (0, 100, 0)
                merged_img[Vehicle_img > 128] = (255, 255, 0)
                merged_img[Walker_img > 128] = (255, 0, 255)
                merged_img[stopline_img > 128] = (200, 200, 200)
                merged_img2[route_img > 128] = (255, 255, 255)
                self.bev_image_vis = cv2.addWeighted(merged_img, 1.0, merged_img2, 0.3, 0)
                cv2.rectangle(self.bev_image_vis, top_left, bottom_right, (255, 255, 255), thickness=-1)

                self.bev_image = np.stack([
                    ego_img, DAS_img, Lane_img, Walker_img, Vehicle_img, stopline_img, route_img
                ], axis=-1)
            else:
                tr = self.vehicle.get_transform()

                data = {
                    "x": float(tr.location.x),
                    "y": float(tr.location.y),
                    "z": float(tr.location.z),
                    "yaw": float(tr.rotation.yaw),
                }


                bev_stack = self.get_bev(self.get_way_v2(), data)

                self.bev_image = bev_stack


                h, w, _ = bev_stack.shape
                vis_img = np.zeros((h, w, 3), dtype=np.uint8)

                vis_img[bev_stack[:,:,1] > 0] = (60, 60, 60)
                vis_img[bev_stack[:,:,2] > 0] = (0, 255, 0)
                vis_img[bev_stack[:,:,5] > 0] = (200, 200, 200)
                vis_img[bev_stack[:,:,6] > 0] = (180, 180, 180)
                vis_img[bev_stack[:,:,4] > 0] = (0, 255, 255)
                vis_img[bev_stack[:,:,3] > 0] = (255, 0, 255)
                vis_img[bev_stack[:,:,0] > 0] = (255, 255, 255)

                self.bev_image_vis = vis_img

            self.bev_image = transform_tensor(self.bev_image).unsqueeze(0).cuda()

        if not self.success_state:
            final_waypoint_location = self.route_waypoints[-1][0].transform.location
            vehicle_location = self.vehicle.get_transform().location

            distance_to_final = vehicle_location.distance(final_waypoint_location)

            if distance_to_final <= 5.0:
                self.success_state = True

        self.viewer_image = self.front_cam_pv_rgb.img

        transform = self.vehicle.get_transform()
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index

        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        velocity = self.vehicle.get_velocity()
        self.speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        self.speed_accum += self.speed

        fwd = vector(self.vehicle.get_velocity())
        wp_fwd = vector(self.current_waypoint.transform.rotation.get_forward_vector())
        self.angle = angle_diff(wp_fwd, fwd)

        tl_state, _, _ = self.traffic_light_handler.get_light_state(self.vehicle)
        self.final_vector = self.get_traffic_light_vector(tl_state, fail_prob=0.05, misclassify_prob=0.2)

        _, dist_to_stop = self.run_red_light.tick(self.vehicle)
        self.dist_to_stop = self._apply_stopline_noise(dist_to_stop)


        tr = self.vehicle.get_transform()
        heading_fwd = vector(tr.get_forward_vector())

        vec_to_next_wp = vector(self.next_waypoint.transform.location) - vector(tr.location)
        self.heading_angle = angle_diff(heading_fwd, vec_to_next_wp)

        current_speed_ms = self.speed / 3.6
        dt = 1.0 / self.fps

        current_acceleration = (current_speed_ms - self.previous_speed_ms) / dt

        if self.step_count > 0:
            self.current_jerk = (current_acceleration - self.previous_acceleration) / dt
        else:
            self.current_jerk = 0.0

        self.previous_speed_ms = current_speed_ms
        self.previous_acceleration = current_acceleration

        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)
        self.last_reward, low_speed_timer= self.reward_fn(self)
        self.total_reward += self.last_reward
        encoded_state = self.encode_state_fn(self)
        self.step_count += 1
        self.step_measurement += 1

        self.prev_steer = self.control.steer
        self.prev_throttle = self.control.throttle
        self.prev_brake = self.control.brake
        self.prev_jerk = self.current_jerk
        self.prev_speed = self.speed

        self.prev_traffic_light_state = self.final_vector
        self.prev_dist_to_stop = self.dist_to_stop

        current_load_limit = self.get_dynamic_target_speed()

        if current_load_limit is None or current_load_limit == 0:
            current_load_limit = 40.0
        self.prev_road_limit = current_load_limit


        current_location = self.vehicle.get_transform().location
        closest_idx = self.get_closest_waypoint_idx(current_location)
        if closest_idx > self.last_visited_wp_idx:
            for i in range(self.last_visited_wp_idx, min(closest_idx, len(self.route_waypoints) - 1)):
                loc1 = self.route_waypoints[i][0].transform.location
                loc2 = self.route_waypoints[i + 1][0].transform.location
                self.distance_traveled_wp += loc1.distance(loc2)
            self.last_visited_wp_idx = closest_idx

        if any(np.isnan(value).any() for value in encoded_state.values()) or \
                any(np.isinf(value).any() for value in encoded_state.values()):
            raise ValueError("Observation contains nan or inf!")
        if np.isnan(self.last_reward) or np.isinf(self.last_reward):
            raise ValueError("Reward is nan or inf!")


        self.vis_info = {
            'total_reward': f"{self.total_reward:.2f}",
            'routes_completed': f"{self.routes_completed:.2f}",
            'avg_center_dev': f"{(self.center_lane_deviation / self.step_count):.2f}",
            'avg_speed': f"{(self.speed_accum / self.step_count):.2f}",
            'mean_reward': f"{(self.total_reward / self.step_count):.2f}",
            'route_completion': f"{(self.distance_traveled_wp / self.route_length):.2f}",
            'traffics': f"{tl_state}",
            'time_out': f"{low_speed_timer}",
        }

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count),
            'route_completion': (self.distance_traveled_wp / self.route_length)
        }

        return encoded_state, self.last_reward, self.terminal_state or self.success_state, info


    def get_traffic_light_vector(self, traffic_state, fail_prob=0.05, misclassify_prob=0.2):

        if random.random() < fail_prob:
            return [0, 0, 0, 1]

        if traffic_state == carla.TrafficLightState.Green:
            correct_vector = [0, 0, 1, 0]
        elif traffic_state == carla.TrafficLightState.Yellow:
            correct_vector = [0, 1, 0, 0]
        elif traffic_state == carla.TrafficLightState.Red:
            correct_vector = [1, 0, 0, 0]
        else:
            correct_vector = [0, 0, 0, 1]


        if random.random() < misclassify_prob and correct_vector != [0, 0, 0, 1]:
            possible_errors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            if correct_vector in possible_errors:
                possible_errors.remove(correct_vector)
            return random.choice(possible_errors)

        return correct_vector

    def _apply_stopline_noise(self, real_dist):

        if real_dist is None or real_dist < 0:
            return 20.0

        if random.random() < 0.20:
            noise = random.uniform(2.0, 5.0)
            if random.random() < 0.5: noise *= -1
        else:
            noise = random.uniform(-0.3, 0.3)

        noisy_dist = max(0.0, real_dist + noise)
        return noisy_dist / 20.0


    def get_way_v2(self):
        ego_pos_list =  [self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y]
        ego_yaw_list =  [self.vehicle.get_transform().rotation.yaw/180*np.pi]


        next_waypoints_state = self.route_waypoints[self.current_waypoint_index: self.current_waypoint_index + 15]

        if next_waypoints_state:
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            ego_pos_batched = []
            ego_yaw_batched = []
            ways=[]
            for i in range(min(len(waypoints), 35)):
                ego_pos_batched.append(ego_pos_list)
                ego_yaw_batched.append(ego_yaw_list)
                ways.append(waypoints[i][:2])

            if ways:
                ways_np = np.array(ways)
                ways_torch = torch.tensor(ways_np, device='cuda', dtype=torch.float32).unsqueeze(1)
                ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
                ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
                ways_torch=self.renderer.world_to_pix_crop_batched(ways_torch, ego_pos_batched_torch, ego_yaw_batched_torch)
                self.ways_bev = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways_torch.device), ways_torch], dim=0)
            else:
                self.ways_bev = torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)
        else:
            self.ways_bev = torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)


        upcoming_waypoints = self.local_planner._waypoints_queue


        max_len = self._calculate_dynamic_len(max_points=8)

        if not upcoming_waypoints:
            return torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)

        ways=[]

        for i in range(min(max_len, len(upcoming_waypoints))):
            ways.append((upcoming_waypoints[i][0].transform.location.x,upcoming_waypoints[i][0].transform.location.y))

        if not ways:
            return torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)
        ways_torch = torch.tensor(ways, device='cuda', dtype=torch.float32).unsqueeze(1)
        ego_pos_batched = []
        ego_yaw_batched = []

        for _ in range(ways_torch.shape[0]):
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)

        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)

        ways_torch=self.renderer.world_to_pix_crop_batched(ways_torch, ego_pos_batched_torch, ego_yaw_batched_torch)
        ways_torch = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways_torch.device), ways_torch], dim=0)

        return ways_torch

    def _calculate_dynamic_len(self, max_points=7):

        min_obstacle_distance = float('inf')

        closest_walker, distance_walker = self._get_closest_hazardous_walker_v2()
        if closest_walker and distance_walker is not None:
            min_obstacle_distance = min(min_obstacle_distance, distance_walker)

        npc_vehicle, npc_distance, _ = self._is_vehicle_hazard()
        if npc_vehicle and npc_distance is not None:
            min_obstacle_distance = min(min_obstacle_distance, npc_distance)

        final_len = max_points
        if min_obstacle_distance < float('inf'):
            obstacle_based_len = max(1, int(min_obstacle_distance))
            final_len = min(final_len, obstacle_based_len)

        return final_len

    def get_truncated_waypoints_state(self, max_points=8, ignore_obstacles=True):

        MAX_WAYPOINT_DIST = 30.0

        upcoming_waypoints = self.local_planner._waypoints_queue

        if ignore_obstacles:
            current_max_len = max_points
        else:

            if hasattr(self, 'valid_waypoint_len'):
                 current_max_len = min(max_points, self.valid_waypoint_len)
            else:

                 current_max_len = max_points

        local_waypoints = []
        ego_transform = self.vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_fwd = ego_transform.get_forward_vector()
        ego_right = ego_transform.get_right_vector()

        iter_len = min(current_max_len, len(upcoming_waypoints))

        for i in range(iter_len):
            wp_loc = upcoming_waypoints[i][0].transform.location
            vec_to_wp = wp_loc - ego_loc

            local_x = vec_to_wp.x * ego_fwd.x + vec_to_wp.y * ego_fwd.y + vec_to_wp.z * ego_fwd.z
            local_y = vec_to_wp.x * ego_right.x + vec_to_wp.y * ego_right.y + vec_to_wp.z * ego_right.z

            norm_x = np.clip(local_x / MAX_WAYPOINT_DIST, -1.0, 1.0)
            norm_y = np.clip(local_y / MAX_WAYPOINT_DIST, -1.0, 1.0)

            local_waypoints.extend([norm_x, norm_y])


        target_len = max_points * 2
        if len(local_waypoints) < target_len:
            local_waypoints.extend([0.0] * (target_len - len(local_waypoints)))

        return local_waypoints


    def sigmoid_to_img(self,tensor):
        return (F.sigmoid(tensor).cpu().detach().numpy()[0, 0] * 255).astype(np.uint8)

    def get_model_input(self,):
        rgb_images = [getattr(self, f"{cam}_rgb").img for cam in self.camera_order]
        resize, resize_dims = self.sample_augmentation()
        imgs = []
        post_rots = []
        post_trans = []

        for img in rgb_images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        imgs = torch.stack(imgs).unsqueeze(0).cuda()
        post_rots = torch.stack(post_rots).unsqueeze(0).cuda()
        post_trans = torch.stack(post_trans).unsqueeze(0).cuda()

        intrinsic, extrinsic, rotation, translation=self.get_cam_para()

        return imgs,post_rots, post_trans,intrinsic, extrinsic, rotation, translation

    def sample_augmentation(self):

        resize = (480 / self.sensor_data['width'], 224 / self.sensor_data['height'])
        resize_dims = (480, 224)
        return resize, resize_dims

    def get_cam_para(self):
        def get_cam_to_ego(dof):
            yaw = dof[5]
            rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
            translation = np.array(dof[:3])[:, None]
            rotation_matrix = rotation.rotation_matrix
            cam_to_ego = np.vstack([
                np.hstack((rotation_matrix, translation)),
                np.array([0, 0, 0, 1])
            ])
            return cam_to_ego, rotation_matrix, translation

        cam_names = ['left_cam', 'front_cam', 'right_cam', 'rear_left_cam','rear_cam', 'rear_right_cam']

        extrinsic_list = []
        rotation_list = []
        translation_list = []
        intrinsic_list = []

        for cam in cam_names:
            cam_dof = [
                self.sensor_cfg[cam]['x'],
                self.sensor_cfg[cam]['y'],
                self.sensor_cfg[cam]['z'],
                self.sensor_cfg[cam]['pitch'],
                self.sensor_cfg[cam]['roll'],
                self.sensor_cfg[cam]['yaw'],
            ]
            cam_to_ego, rot, tran = get_cam_to_ego(cam_dof)
            extrinsic_list.append(torch.from_numpy(cam_to_ego).float().unsqueeze(0))
            rotation_list.append(torch.from_numpy(rot).float().unsqueeze(0))
            translation_list.append(torch.from_numpy(tran).float().unsqueeze(0))

            w = self.sensor_cfg[cam]['width']
            h = self.sensor_cfg[cam]['height']
            fov = self.sensor_cfg[cam]['fov']
            f = w / (2 * np.tan(fov * np.pi / 360))
            Cu = w / 2
            Cv = h / 2
            intrinsic = torch.tensor([
                [f, 0, Cu],
                [0, f, Cv],
                [0, 0, 1]
            ], dtype=torch.float32).unsqueeze(0)
            intrinsic_list.append(intrinsic)

        extrinsic = torch.cat(extrinsic_list, dim=0).unsqueeze(0).cuda()
        rotation = torch.cat(rotation_list, dim=0).unsqueeze(0).cuda()
        translation = torch.cat(translation_list, dim=0).squeeze(-1).unsqueeze(0).cuda()
        intrinsic = torch.cat(intrinsic_list, dim=0).unsqueeze(0).cuda()

        return intrinsic, extrinsic, rotation, translation


    def _on_collision(self, event):
        actor = event.other_actor
        actor_type = actor.type_id

        if get_actor_display_name(actor) != "Road":
            if actor_type.startswith("vehicle."):
                self.collision_true = True
            elif actor_type.startswith("walker.pedestrian"):
                self.hit_true = True
            else:
                self.terminal_state = True


    def _draw_path(self, camera, images):
        image = images.copy()
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        world_2_camera = np.array(camera.sensor.get_transform().get_inverse_matrix())
        image_w = int(camera.sensor.attributes['image_size_x'])
        image_h = int(camera.sensor.attributes['image_size_y'])
        fov = float(camera.sensor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            K = build_projection_matrix(image_w, image_h, fov)

            image_point = get_image_point(waypoint_location, K, world_2_camera)

            if image_point is not None:
                x, y = image_point

                if not (math.isfinite(x) and math.isfinite(y)):
                    continue

                if i == len(self.route_waypoints) - 1:
                    color = (255,0,0)
                else:
                    color = (0, 0, 255)

                image = cv2.circle(image, (int(x), int(y)), radius=3, color=color, thickness=-1)
        return image

    def save_terminal_stats(self, model, town, train=False):
        if train:
            path = f"./results/terminal_reason/train_{model}_termianl_reason_stats_{town}.csv"
        else:
            path = f"./results/terminal_reason/eval_{model}_termianl_reason_stats_{town}.csv"

        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Terminal Reason", "Count"])
            for reason, count in self.terminal_reason_counter.items():
                writer.writerow([reason, count])

    def get_closest_waypoint_idx(self, current_location):
        min_dist = float('inf')
        closest_idx = 0
        for i, (wp, _) in enumerate(self.route_waypoints):
            dist = wp.transform.location.distance(current_location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def _is_vehicle_hazard(self):

        hazard_vehicles = []

        vehicle_list = [self.world.get_actor(id) for id in self.npc_vehicle_list]
        vehicle_list = [v for v in vehicle_list if v is not None]

        ego_vehicle = self.vehicle
        ego_location = ego_vehicle.get_transform().location
        ego_velocity = ego_vehicle.get_velocity()
        ego_yaw = ego_vehicle.get_transform().rotation.yaw
        ego_orientation = np.array([np.cos(np.radians(ego_yaw)), np.sin(np.radians(ego_yaw))])
        ego_position = np.array([ego_location.x, ego_location.y])

        ego_speed_lookahead = max(20, 3.0 * np.linalg.norm([ego_velocity.x, ego_velocity.y]))
        ego_wp = self.world.get_map().get_waypoint(ego_location)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == ego_vehicle.id:
                continue

            npc_location = target_vehicle.get_transform().location
            npc_yaw = target_vehicle.get_transform().rotation.yaw
            npc_wp = self.world.get_map().get_waypoint(npc_location)


            if npc_wp.road_id != ego_wp.road_id or npc_wp.lane_id * ego_wp.lane_id < 0:
                continue

            npc_orientation = np.array([np.cos(np.radians(npc_yaw)), np.sin(np.radians(npc_yaw))])
            npc_position = np.array([npc_location.x, npc_location.y])

            relative_position = npc_position - ego_position
            distance = np.linalg.norm(relative_position)
            direction_to_npc = relative_position / (distance + 1e-4)

            velocity = target_vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            angle_to_npc = np.degrees(np.arccos(np.clip(np.dot(ego_orientation, direction_to_npc), -1.0, 1.0)))
            angle_between_headings = np.degrees(np.arccos(np.clip(np.dot(ego_orientation, npc_orientation), -1.0, 1.0)))

            if angle_between_headings > 60.0 and not (angle_to_npc < 15 and distance < ego_speed_lookahead):
                continue
            elif angle_to_npc > 30.0:
                continue
            elif distance > ego_speed_lookahead:
                continue

            hazard_vehicles.append((distance, target_vehicle, speed_ms))

        if not hazard_vehicles:
            return None, None, None

        hazard_vehicles.sort(key=lambda x: x[0])

        closest_distance = hazard_vehicles[0][0]
        closest_vehicle = hazard_vehicles[0][1]
        closest_vehicle_speed = hazard_vehicles[0][2]


        if closest_distance <= 20.0:
            return closest_vehicle, closest_distance, closest_vehicle_speed
        else:
            return None, None, None


    def _get_closest_hazardous_walker_v2(self):

        closest_walker = None
        min_distance = 15.0
        ego_location = self.vehicle.get_location()
        carla_map = self.world.get_map()

        for controller in self.controller_actors:
            if not controller or not controller.is_alive:
                continue

            walker = controller.parent
            if not walker or not walker.is_alive:
                continue

            walker_location = walker.get_location()
            distance = ego_location.distance(walker_location)

            if distance <= min_distance:
                waypoint = carla_map.get_waypoint(walker_location, project_to_road=False, lane_type=carla.LaneType.Any)

                if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                    distance = ego_location.distance(walker_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_walker = walker

        return closest_walker, min_distance if closest_walker else None

    def get_right_turn_indicator(self, max_points=7):
        waypoints_queue = self.local_planner._waypoints_queue
        target_waypoints = [waypoints_queue[i][0] for i in range(min(max_points, len(waypoints_queue)))]
        for i in range(len(target_waypoints) - 1):
            if _compute_connection(target_waypoints[i], target_waypoints[i + 1]) == RoadOption.RIGHT:
                return True
        return False

    def get_dynamic_target_speed(self):

        speed_limit = self.vehicle.get_speed_limit()
        if speed_limit > 0:
            return speed_limit
        else:
            return None


    def _cleanup_vehicles(self):
        print("--- [DEBUG] Starting _cleanup_vehicles ---")
        destroy_commands = []

        for actor_id in self.npc_vehicle_list:
            destroy_commands.append(carla.command.DestroyActor(actor_id))

        if destroy_commands:
            self.client.apply_batch_sync(destroy_commands, True)
            self._tick_safe()

        self.npc_vehicle_list.clear()


    def _cleanup_walkers_v4(self):
        print("\n" + "="*60)
        print("Cleanup: Destroying walkers based on the tracked controller list...")

        destroy_commands = []

        for controller in self.controller_actors[:]:

            if not controller or not controller.is_alive:
                continue

            try:
                controller.stop()
            except Exception as e:
                print(f"Warning: Failed to stop controller {controller.id}: {e}")
            walker = controller.parent

            if walker and walker.is_alive:
                destroy_commands.append(carla.command.DestroyActor(walker.id))
            destroy_commands.append(carla.command.DestroyActor(controller.id))

        if destroy_commands:
            print(f"Attempting to destroy {len(destroy_commands)} actors (walkers + controllers) from the tracked list.")
            self.client.apply_batch_sync(destroy_commands, True)
            self._tick_safe()

        else:
            print("No actors to destroy.")

        self.controller_actors.clear()
        self.walker_destinations.clear()


        actors_after = self.world.get_actors()
        walkers_after = actors_after.filter("walker.pedestrian.*")
        controllers_after = actors_after.filter("controller.ai.walker")
        print(f"[After Cleanup] Walkers in world: {len(walkers_after)}, Controllers in world: {len(controllers_after)}")
        print("Cleanup based on list complete.")
        print("="*60 + "\n")

    def _verify_and_force_cleanup_walkers(self):

        walkers = self.world.get_actors().filter('*.walker.pedestrian.*')
        controllers = self.world.get_actors().filter('*.controller.ai.walker.*')

        if len(walkers) == len(controllers):
            return

        print(f"!! Walker/Controller mismatch detected ({len(walkers)}/{len(controllers)}). Forcing cleanup...")

        destroy_commands = []
        for controller in controllers:
            controller.stop()
            destroy_commands.append(carla.command.DestroyActor(controller.id))
        for walker in walkers:
            destroy_commands.append(carla.command.DestroyActor(walker.id))

        if destroy_commands:
            self.client.apply_batch_sync(destroy_commands, True)
            time.sleep(0.5)
            self._tick_safe()

    def _spawn_vehicles(self):

        available_spawn_points = [p for p in self.vehicle_spawn_points if p.location.distance(self.previous_location) > 5]
        random.shuffle(available_spawn_points)

        vehicle_batch = []

        for bp, transform in zip(self.vehicle_bp, available_spawn_points):
            bp.set_attribute('role_name', 'autopilot')

            vehicle_batch.append(carla.command.SpawnActor(bp, transform).then(
                 carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port())
            ))

        vehicle_results = self.client.apply_batch_sync(vehicle_batch, True)
        self._tick_safe()

        new_vehicle_ids = [r.actor_id for r in vehicle_results if not r.error]
        self.npc_vehicle_list.extend(new_vehicle_ids)

        print("Applying settings to each spawned vehicle")

        for actor_id in new_vehicle_ids:
            vehicle_actor = self.world.get_actor(actor_id)
            if not vehicle_actor:
                continue

            self.traffic_manager.ignore_lights_percentage(vehicle_actor, self.ignore_lights_prob)

            if random.uniform(0, 100) < self.high_speeding_vehicle_prob:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle_actor, self.high_speeding_amount)
            elif random.uniform(0, 100) < self.low_speeding_vehilce_prob:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle_actor, self.low_speeding_amount)

        print("Applying settings to each spawned vehicle complete")

        for r in vehicle_results:
            if r.error:
                print(f"Failed to spawn vehicle. Reason: {r.error}")

        print(f"Successfully spawned {len(new_vehicle_ids)} vehicles.")

    def _is_valid_blueprint(self, blueprint):

        bp_id = blueprint.id


        bicycle_blacklist = [
            'vehicle.bh.crossbike',
            'vehicle.diamondback.century',
            'vehicle.gazelle.omafiets',
        ]

        if bp_id in bicycle_blacklist:
            return False


        if not blueprint.has_attribute('base_type'):
            return False

        return True

    def _spawn_walkers_v2(self):
        if not self.successful_spawn_location or len(self.successful_spawn_location) < self.num_walkers:
            raise ValueError("Not enough cached walker spawn locations. Run the cache generation logic first.")

        walker_ids = []
        batch_size = 10

        for i in range(0, self.num_walkers, batch_size):
            walker_batch = []
            for bp, location in zip(self.walker_bp[i:i+batch_size], self.successful_spawn_location[i:i+batch_size]):
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                walker_batch.append(carla.command.SpawnActor(bp, location))

            if not walker_batch:
                continue

            print(f"Spawning walker batch {i//batch_size + 1}...")
            walker_results = self.client.apply_batch_sync(walker_batch, True)
            self._tick_safe()

            for r in walker_results:
                if r.error:
                    print(f"Failed to spawn walker. Reason: {r.error}")
                else:
                    walker_ids.append(r.actor_id)


        print("controller spawn start")

        controller_batch = [carla.command.SpawnActor(self.controller_bp, carla.Transform(), parent_id) for parent_id in walker_ids]
        print("controller spawn command")
        controller_results = self.client.apply_batch_sync(controller_batch, True)
        self._tick_safe()

        for r in controller_results:
            if r.error:
                print(f"Failed to spawn controller. Reason: {r.error}")

        self.controller_actors = [self.world.get_actor(r.actor_id) for r in controller_results if not r.error]
        self.walker_destinations.clear()

        print("controller setup")
        for controller in self.controller_actors:
            controller.start()
            destination = self.world.get_random_location_from_navigation()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            self.walker_destinations[controller.id] = destination
            controller.set_max_speed(1 + random.random())

        time.sleep(0.1)
        self.world.tick()

        print(f"Successfully spawned {len(walker_ids)} walkers and {len(self.controller_actors)} controllers.")


    def walkers_position_save(self):
        all_temp_actor_ids = []

        while len(self.successful_spawn_location) < self.num_walkers:
            walkers_to_spawn = self.num_walkers - len(self.successful_spawn_location)

            self.pedestrian_batch = []
            spawn_points_this_loop = []

            for _ in range(walkers_to_spawn):
                spawn_point = carla.Transform()
                location = self.world.get_random_location_from_navigation()
                if location:
                    spawn_point.location = location
                    walker_bp = random.choice(self.walker_bp_lib)
                    command = carla.command.SpawnActor(walker_bp, spawn_point)
                    self.pedestrian_batch.append(command)
                    spawn_points_this_loop.append(spawn_point)

            if not self.pedestrian_batch:
                print("No more walkers can be spawned. Current walker count:", len(self.successful_spawn_location), "Retrying...")
                self._tick_safe()
                continue

            pedestrian_results = self.client.apply_batch_sync(self.pedestrian_batch, True)
            self._tick_safe()

            temp_actor_ids = []
            for i, r in enumerate(pedestrian_results):
                if not r.error:
                    self.successful_spawn_location.append(spawn_points_this_loop[i])
                    temp_actor_ids.append(r.actor_id)

            if temp_actor_ids:
                print(f"{len(temp_actor_ids)} locations found. Total so far: {len(self.successful_spawn_location)} locations")
                all_temp_actor_ids.extend(temp_actor_ids)


        if all_temp_actor_ids:
            print(f"Destroying {len(all_temp_actor_ids)} temporary walkers.")
            destroy_commands = [carla.command.DestroyActor(actor_id) for actor_id in all_temp_actor_ids]
            self.client.apply_batch_sync(destroy_commands, True)
        self._tick_safe()


        print("Walker spawn location cache created.")


    def _retask_walkers(self):
        for controller in self.controller_actors:
            if not controller or not controller.is_alive:
                continue

            walker = controller.parent
            if not walker or not walker.is_alive:
                continue

            current_location = walker.get_location()
            target_destination = self.walker_destinations.get(controller.id)

            if target_destination and current_location.distance(target_destination) < 2.0:
                new_destination = self.world.get_random_location_from_navigation()
                if new_destination:
                    controller.go_to_location(new_destination)
                    self.walker_destinations[controller.id] = new_destination
                    print(f"Retasked walker {walker.id} to new destination.")
                else:
                    print(f"Failed to get a new destination for walker {walker.id}.")


    def _check_path_collision_and_cut(self, waypoints_tensor, danger_mask):

        if waypoints_tensor is None or len(waypoints_tensor) == 0:
            return 0


        pts = waypoints_tensor.squeeze(1).cpu().numpy().astype(np.int32)

        h, w = danger_mask.shape
        valid_length = len(pts)

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i+1]

            if not (0 <= p1[0] < w and 0 <= p1[1] < h):
                valid_length = i
                break


            dist = np.linalg.norm(p2 - p1)
            steps = max(1, int(dist))

            collision_found = False
            for s in range(steps + 1):
                t = s / steps
                curr_x = int(p1[0] * (1-t) + p2[0] * t)
                curr_y = int(p1[1] * (1-t) + p2[1] * t)

                if not (0 <= curr_x < w and 0 <= curr_y < h):
                    continue

                if danger_mask[curr_y, curr_x] > 0:
                    valid_length = i + 1
                    collision_found = True
                    break

            if collision_found:
                break

        return valid_length

    def get_bev(self, ways, data):
        x = data["x"]
        y = data["y"]
        z = data["z"]
        yaw = data["yaw"]

        cx, cy = world_to_pixel(x, y, self.OFFSET_X, self.OFFSET_Y, 5.0)
        w = OUT_SIZE
        h = OUT_SIZE


        ego_img = np.zeros((h, w), dtype=np.uint8)
        rect_w, rect_h = 12, 25
        center_x, center_y = w // 2, h // 2
        top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
        bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)
        cv2.rectangle(ego_img, top_left, bottom_right, (255), thickness=-1)


        if z > self.limit:
            das_img = crop_yaw_nn(self.das_04, (cx, cy), yaw, fill=0)
            lane_img = crop_yaw_nn(self.lane_04, (cx, cy), yaw, fill=0)
        else:
            das_img = crop_yaw_nn(self.das_global, (cx, cy), yaw, fill=0)
            lane_img = crop_yaw_nn(self.lane_global, (cx, cy), yaw, fill=0)


        stopline_img = crop_yaw_nn(self.stop_global, (cx, cy), yaw, fill=0)


        data_rad = data.copy()
        data_rad['yaw'] = math.radians(data['yaw'])


        pos_batched = []
        for controller in self.controller_actors:
            if controller and controller.is_alive:
                walker = controller.parent
                if walker and walker.is_alive:
                    if (walker.get_location().distance(self.vehicle.get_location()) < self.detection_radius):
                        pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])

        walker_img = make_ped_bev_ego_aligned(data_rad, pos_batched, out_size=OUT_SIZE, out_mpp=0.20, box_px=8, ped_value=180)


        pos_batched = []
        vehicles = [self.world.get_actor(actor_id) for actor_id in self.npc_vehicle_list]
        for v in vehicles:
            if v and v.is_alive and v.id != self.vehicle.id:
                if v.get_location().distance(self.vehicle.get_location()) < self.detection_radius:
                    bb = v.bounding_box
                    bb_center = v.get_transform().transform(bb.location)
                    yaw_rad = math.radians(v.get_transform().rotation.yaw)
                    pos_batched.append([bb_center.x, bb_center.y, yaw_rad, 2.0*bb.extent.x, 2.0*bb.extent.y])

        vehicle_img = make_vehicle_bev_ego_aligned(data_rad, pos_batched, out_size=OUT_SIZE)


        danger_mask = np.zeros((h, w), dtype=np.uint8)

        combined_danger = np.zeros((h, w), dtype=np.uint8)
        combined_danger[vehicle_img > 0] = 255
        combined_danger[walker_img > 0] = 255


        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        danger_mask = cv2.dilate(combined_danger, kernel, iterations=1)


        if hasattr(self, 'final_vector') and self.final_vector is not None:
             if self.final_vector[0] == 1 or self.final_vector[1] == 1:
                 danger_mask[stopline_img > 0] = 255


        self.valid_waypoint_len = self._check_path_collision_and_cut(ways, danger_mask)


        route_birdview = np.zeros((w, h), dtype=np.uint8)

        if ways is not None and len(ways) > 0:
            cut_ways = ways[:self.valid_waypoint_len]

            if len(cut_ways) > 0:
                cv2.polylines(
                    route_birdview,
                    [np.round(cut_ways.cpu().numpy()).astype(np.int32)],
                    False,
                    1,
                    thickness=18
                )
        route_birdview = route_birdview * 255


        stacked_img = np.stack([
            ego_img, das_img, lane_img, walker_img, vehicle_img, stopline_img, route_birdview
        ], axis=-1)

        return stacked_img


    def _tick_safe(self):
        try:
            time.sleep(0.5)
            self.world.tick()

        except RuntimeError as e:
            print(f"World tick failed: {e}")

            for i in range(5):
                try:
                    print(f"Retrying tick ({i+1}/3)...")
                    time.sleep(1)
                    self.world.tick()
                    print("Tick successful on retry.")
                    return
                except RuntimeError:
                    continue


            print("\n[Warning] Tick continues to fail. Attempting to destroy ALL walkers to recover...")

            try:
                destroy_commands = []

                for controller in list(self.controller_actors):
                    if controller and controller.is_alive:
                        try: controller.stop()
                        except: pass

                        if controller.parent and controller.parent.is_alive:
                            destroy_commands.append(carla.command.DestroyActor(controller.parent.id))
                        destroy_commands.append(carla.command.DestroyActor(controller.id))


                all_walkers = self.world.get_actors().filter('*walker.pedestrian.*')
                all_controllers = self.world.get_actors().filter('*controller.ai.walker.*')

                for w in all_walkers: destroy_commands.append(carla.command.DestroyActor(w.id))
                for c in all_controllers: destroy_commands.append(carla.command.DestroyActor(c.id))

                if destroy_commands:
                    print(f" -> Sending destroy commands for {len(destroy_commands)} actors...")

                    self.client.apply_batch(destroy_commands)

                self.controller_actors.clear()
                self.walker_destinations.clear()

                time.sleep(2.0)

                print("Retrying tick after emergency walker cleanup...")
                self.world.tick()
                print("✅ Tick successful! Episode continues without walkers.")
                return

            except Exception as e2:
                print(f"Recovery failed: {e2}")
                raise RuntimeError(f"CRITICAL: Simulator unresponsive even after walker cleanup. Original error: {e}")


def draw_obb(mask, cx, cy, hl, hw, theta, value=255):
    c = math.cos(theta)
    s = math.sin(theta)

    pts = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ], dtype=np.float32)

    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy

    pts_i = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, pts_i, int(value))
def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
def make_vehicle_bev_ego_aligned(
    data, npc_agents,
    out_size=200,
    out_mpp=0.20,
    npc_value=255
    ):
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    c0 = out_size / 2.0

    ex, ey = data['x'], data['y']
    cy = data['yaw']
    c = math.cos(cy)
    s = math.sin(cy)

    for npc in npc_agents:
        nx, ny, nyaw, nlen_m, nwid_m = npc

        car_len_px = max(1, int(round(nlen_m / out_mpp)))
        car_wid_px = max(1, int(round(nwid_m / out_mpp)))
        hl = car_len_px * 0.5
        hw = car_wid_px * 0.5

        dx = nx - ex
        dy = ny - ey


        f =  c * dx + s * dy
        r = -s * dx + c * dy

        ox = c0 + (r / out_mpp)
        oy = c0 - (f / out_mpp)

        if ox < -car_len_px or ox > out_size + car_len_px or oy < -car_len_px or oy > out_size + car_len_px:
            continue

        rel_yaw = normalize_angle(nyaw - data['yaw'])
        theta = -math.pi/2 + rel_yaw
        draw_obb(mask, ox, oy, hl, hw, theta=theta, value=npc_value)

    return mask

def make_ped_bev_ego_aligned(data, ped_agents,
                             out_size=200,
                             out_mpp=0.20,
                             box_px=3,
                             ped_value=180):

    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    c0 = out_size / 2.0

    ex, ey = data['x'], data['y']
    cy = data['yaw']
    c = math.cos(cy)
    s = math.sin(cy)

    half = box_px // 2

    for ped in ped_agents:
        px, py = ped[0], ped[1]

        dx = px - ex
        dy = py - ey


        f =  c * dx + s * dy
        r = -s * dx + c * dy

        ox = c0 + (r / out_mpp)
        oy = c0 - (f / out_mpp)

        if ox < -2 or ox > out_size + 2 or oy < -2 or oy > out_size + 2:
            continue

        ix = int(round(ox))
        iy = int(round(oy))

        x0 = max(0, ix - half)
        x1 = min(out_size, ix + half + 1)
        y0 = max(0, iy - half)
        y1 = min(out_size, iy + half + 1)

        mask[y0:y1, x0:x1] = ped_value

    return mask

def world_to_pixel(x, y, OFFSET_X, OFFSET_Y, pixels_per_meter):
    px = (x - OFFSET_X) * pixels_per_meter
    py = (y - OFFSET_Y) * pixels_per_meter
    return int(px), int(py)

OUT_SIZE = 500
ANGLE_BIAS_DEG = 90.0

c0 = (OUT_SIZE - 1) / 2.0
u = np.arange(OUT_SIZE, dtype=np.float32)
v = np.arange(OUT_SIZE, dtype=np.float32)
uu, vv = np.meshgrid(u, v)
DU = uu - c0
DV = vv - c0


def crop_yaw_nn(img_u8, center_px, yaw_deg, angle_bias_deg=ANGLE_BIAS_DEG, fill=0):

    if img_u8 is None:
        return None

    H, W = img_u8.shape[:2]
    cx, cy = float(center_px[0]), float(center_px[1])

    theta = math.radians(float(yaw_deg) + float(angle_bias_deg))
    ct = math.cos(theta)
    st = math.sin(theta)


    xg = cx + (DU * ct - DV * st)
    yg = cy + (DU * st + DV * ct)

    xi = np.rint(xg).astype(np.int32)
    yi = np.rint(yg).astype(np.int32)

    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

    if img_u8.ndim == 2:
        out = np.full((OUT_SIZE, OUT_SIZE), int(fill), dtype=np.uint8)
        out[inside] = img_u8[yi[inside], xi[inside]]
        return out
    else:
        C = img_u8.shape[2]
        out = np.full((OUT_SIZE, OUT_SIZE, C), int(fill), dtype=np.uint8)
        out[inside] = img_u8[yi[inside], xi[inside]]
        return out
