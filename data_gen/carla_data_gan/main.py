import carla
import numpy as np
from carla_data_gan.agents.navigation.basic_agent import BasicAgent
from carla_data_gan.agents.navigation.local_planner import RoadOption
from carla_data_gan.agents.navigation.behavior_types import Cautious, Aggressive, Normal
from carla_data_gan.agents.tools.misc import get_speed, positive
import numpy.random as random
from collector.sensors import CameraSensor_RGB, CameraSensor_Seg, CameraSensor_Depth, LidarSensor, ImuSensor, CollisionSensor
MIN_WIDTH = 5
MIN_HEIGHT = 5

class BehaviorAgent(BasicAgent):

    def __init__(self, vehicle, behavior='normal'):
        super(BehaviorAgent, self).__init__(vehicle)
        self._look_ahead_steps = 0
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 1.0
        if behavior == 'cautious':
            self._behavior = Cautious()
        elif behavior == 'normal':
            self._behavior = Normal()
        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW
        self._look_ahead_steps = int(self._speed_limit / 10)
        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter('*traffic_light*')
        affected, _ = self._affected_by_traffic_light(lights_list)
        return affected

    def traffic_light_manager2(self):
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter('*traffic_light*')
        affected, _ = self._affected_by_traffic_light2(lights_list)
        return affected

    def set_collision_sensor(self, collision_sensor):
        self._collision_sensor = collision_sensor

    def _tailgating(self, waypoint, vehicle_list):
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()
        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and (right_wpt.lane_type == carla.LaneType.Driving):
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print('Tailgating, moving to the right!')
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location, right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and (left_wpt.lane_type == carla.LaneType.Driving):
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print('Tailgating, moving to the left!')
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location, left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        vehicle_list = self._world.get_actors().filter('*vehicle*')

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW and (not waypoint.is_junction) and (self._speed > 10) and (self._behavior.tailgate_counter == 0):
                self._tailgating(waypoint, vehicle_list)
        return (vehicle_state, vehicle, distance)

    def pedestrian_avoid_manager(self, waypoint):
        walker_list = self._world.get_actors().filter('*walker.pedestrian*')

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
        return (walker_state, walker, distance)

    def car_following_manager(self, vehicle, distance, debug=False):
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0.0, 1.0)
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([positive(vehicle_speed - self._behavior.speed_decrease), self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([max(self._min_speed, vehicle_speed), self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        else:
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        return control

    def location_to_dict(self, loc):
        return {'x': loc.x, 'y': loc.y, 'z': loc.z} if loc else None

    def run_step(self, debug=False):
        self._update_information()
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1
        ego_vehicle_tf = self._vehicle.get_transform()
        ego_vehicle_loc = ego_vehicle_tf.location
        ego_vehicle_rot = ego_vehicle_tf.rotation
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        wp1 = self.waypoint_after_meters(10.0) or ego_vehicle_wp
        wp2 = self.waypoint_after_meters(15.0) or ego_vehicle_wp
        wp3 = self.waypoint_after_meters(20.0) or ego_vehicle_wp
        wp4 = self.waypoint_after_meters(25.0) or ego_vehicle_wp
        info_dict = {'current_speed': self._speed, 'current_waypoint': ego_vehicle_wp.transform.location, 'target_waypoint': self._local_planner.target_waypoint.transform.location if self._local_planner.target_waypoint else None, 'G_waypoints': [wp1.transform.location.x, wp1.transform.location.y], 'G_waypoints2': [wp2.transform.location.x, wp2.transform.location.y], 'G_waypoints3': [wp3.transform.location.x, wp3.transform.location.y], 'G_waypoints4': [wp4.transform.location.x, wp4.transform.location.y], 'future_waypoints': [], 'current_control': {'throttle': self._vehicle.get_control().throttle, 'steer': self._vehicle.get_control().steer, 'brake': self._vehicle.get_control().brake}, 'next_control': {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}, 'traffic_light': self.traffic_light_manager(), 'traffic_light_distance': self.traffic_light_distance(), 'traffic_light_20': False, 'collision': False, 'vehicle_in_front': vehicle_state, 'ego_vehicle_loc': self.location_to_dict(ego_vehicle_loc), 'ego_vehicle_rot': {'pitch': ego_vehicle_rot.pitch, 'yaw': ego_vehicle_rot.yaw, 'roll': ego_vehicle_rot.roll}}
        collision_detected = False
        if hasattr(self, '_collision_sensor'):
            history = self._collision_sensor.get_collision_history()
            recent_frames = [k for k in history if k > self._vehicle.get_world().get_snapshot().frame - 5]
            if len(recent_frames) > 0:
                collision_detected = True
        if collision_detected:
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            info_dict['next_control'] = {'throttle': control.throttle, 'steer': control.steer, 'brake': control.brake}
            info_dict.update({'collision': True, 'traffic_light': self.traffic_light_manager(), 'vehicle_in_front': vehicle_state, 'ego_vehicle_loc': self.location_to_dict(ego_vehicle_loc)})
            return (control, info_dict)
        upcoming_waypoints = self._local_planner._waypoints_queue
        ways = []
        for i in range(min(10, len(upcoming_waypoints))):
            info_dict['future_waypoints'].append(upcoming_waypoints[i][0].transform.location)
        if self.traffic_light_manager2():
            info_dict['traffic_light_20'] = True
        ways = []
        for i in range(min(5, len(upcoming_waypoints))):
            ways.append((upcoming_waypoints[i][0].transform.location.x, upcoming_waypoints[i][0].transform.location.y))
        self.future_waypoints = ways
        control = None
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        if walker_state:
            distance = w_distance - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            if distance < self._behavior.braking_distance:
                control = self.emergency_stop()
                info_dict['next_control'] = {'throttle': control.throttle, 'steer': control.steer, 'brake': control.brake}
                return (control, info_dict)
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        if vehicle_state:
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            if distance < self._behavior.braking_distance:
                control = self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)
            info_dict['next_control'] = {'throttle': control.throttle, 'steer': control.steer, 'brake': control.brake}
            return (control, info_dict)
        elif self._incoming_waypoint.is_junction and self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]:
            target_speed = min([self._behavior.max_speed, self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        else:
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        if self.traffic_light_manager():
            control = self.emergency_stop()
            info_dict['next_control'] = {'throttle': control.throttle, 'steer': control.steer, 'brake': control.brake}
            return (control, info_dict)
        info_dict['next_control'] = {'throttle': control.throttle, 'steer': control.steer, 'brake': control.brake}
        return (control, info_dict)

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

class World:

    def __init__(self, client, conf):
        self.sensors = []
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.player = None
        self.collision_sensor = None
        self.conf = conf
        self.vehicle_tags = [int(getattr(carla.CityObjectLabel, name)) for name in ('Vehicles', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle') if hasattr(carla.CityObjectLabel, name)]
        self.pedestrian_tags = [int(carla.CityObjectLabel.Pedestrians)]
        self.detection_radius = 60.0
        self.light_radius = 30.0
        sensor_cfg = conf
        cam_sensors = {k: v for k, v in sensor_cfg.items() if k != 'lidar'}
        cam_cfg = list(cam_sensors.items())[0][1]
        self._sensor_data = {'width': cam_cfg['width'], 'height': cam_cfg['height'], 'fov': cam_cfg['fov']}
        self._sensor_data['calibration'] = self._get_camera_to_car_calibration(self._sensor_data)
        try:
            self.restart()
        except BaseException:
            self.destroy()
            raise

    def restart(self):
        self.sensors = []
        vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        color = vehicle_bp.get_attribute('color').recommended_values[0]
        vehicle_bp.set_attribute('color', color)
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        for attempt in range(100):
            if self.player is not None:
                break
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        if self.player is None:
            raise RuntimeError('Failed to spawn ego vehicle after 100 attempts')
        self._vehicle = self.player
        self.world.tick()
        self.collision_sensor = CollisionSensor(self.player)
        self.sensors.append(self.collision_sensor.sensor)
        self.imu_sensor = ImuSensor(self.player)
        self.sensors.append(self.imu_sensor.sensor)
        for name in list(self.conf.keys()):
            if name == 'lidar':
                continue
            rgb_sensor = CameraSensor_RGB(self.player, self.conf[name])
            setattr(self, f'{name}_rgb', rgb_sensor)
            self.sensors.append(rgb_sensor.sensor)
            if self.conf[name].get('seg', False) or self.conf[name].get('od', False):
                seg_sensor = CameraSensor_Seg(self.player, self.conf[name])
                setattr(self, f'{name}_seg', seg_sensor)
                self.sensors.append(seg_sensor.sensor)
            if self.conf[name].get('depth', False):
                depth_sensor = CameraSensor_Depth(self.player, self.conf[name])
                setattr(self, f'{name}_depth', depth_sensor)
                self.sensors.append(depth_sensor.sensor)
        if 'lidar' in self.conf:
            self.lidar_sensor = LidarSensor(self.player, self.conf['lidar'])
            self.sensors.append(self.lidar_sensor.sensor)

    def _get_2d_bbs(self, sensor_transform, bb_3d, seg_img):
        results = []
        for vehicle in bb_3d['vehicles']:
            trig_loc_world = self._create_bb_points(vehicle).T
            cords_x_y_z = self._world_to_sensor(trig_loc_world, sensor_transform, False)
            cords_x_y_z = np.array(cords_x_y_z)[:3, :]
            veh_bb = self._coords_to_2d_bb(cords_x_y_z)
            if veh_bb is not None:
                x1, y1 = veh_bb[0]
                x2, y2 = veh_bb[1]
                width = x2 - x1
                height = y2 - y1
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue
                region = seg_img[y1:y2, x1:x2]
                visible_ratio = np.count_nonzero(np.isin(region, self.vehicle_tags)) / region.size if region.size else 0
                if visible_ratio > 0.15:
                    results.append([x1, y1, x2, y2, 0])
        for pedestrian in bb_3d['pedestrians']:
            trig_loc_world = self._create_bb_points(pedestrian).T
            cords_x_y_z = self._world_to_sensor(trig_loc_world, sensor_transform, False)
            cords_x_y_z = np.array(cords_x_y_z)[:3, :]
            ped_bb = self._coords_to_2d_bb(cords_x_y_z)
            if ped_bb is not None:
                x1, y1 = ped_bb[0]
                x2, y2 = ped_bb[1]
                width = x2 - x1
                height = y2 - y1
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue
                region = seg_img[y1:y2, x1:x2]
                visible_ratio = np.count_nonzero(np.isin(region, self.pedestrian_tags)) / region.size if region.size else 0
                if visible_ratio > 0.1:
                    results.append([x1, y1, x2, y2, 1])
        return results

    def _get_3d_bbs(self, max_distance=50):
        bounding_boxes = {'traffic_lights': [], 'stop_signs': [], 'vehicles': [], 'trucks': [], 'bicycles': [], 'pedestrians': []}
        bounding_boxes['traffic_lights'] = self._find_obstacle_3dbb('*traffic_light*', max_distance)
        bounding_boxes['stop_signs'] = self._find_obstacle_3dbb('*stop*', max_distance)
        bounding_boxes['pedestrians'] = self._find_obstacle_3dbb('*walker*', max_distance)
        vehicle_bbs = self._find_obstacle_3dbb('*vehicle*', max_distance, return_with_type=True)
        for bb, type_id in vehicle_bbs:
            if any((x in type_id for x in ['cargo', 'truck'])):
                bounding_boxes['trucks'].append(bb)
            elif any((x in type_id for x in ['bike', 'bicycle', 'motorcycle'])):
                bounding_boxes['bicycles'].append(bb)
            else:
                bounding_boxes['vehicles'].append(bb)
        return bounding_boxes

    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50, return_with_type=False):
        obst = []
        _actors = self.world.get_actors()
        _obstacles = _actors.filter(obstacle_type)
        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(self._vehicle.get_location())
            if 0 < distance_to_car <= max_distance:
                if hasattr(_obstacle, 'bounding_box'):
                    loc = _obstacle.bounding_box.location
                    _obstacle.get_transform().transform(loc)
                    extent = _obstacle.bounding_box.extent
                    _rotation_matrix = self.get_matrix(carla.Transform(carla.Location(0, 0, 0), _obstacle.get_transform().rotation))
                    rotated_extent = np.squeeze(np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3]))
                    bb = np.array([[loc.x, loc.y, loc.z], [rotated_extent[0], rotated_extent[1], rotated_extent[2]]])
                else:
                    loc = _obstacle.get_transform().location
                    bb = np.array([[loc.x, loc.y, loc.z], [0.5, 0.5, 2]])
                if return_with_type:
                    obst.append((bb, _obstacle.type_id))
                else:
                    obst.append(bb)
        return obst

    def _create_bb_points(self, bb):
        cords = np.zeros((8, 4))
        extent = bb[1]
        loc = bb[0]
        cords[0, :] = np.array([loc[0] + extent[0], loc[1] + extent[1], loc[2] - extent[2], 1])
        cords[1, :] = np.array([loc[0] - extent[0], loc[1] + extent[1], loc[2] - extent[2], 1])
        cords[2, :] = np.array([loc[0] - extent[0], loc[1] - extent[1], loc[2] - extent[2], 1])
        cords[3, :] = np.array([loc[0] + extent[0], loc[1] - extent[1], loc[2] - extent[2], 1])
        cords[4, :] = np.array([loc[0] + extent[0], loc[1] + extent[1], loc[2] + extent[2], 1])
        cords[5, :] = np.array([loc[0] - extent[0], loc[1] + extent[1], loc[2] + extent[2], 1])
        cords[6, :] = np.array([loc[0] - extent[0], loc[1] - extent[1], loc[2] + extent[2], 1])
        cords[7, :] = np.array([loc[0] + extent[0], loc[1] - extent[1], loc[2] + extent[2], 1])
        return cords

    def _get_camera_to_car_calibration(self, sensor):
        calibration = np.identity(3)
        calibration[0, 2] = sensor['width'] / 2.0
        calibration[1, 2] = sensor['height'] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor['width'] / (2.0 * np.tan(sensor['fov'] * np.pi / 360.0))
        return calibration

    def _world_to_sensor(self, cords, sensor, move_cords=False):
        sensor_world_matrix = self.get_matrix(sensor)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        if move_cords:
            _num_cords = range(sensor_cords.shape[1])
            modified_cords = np.array([])
            for i in _num_cords:
                if sensor_cords[0, i] < 0:
                    for j in _num_cords:
                        if sensor_cords[0, j] > 0:
                            _direction = sensor_cords[:, i] - sensor_cords[:, j]
                            _distance = -sensor_cords[0, j] / _direction[0]
                            new_cord = sensor_cords[:, j] + _distance[0, 0] * _direction * 0.9999
                            modified_cords = np.hstack([modified_cords, new_cord]) if modified_cords.size else new_cord
                else:
                    modified_cords = np.hstack([modified_cords, sensor_cords[:, i]]) if modified_cords.size else sensor_cords[:, i]
            return modified_cords
        else:
            return sensor_cords

    def get_matrix(self, transform):
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _coords_to_2d_bb(self, cords):
        cords_y_minus_z_x = np.vstack((cords[1, :], -cords[2, :], cords[0, :]))
        bbox = (self._sensor_data['calibration'] @ cords_y_minus_z_x).T
        camera_bbox = np.vstack([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]).T
        if np.any(camera_bbox[:, 2] > 0):
            camera_bbox = np.array(camera_bbox)
            _positive_bb = camera_bbox[camera_bbox[:, 2] > 0]
            min_x = int(np.clip(np.min(_positive_bb[:, 0]), 0, self._sensor_data['width']))
            min_y = int(np.clip(np.min(_positive_bb[:, 1]), 0, self._sensor_data['height']))
            max_x = int(np.clip(np.max(_positive_bb[:, 0]), 0, self._sensor_data['width']))
            max_y = int(np.clip(np.max(_positive_bb[:, 1]), 0, self._sensor_data['height']))
            return [(min_x, min_y), (max_x, max_y)]
        else:
            return None

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        for sensor in getattr(self, 'sensors', []):
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors = []
        if self.player is not None:
            try:
                self.player.destroy()
            except RuntimeError:
                pass
            self.player = None
