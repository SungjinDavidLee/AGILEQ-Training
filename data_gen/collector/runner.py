import random
import socket
import sys
from pathlib import Path

import carla
import numpy as np
import yaml

from carla_data_gan.main import BehaviorAgent, World
from collector.bev import BEVMap, validate_maps
from collector.config import load_config
from collector.sensors import synchronized_sensors
from collector.server import launch_server, stop_server, wait_for_server
from collector.traffic import WEATHERS, spawn_vehicles, spawn_walkers, town_npc_settings
from collector.writer import new_scene, write_sample


REQUIRED_INFO = (
    "ego_vehicle_loc", "current_speed", "current_waypoint", "target_waypoint",
    "future_waypoints", "current_control", "next_control", "traffic_light",
    "traffic_light_20", "traffic_light_distance", "vehicle_in_front", "collision",
    "ego_vehicle_rot", "G_waypoints",
)


def cleanup_actors(client, world, owned_ids):
    if not owned_ids:
        return
    try:
        for actor in world.get_actors(owned_ids):
            if actor.type_id == "controller.ai.walker":
                try:
                    actor.stop()
                except RuntimeError as error:
                    print(f"Controller cleanup: {error}", file=sys.stderr)
        responses = client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in owned_ids], False)
        for response in responses:
            if response.error:
                print(f"Actor cleanup: {response.error}", file=sys.stderr)
    except RuntimeError as error:
        print(f"Actor cleanup: {error}", file=sys.stderr)


def make_ego(client, sensors):
    ego = World(client, sensors)
    try:
        agent = BehaviorAgent(ego.player, behavior="normal")
        agent.set_collision_sensor(ego.collision_sensor)
        destinations = ego.map.get_spawn_points()
        if not destinations:
            raise RuntimeError("Map contains no vehicle spawn points")
        agent.set_destination(random.choice(destinations).location)
        return ego, agent, destinations
    except BaseException:
        ego.destroy()
        raise


def collect_town(client, traffic_manager, town, config, args):
    bev_map = BEVMap(args.map_root, town)
    world = client.load_world(town)
    previous = world.get_settings()
    ego = None
    owned_ids = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_respawn_dormant_vehicles(True)
        world.set_pedestrians_seed(args.seed)
        vehicle_count, walker_count = town_npc_settings.get(town, (40, 20))
        spawn_vehicles(client, world, traffic_manager, args.vehicles if args.vehicles is not None else vehicle_count, owned_ids)
        spawn_walkers(client, world, args.walkers if args.walkers is not None else walker_count, owned_ids)
        ego, agent, destinations = make_ego(client, config["sensors"])
        scene = new_scene(args.output, town, config, vars(args))
        count, ticks, idle_ticks = 0, 0, 0
        width = max(5, len(str(config["data_num"])) + 1)
        while count < config["data_num"]:
            ticks += 1
            idle_ticks += 1
            if idle_ticks > args.max_idle_ticks:
                raise RuntimeError(f"No complete sample for {args.max_idle_ticks} ticks in {town}")
            frame = world.tick(args.timeout)
            if agent.done():
                agent.set_destination(random.choice(destinations).location)
            control, info = agent.run_step()
            control.manual_gear_shift = False
            control.steer = float(np.clip(control.steer + random.uniform(-args.steer_noise, args.steer_noise), -1, 1))
            info["next_control"] = {key: getattr(control, key) for key in ("throttle", "steer", "brake")}
            ego.player.apply_control(control)
            weather_changed = args.weather_every > 0 and ticks % args.weather_every == 0
            if info.get("collision", False) or weather_changed:
                ego.destroy()
                ego = None
                if weather_changed:
                    world.set_weather(random.choice(WEATHERS))
                ego, agent, destinations = make_ego(client, config["sensors"])
                scene = new_scene(args.output, town, config, vars(args))
                continue
            if ticks % args.save_every:
                continue
            if not all(key in info and info[key] is not None for key in REQUIRED_INFO) or not agent.future_waypoints:
                continue
            sensors = synchronized_sensors(ego, config["sensors"])
            for name, sensor in sensors.items():
                try:
                    sensor.wait_for_frame(frame, args.sensor_timeout)
                except (TimeoutError, RuntimeError) as error:
                    raise RuntimeError(f"{name}: {error}") from error
            snapshot = world.get_snapshot()
            if snapshot.frame != frame:
                raise RuntimeError("World advanced outside the collector; use a dedicated synchronous server")
            write_sample(
                scene, f"{count:0{width}d}_a", ego, agent, info, config["sensors"],
                frame, snapshot.timestamp.elapsed_seconds,
                {name: sensor.frame for name, sensor in sensors.items()}, bev_map,
            )
            count += 1
            idle_ticks = 0
            if count == 1 or count % 100 == 0 or count == config["data_num"]:
                print(f"{town}: {count}/{config['data_num']} samples | {scene.name}", flush=True)
    finally:
        if ego is not None:
            ego.destroy()
        cleanup_actors(client, world, owned_ids)
        try:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(previous)
        except RuntimeError as error:
            print(f"World cleanup: {error}", file=sys.stderr)


def run(args, config):
    validate_maps(args.map_root, config["towns"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    output_config = output / "sensor_config.yaml"
    if output_config.exists():
        existing = load_config(output_config)
        if existing["sensors"] != config["sensors"]:
            raise ValueError("Output contains a different sensor configuration; use a new --output directory")
    else:
        output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    process = None
    try:
        if not args.connect:
            for port in (args.port, args.port + 1, args.port + 2, args.tm_port):
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError as error:
                        raise RuntimeError(f"Port {port} is busy; select different --port/--tm-port or use --connect") from error
            process = launch_server(args.carla_root, args.port, 1.0 / args.fixed_delta, output / "carla_server.log")
        client = carla.Client(args.host, args.port)
        wait_for_server(client, process, args.startup_timeout)
        client.set_timeout(args.timeout)
        client_version, server_version = client.get_client_version(), client.get_server_version()
        if client_version != server_version:
            raise RuntimeError(f"CARLA Python client {client_version} does not match server {server_version}")
        traffic_manager = client.get_trafficmanager(args.tm_port)
        for town in config["towns"]:
            collect_town(client, traffic_manager, town, config, args)
    finally:
        stop_server(process)
