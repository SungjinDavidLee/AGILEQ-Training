from typing import List
import random
import numpy.random as npr
import carla
WEATHERS = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon, carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.SoftRainSunset]

def get_blueprints(world: carla.World, pattern: str, generation: str='All') -> List:
    bps = world.get_blueprint_library().filter(pattern)
    if generation.lower() == 'all' or len(bps) == 1:
        return bps
    try:
        gen = int(generation)
        return [bp for bp in bps if int(bp.get_attribute('generation')) == gen]
    except ValueError:
        print(f"[WARN] 잘못된 generation='{generation}' → 전체 사용")
        return bps

def spawn_vehicles(client, world: carla.World, tm: carla.TrafficManager, num_veh: int, owned_ids) -> List[int]:
    blueprints = get_blueprints(world, 'vehicle.*')
    blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4 and (not bp.id.endswith(('microlino', 'carlacola', 'cybertruck', 't2', 'sprinter', 'firetruck', 'ambulance', 'bus')))]
    spawn_points = world.get_map().get_spawn_points()
    npr.shuffle(spawn_points)
    num_veh = min(num_veh, len(spawn_points))
    batch = []
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    for transform in spawn_points[:num_veh]:
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        if bp.has_attribute('driver_id'):
            bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm.get_port())))
    resp = client.apply_batch_sync(batch, True)
    ids = [r.actor_id for r in resp if not r.error]
    owned_ids.extend(ids)
    for a in world.get_actors(ids):
        tm.update_vehicle_lights(a, True)
    return ids

def spawn_walkers(client, world, num_walk, owned_ids):
    blueprints = get_blueprints(world, "walker.pedestrian.*")
    batch, speeds = [], []
    for _ in range(num_walk):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        speed = float(blueprint.get_attribute("speed").recommended_values[1]) if blueprint.has_attribute("speed") else 0.0
        batch.append(carla.command.SpawnActor(blueprint, carla.Transform(location)))
        speeds.append(speed)
    walkers = []
    for response, speed in zip(client.apply_batch_sync(batch, True), speeds):
        if not response.error:
            owned_ids.append(response.actor_id)
            walkers.append((response.actor_id, speed))
    blueprint = world.get_blueprint_library().find("controller.ai.walker")
    batch = [carla.command.SpawnActor(blueprint, carla.Transform(), actor_id) for actor_id, _ in walkers]
    controllers = []
    for response, (_, speed) in zip(client.apply_batch_sync(batch, True), walkers):
        if not response.error:
            owned_ids.append(response.actor_id)
            controllers.append((response.actor_id, speed))
    world.set_pedestrians_cross_factor(0.0)
    for actor_id, speed in controllers:
        controller = world.get_actor(actor_id)
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination is not None:
            controller.go_to_location(destination)
        controller.set_max_speed(speed)
    return [actor_id for actor_id, _ in walkers], [actor_id for actor_id, _ in controllers]

town_npc_settings = {'Town01': (30, 20), 'Town02': (30, 25), 'Town03': (40, 25), 'Town04': (50, 30), 'Town05': (50, 30), 'Town06': (25, 15), 'Town07': (20, 10), 'Town10': (40, 30)}
