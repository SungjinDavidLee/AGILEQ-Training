import argparse
import math
from pathlib import Path

from collector.config import load_config


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Collect synchronized CARLA camera images and BEV labels from YAML")
    parser.add_argument("--config", default=str(ROOT / "data/sensor_config.yaml"))
    parser.add_argument("--output", default=str(ROOT / "data"))
    parser.add_argument("--carla-root", default=str(ROOT / "carla"))
    parser.add_argument("--connect", action="store_true", help="Use an existing dedicated CARLA server; requested towns will be loaded")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--map-root", default=str(ROOT / "data5"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--towns", nargs="+")
    parser.add_argument("--vehicles", type=int)
    parser.add_argument("--walkers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--weather-every", type=int, default=1000)
    parser.add_argument("--steer-noise", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--max-idle-ticks", type=int, default=2000)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    for name in ("samples", "save_every", "timeout", "startup_timeout", "sensor_timeout", "max_idle_ticks"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("vehicles", "walkers", "weather_every", "seed"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.seed >= 2**32:
        parser.error("--seed must be below 2**32")
    if not 0 < args.fixed_delta <= 0.1:
        parser.error("--fixed-delta must be greater than 0 and at most 0.1")
    if not 0 <= args.steer_noise <= 1:
        parser.error("--steer-noise must be between 0 and 1")
    if not 1024 <= args.port <= 65533 or not 1024 <= args.tm_port <= 65535 or args.tm_port in range(args.port, args.port + 3):
        parser.error("Choose distinct unprivileged RPC and traffic manager ports")
    if not args.connect and args.host not in ("127.0.0.1", "localhost"):
        parser.error("Remote --host requires --connect")
    for name in ("config", "output", "carla_root", "map_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.samples is not None:
        config["data_num"] = args.samples
    if args.towns:
        import re
        if len(set(args.towns)) != len(args.towns) or any(not re.fullmatch(r"[A-Za-z0-9_]+", town) for town in args.towns):
            raise ValueError("--towns must contain unique safe CARLA map names")
        config["towns"] = args.towns
    if args.check_config:
        from collector.bev import validate_maps
        validate_maps(args.map_root, config["towns"])
        cameras = [name for name in config["sensors"] if name != "lidar"]
        print(f"Configuration OK: towns={config['towns']}, samples_per_town={config['data_num']}, cameras={cameras}, lidar={'lidar' in config['sensors']}")
        return
    from collector.runner import run
    run(args, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
