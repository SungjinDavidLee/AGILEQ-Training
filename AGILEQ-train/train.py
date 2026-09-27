import warnings
import os
import wandb

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from config import CONFIGS
import time
from utils import ActionRewardLoggerCallback
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2002, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timestep to train for")
parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--num_checkpoints", type=int, default=12, help="Checkpoint frequency")
parser.add_argument("--config", type=str, default="TQC", help="PPO, SAC, DDPG, TD3, TQC, crossq")
parser.add_argument("--save_dir", type=str, default="./results", help="Save directory name")
parser.add_argument("--town", type=str, default="town05", help="town")

args = vars(parser.parse_args())
CONFIG = CONFIGS[args["config"]]

from sb3_contrib import TQC, CrossQ
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from crossq_pp.crossq_pp import CrossQpp

from carla_env.envs.carla_env_train import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards_crossq_up import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class, SaveOnBestRouteCompletionCallback

from dataclasses import dataclass, field


@dataclass
class WandbCallbackConfig:
    verbose: bool = True
    gradient_save_freq: int = 100
    model_save_freq: int = 50000

wandb_callback = WandbCallbackConfig()


log_dir = args["save_dir"]
os.makedirs(log_dir, exist_ok=True)
reload_model = args["reload_model"]
total_timesteps = args["total_timesteps"]

seed = CONFIG["seed"]

algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC,
                    "TD3": TD3, "TQC": TQC, "crossq": CrossQ,
                    "crossq_pp" : CrossQpp}

if CONFIG["algorithm"] not in algorithm_dict:
    raise ValueError(f"Invalid algorithm name: {CONFIG['algorithm']}")

AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

observation_space, encode_state_fn = create_encode_state_fn(CONFIG["state"])

print("args no render : ", args["no_render"])

wandb.init(
    entity="pyc05040504-konkuk-university",
    project="carla-crossq-project",
    config=CONFIG,
    sync_tensorboard=True
)

env = CarlaRouteEnv(host=args["host"], port=args["port"],
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=observation_space,
                    encode_state_fn=encode_state_fn,
                    fps=args["fps"], action_smoothing=CONFIG["action_smoothing"], train_town=args["town"],
                    action_space_type='continuous', activate_spectator=False, activate_render= args["no_render"])

obs = env.reset()
print("Observation type:", type(obs))
if isinstance(obs, dict):
    print("Observation keys:", obs.keys())


for wrapper_class_str in CONFIG["wrappers"]:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)


if reload_model == "":
    model = AlgorithmRL('MultiInputPolicy',env,  seed=seed, verbose=0, tensorboard_log=log_dir,device='cuda',**CONFIG["algorithm_params"])
    model_suffix = f"{int(time.time())}_id{args['config']}"
else:
    model = AlgorithmRL.load(reload_model, env=env, device='cuda', seed=seed, verbose=0,**CONFIG["algorithm_params"])
    model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"

model_name = f'{model.__class__.__name__}_{model_suffix}'

model_dir = os.path.join(log_dir, model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))


log_json_path = os.path.join(model_dir, "training_log_dw_reward.json")

action_logger = ActionRewardLoggerCallback(save_path=log_json_path, save_freq=1000)

best_completion_callback = SaveOnBestRouteCompletionCallback(
    log_dir=model_dir,
    window_size=100
)

try:
    print("[INFO] Starting model training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1,
                callback=[
                    HParamCallback(CONFIG),
                    TensorboardCallback(1),
                    CheckpointCallback(
                        save_freq=total_timesteps // args["num_checkpoints"],
                        save_path=model_dir,
                        name_prefix="model"
                    ),
                    action_logger,
                    best_completion_callback,
                    WandbCallback(
                        verbose=wandb_callback.verbose,
                        gradient_save_freq=wandb_callback.gradient_save_freq,
                        model_save_freq=wandb_callback.model_save_freq,
                        model_save_path=f"models/{model_name}",
                    )

                ],
                reset_num_timesteps=False)

    env.save_terminal_stats(model=CONFIG["algorithm"], town=args["town"], train=True)

except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user (Ctrl+C).")

except RuntimeError as e:
    print(f"\n[CRITICAL] A RuntimeError occurred: {e}")
    print("[CRITICAL] This likely means the CARLA simulator has crashed (core dumped).")

finally:
    print("[INFO] Closing the CARLA environment. This may take a moment...")
    env.close()

print("[INFO] Training script finished.")
