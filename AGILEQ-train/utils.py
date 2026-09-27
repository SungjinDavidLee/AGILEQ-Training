import cv2
import math
import json
import wandb
import os
from collections import deque

import gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class SaveOnBestRouteCompletionCallback(BaseCallback):

    def __init__(self, log_dir: str, window_size: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_completion")

        self.completion_window = deque(maxlen=window_size)
        self.best_mean_completion = -np.inf
        self.completed_episodes = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if np.any(self.locals["dones"]):

            for info in self.locals["infos"]:

                if "route_completion" in info:
                    self.completed_episodes += 1
                    current_completion = info["route_completion"]
                    self.completion_window.append(current_completion)

                    mean_completion = np.mean(self.completion_window)

                    if self.verbose > 0:
                        print(f"\n[Callback] Episode {self.completed_episodes} End! (Timesteps: {self.num_timesteps})")
                        print(f"[Callback] This Ep Completion: {current_completion*100:.1f}% | Window({len(self.completion_window)}) Mean: {mean_completion*100:.1f}%")

                    if mean_completion > self.best_mean_completion:
                        self.best_mean_completion = mean_completion
                        if self.verbose > 0:
                            print(f"[Callback] 🏆 New Best Route Completion ({mean_completion*100:.1f}%)! Saving model...")

                        self.model.save(os.path.join(self.save_path, "best_model.zip"))
        return True


class ActionRewardLoggerCallback(BaseCallback):

    def __init__(self, save_path, save_freq=2000, verbose=0):

        super(ActionRewardLoggerCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

        self.data = {
            "steps": [],
            "steer": [],
            "dw": [],
            "rewards": []
        }

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")

        if actions is not None and rewards is not None:
            steer = float(actions[0][0])
            dw = float(actions[0][1])
            reward = float(rewards[0])

            self.data["steps"].append(self.num_timesteps)
            self.data["steer"].append(steer)
            self.data["dw"].append(dw)
            self.data["rewards"].append(reward)


            if self.n_calls % self.save_freq == 0:
                self._save_to_json(verbose=False)

        return True

    def _on_training_end(self) -> None:
        print(f"[INFO] Training ended. Final saving to {self.save_path}...")
        self._save_to_json(verbose=True)

    def _save_to_json(self, verbose=False):
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.data, f)

            if verbose:
                print(f"[INFO] Log saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save logs: {e}")


def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, int(fps), (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


class HParamCallback(BaseCallback):
    def __init__(self, config):

        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            elif isinstance(v, dict):
                hparam_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    hparam_dict[k][k_inner] = v_inner.__str__()
                hparam_dict[k] = str(hparam_dict[k])
            else:
                hparam_dict[k] = v.__str__()

        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):


    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            infos = self.locals['infos'][0]


            self.logger.record("custom/routes_completed", infos['routes_completed'])
            self.logger.record("custom/total_distance", infos['total_distance'])
            self.logger.record("custom/avg_center_dev", infos['avg_center_dev'])
            self.logger.record("custom/avg_speed", infos['avg_speed'])

            try:
                terminal_counter = self.training_env.get_attr("terminal_reason_counter")[0]

                filter_keys = ["too-fast", "Too fast"]

                data = []
                for reason, count in terminal_counter.items():
                    if reason not in filter_keys:
                        data.append([reason, count])

                table = wandb.Table(data=data, columns=["Reason", "Count"])

                wandb.log({
                    "custom/terminal_reasons_bar_chart": wandb.plot.bar(
                        table, "Reason", "Count", title="Terminal Reason Distribution"
                    )
                })

            except Exception as e:
                print(f"[Warning] Failed to log wandb bar chart: {e}")

            self.logger.dump(self.num_timesteps)


        elif self.num_timesteps % 1000 == 0:
            self.logger.dump(self.num_timesteps)

        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_path, frame_size, video_length=-1, fps=30, skip_frame=1, verbose=0):
        super().__init__(verbose)
        self.video_recorder = VideoRecorder(video_path, frame_size, fps)
        self.max_length = video_length
        self.skip_frame = skip_frame

    def _on_step(self) -> bool:
        if self.max_length != -1 and self.num_timesteps > self.max_length:
            self.video_recorder.release()
            return False
        if self.num_timesteps % self.skip_frame != 0:
            return True
        display = self.training_env.unwrapped.envs[0].env.display
        frame = np.array(pygame.surfarray.array3d(display), dtype=np.uint8).transpose([1, 0, 2])

        self.video_recorder.add_frame(frame)
        return True

    def _on_training_end(self) -> None:
        self.video_recorder.release()


def lr_schedule(initial_value: float, end_value: float, rate: float):


    def func(progress_remaining: float) -> float:

        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"

    return func


class HistoryWrapperObsDict(gym.Wrapper):


    def __init__(self, env: gym.Env, horizon: int = 2, obs_key: str = 'vae_latent') -> object:
        self.obs_key = obs_key
        assert isinstance(env.observation_space.spaces[obs_key], gym.spaces.Box)
        print("Wrapping the env with HistoryWrapperObsDict.")
        wrapped_obs_space = env.observation_space.spaces[self.obs_key]
        wrapped_action_space = env.action_space

        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))


        env.observation_space.spaces[obs_key] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):

        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset()
        obs = obs_dict[self.obs_key]
        self.obs_history[..., -obs.shape[-1]:] = obs

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict[self.obs_key]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict, reward, done, info


class FrameSkip(gym.Wrapper):


    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        print("Wrapping the env with FrameSkip.")
        self._skip = skip

    def step(self, action: np.ndarray):

        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


def parse_wrapper_class(wrapper_class_str: str):

    wrap_class, wrap_params = wrapper_class_str.split("_", 1)
    wrap_params = wrap_params.split("_")
    wrap_params = [int(param) if param.isnumeric() else param for param in wrap_params]

    if wrap_class == "HistoryWrapperObsDict":
        return HistoryWrapperObsDict, wrap_params
    elif wrap_class == "FrameSkip":
        return FrameSkip, wrap_params
