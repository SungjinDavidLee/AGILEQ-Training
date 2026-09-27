import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule


import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class XtMaCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=192, states_neurons=[256]):
        super().__init__(observation_space, features_dim=features_dim)


    def forward(self, observations):

        return observations['outtoken']

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

policy_kwargs = dict(
    features_extractor_class=XtMaCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=192, states_neurons=[256]),

    net_arch=[400,300],

)


algorithm_params = {
    "PPO": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        gamma=0.99,
        batch_size=256,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=policy_kwargs,
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=30000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=policy_kwargs,

    ),
    "DDPG": dict(
        gamma=0.99,
        buffer_size=300000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=policy_kwargs,

    ),
    "TD3": dict(
        gamma=0.99,
        batch_size=256,
        learning_starts=10000,
        buffer_size=30000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        train_freq=64,
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=policy_kwargs,

    ),

    "TQC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=30000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.02,
        gradient_steps=64,
        learning_starts=10000,
        train_freq=64,
        use_sde=True,
        policy_kwargs=policy_kwargs,
    ),

    "crossq": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=30000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        policy_delay=2,
        train_freq=(64,"step"),
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=policy_kwargs,
    ),
}

states = {
    "test": ["out_token"],

    "2": ["steer", "throttle", "speed", "maneuver"],

}

reward_params = {
    "reward_params": dict(
        early_stop=True,
        min_speed=20.0,
        max_speed=35.0,
        target_speed=25.0,
        max_distance=2.0,
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
}

CONFIG_PPO = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_SAC = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_DDPG = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_TD3 = {
    "algorithm": "TD3",
    "algorithm_params": algorithm_params["TD3"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_TQC = {
    "algorithm": "TQC",
    "algorithm_params": algorithm_params["TQC"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (240, 160),
    "seed": 100,
    "wrappers": []
}

CONFIG_CROSSQ = {
    "algorithm": "crossq",
    "algorithm_params": algorithm_params["crossq"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIGS = {
    "PPO": CONFIG_PPO,
    "SAC": CONFIG_SAC,
    "DDPG": CONFIG_DDPG,
    "TD3": CONFIG_TD3,
    "TQC": CONFIG_TQC,
    "crossq": CONFIG_CROSSQ,
}


'''
        self.cnn = nn.Sequential(   # input : (6, 196, 196)
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2), # (8, 96, 96)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2), # (16, 46, 46)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), # (32, 21, 21)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2), # (32, 10, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # (64, 4, 4)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64, 2, 2)
            nn.ReLU(),
            nn.Flatten(), # 64 * 2 * 2
        )
'''
