import gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    print("start train")
    env = gym.make('gym_robot_arm:robot-arm-v1')
    #env.seed(0)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy",
        env,
        action_noise=action_noise,
        tensorboard_log="./td3_single_20W_ori/",
        verbose=1
        )
    print("start learn")
    model.learn(total_timesteps=200000, tb_log_name="run_" + str(1))

    model.save('./td3_single_20W_ori/' + str(200000))
    del model
