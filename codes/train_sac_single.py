import gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.sac.policies import MlpPolicy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    print("start train")
    env = gym.make('gym_robot_arm:robot-arm-v1')
    #env.seed(0)
    #n_sampled_goal = 4

    model = SAC(MlpPolicy,
        env,
        tensorboard_log="./sac_single_50W_continuous_2/",
        verbose=1
        )
    model.learn(total_timesteps=400000, tb_log_name="run_" + str(1))

    model.save('./sac_single_50W_continuous_2/' + str(400000))
    del model

