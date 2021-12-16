
import gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import statistics
from time import sleep

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    print("start train")
    env = gym.make('gym_robot_arm:robot-arm-v1')
    env.seed(0)
    model_name = 'test_models/level'
    model = SAC.load(model_name)
    print(model_name)
    obs = env.reset()
    ep_reward_list = []
    whether_done_list = []
    for i_episode in range(50):
        episode_reward = 0
        observation = env.reset()
        done = False
        for t in range(50):
            #env.render()
            #sleep(0.1)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Reward:", episode_reward)
                break
        ep_reward_list.append(episode_reward)
        whether_done_list.append(done)
    env.close()

    # print sum of reward and ration of success tasks
    print(ep_reward_list)
    print(statistics.mean(ep_reward_list))
    print(whether_done_list)
    print(sum(whether_done_list)/len(whether_done_list))