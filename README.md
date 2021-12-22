# CSC498_Project
The Final project For CSC498 Introduction to Reinforcement Learning

Original Environment:https://github.com/ekorudiawan/gym-robot-arm.git

# To build the environment
```bash
conda create -n 2d_robot python=3.9
conda activate 2d_robot
pip install gym
pip install pygame
conda install -c anaconda scipy
pip install stable-baselines3[extra]
git clone https://github.com/ekorudiawan/gym-robot-arm.git
cd gym-robot-arm

# Copy files under codes to gym-robot-arm

# Train sac
python train_sac_single.py
# Train td3
python train_td3_single.py
# Test
python test.py

# modify environment
# copy environment file under one of the envs folder to gym-robot-arm/gym-robot-arm/envs
# and replace the file inside

#Show training process:
tensorboard --logdir={dir to the log position} --port 8123
http://localhost:8123/

```





