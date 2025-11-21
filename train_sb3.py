import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gazebo_rl_env.gazebo_env import GazeboRLEnv

def main():
    env = GazeboRLEnv()
    # optional check
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("ppo_gazebo_track")
    env.close()

if __name__ == "__main__":
    main()

