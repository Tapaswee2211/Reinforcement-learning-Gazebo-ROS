"""
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
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gazebo_rl_env.gazebo_env import GazeboRLEnv

def main():
    env = GazeboRLEnv()
    check_env(env, warn=True)

    try:
        model = PPO.load("ppo_gazebo_track", env=env)
        print("Loaded existing model.")
    except Exception:
        print("Training new model.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.001,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
        )


    model.learn(total_timesteps=500_000)
    model.save("ppo_gazebo_track")
    print(" Model saved as 'ppo_gazebo_track.zip'")

    env.close()

if __name__ == "__main__":
    main()

