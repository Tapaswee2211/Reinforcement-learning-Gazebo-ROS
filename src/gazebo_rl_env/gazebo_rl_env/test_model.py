from stable_baselines3 import PPO
from gazebo_rl_env.gazebo_env import GazeboRLEnv

def main():
    env = GazeboRLEnv()
    model = PPO.load("ppo_gazebo_track", env=env)
    obs, _ = env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()

