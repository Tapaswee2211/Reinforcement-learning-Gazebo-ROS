import rclpy
from gazebo_rl_env.gazebo_env import GazeboRLEnv
import numpy as np

def main(args=None):
    env_node = GazeboRLEnv()

    for episode in range(5):
        state, _ = env_node.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = env_node.action_space.sample()
            state, reward, done, _, _ = env_node.step(action)
            total_reward += reward

        env_node.get_logger().info(f"Episode {episode+1}: Total reward = {total_reward}")

    env_node.close()

if __name__ == "__main__":
    main()

