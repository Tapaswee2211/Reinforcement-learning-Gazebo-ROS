import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class GazeboRLEnv(Node):
    def __init__(self):
        super().__init__('gazebo_rl_env_node')
        self.publisher_ = self.create_publisher(String, 'rl_agent_topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Gazebo RL Environment node started!')

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from Gazebo RL Environment!'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = GazeboRLEnv()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

