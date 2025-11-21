"""
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class GazeboRLEnv(Node, gym.Env):
    def __init__(self):
        rclpy.init()
        Node.__init__(self, 'gazebo_rl_training_env')
        gym.Env.__init__(self)

        self.publisher_ = self.create_publisher(Float64, '/robot/control', 10)
        self.subscription = self.create_subscription(Float64, '/robot/state', self.state_callback, 10)

        self.state = np.zeros(3)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def state_callback(self, msg):
        self.state = np.array([msg.data, 0.0, 0.0])

    def step(self, action):
        msg = Float64()
        msg.data = float(action[0])
        self.publisher_.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)

        reward = -abs(self.state[0])  # simple negative distance reward
        done = abs(self.state[0]) < 0.01
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(3)
        return self.state, {}

    def close(self):
        rclpy.shutdown()
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class GazeboRLEnv(Node, gym.Env):
    def __init__(self, ns=''):
        rclpy.init()
        Node.__init__(self, 'gazebo_rl_training_env')
        gym.Env.__init__(self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # observation: [x, y, yaw, min_scan_range]
        low = np.array([-np.inf, -np.inf, -np.pi, 0.0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.pi, 30.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # action: linear_x, angular_z
        self.action_space = spaces.Box(low=np.array([0.0, -1.5], dtype=np.float32),
                                       high=np.array([0.6, 1.5], dtype=np.float32),
                                       dtype=np.float32)

        self.odom = None
        self.scan = None
        self._last_distance_to_center = None

        # client for reset
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_simulation service...')

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.odom = (px, py, yaw)

    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        # replace inf/nan with large number
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max if hasattr(msg, 'range_max') else 30.0)
        self.scan = ranges

    def _get_obs(self):
        if self.odom is None or self.scan is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        px, py, yaw = self.odom
        min_range = float(np.min(self.scan))
        return np.array([px, py, yaw, min_range], dtype=np.float32)

    def step(self, action):
        # publish command
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        # allow one physics step / callbacks
        rclpy.spin_once(self, timeout_sec=0.1)

        obs = self._get_obs()

        # reward design:
        # + forward progress (approx by linear velocity)
        # - penalty for being too close to obstacles
        forward = twist.linear.x
        dist_penalty = 0.0
        if obs[3] < 0.3:
            dist_penalty = 5.0  # collision imminent
        reward = forward * 1.0 - dist_penalty

        done = False
        if obs[3] < 0.12:  # collision
            done = True
            reward -= 20.0

        return obs, float(reward), bool(done), False, {}

    def reset(self, seed=None, options=None):
        # call reset simulation service
        req = Empty.Request()
        self.reset_client.call_async(req)
        # wait short time for reset to happen and topics to repopulate
        rclpy.spin_once(self, timeout_sec=0.5)
        # clear internal state
        self.odom = None
        self.scan = None
        # wait until we have initial messages
        timeout = 5.0
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while (self.odom is None or self.scan is None) and (self.get_clock().now().nanoseconds * 1e-9 - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        obs = self._get_obs()
        return obs, {}

    def close(self):
        self.destroy_node()
        rclpy.shutdown()

