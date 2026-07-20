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
################################################################################
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import numpy as np
import gymnasium as gym
import math

class GazeboRLEnv(Node, gym.Env):
    def __init__(self, ns=''):

        rclpy.init()
        Node.__init__(self, 'gazebo_rl_training_env')
        gym.Env.__init__(self)

        # ----------------------------
        # Initialize placeholders first
        # ----------------------------
        self.laser_data = np.ones(360, dtype=np.float32) * 10.0

        self.odom = None
        self.goal_position = np.array([2.0, 0.0])   # example goal (2m in front)
        self.scan = None
        self._last_distance_to_goal = None

        # ----------------------------
        # ROS2 publishers/subscribers
        # ----------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ----------------------------
        # Define action & observation spaces
        # ----------------------------
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=(len(self.laser_data),), dtype=np.float32
        )

        # ----------------------------
        # Reset service
        # ----------------------------
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_simulation service...')

    # ----------------------------
    # Laser callback
    # ----------------------------
    def scan_cb(self, msg):
        self.laser_data = np.array(msg.ranges, dtype=np.float32)
        if len(self.laser_data) != 360:
            self.laser_data = np.resize(self.laser_data, (360,))

    # ----------------------------
    # Odometry callback
    # ----------------------------
    def odom_cb(self, msg):
        self.odom = msg

    # ----------------------------
    # Observation getter
    # ----------------------------
    def _get_obs(self):
        # Replace inf/nan with max range
        clean_scan = np.nan_to_num(self.laser_data, nan=10.0, posinf=10.0, neginf=0.0)
        # Clip to range [0, 10]
        clean_scan = np.clip(clean_scan, 0.0, 10.0)
        return clean_scan.astype(np.float32)


    # ----------------------------
    # Goal check function
    # ----------------------------
    def _check_goal_reached(self):
        if self.odom is None:
            return False
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        goal_dist = math.sqrt((self.goal_position[0] - x) ** 2 + (self.goal_position[1] - y) ** 2)
        return goal_dist < 0.3  # reached if within 0.3m

    # ----------------------------
    # Step function (your reward logic)
    # ----------------------------
    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        rclpy.spin_once(self, timeout_sec=0.1)

        obs = self._get_obs()

        # Reward components
        forward_speed = twist.linear.x
        laser_min_distance = float(np.min(obs))
        laser_min_distance_penalty = max(0, (0.5 - laser_min_distance))
        stuck_penalty = 1.0 if abs(forward_speed) < 0.01 else 0.0

        collision = laser_min_distance < 0.12
        reached_goal = self._check_goal_reached()

        reward = 0
        reward += forward_speed * 0.1
        reward -= laser_min_distance_penalty * 0.5
        reward -= stuck_penalty * 1.0
        if collision:
            reward -= 10
        if reached_goal:
            reward += 20

        done = collision or reached_goal

        reward = np.clip(reward, -20.0, 20.0)
        if np.any(np.isnan(obs)) or np.isnan(reward):
            print("[WARNING] NaN detected in obs or reward")

        return obs, float(reward), bool(done), False, {}


    # ----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        req = Empty.Request()
        self.reset_client.call_async(req)
        rclpy.spin_once(self, timeout_sec=0.5)
        obs = self._get_obs()
        return obs, {}

    def close(self):
        self.destroy_node()
        rclpy.shutdown()

##############################################################################

"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import math
import time
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy


class GazeboRLEnv(Node, gym.Env):
    def __init__(self):
        # Initialize ROS2 and Gym
        rclpy.init()
        Node.__init__(self, 'gazebo_rl_training_env')
        gym.Env.__init__(self)

        # ----------------------------
        # Observation placeholders
        # ----------------------------
        self.laser_data = np.ones(360, dtype=np.float32) * 10.0
        self.current_position = np.array([0.0, 0.0])
        self.prev_position = np.array([0.0, 0.0])
        self.goal_position = np.array([2.0, 0.0])
        self.stuck_steps = 0
        self.collision_threshold = 0.25
        self.odom = None

        # ----------------------------
        # ROS2 Publishers & Subscribers
        # ----------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # ----------------------------
        # Reset Service
        # ----------------------------
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_simulation service...')

        # ----------------------------
        # Gym Spaces
        # ----------------------------
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]),
                                       high=np.array([0.3, 1.0]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=0.0,
                                            high=10.0,
                                            shape=(360,),
                                            dtype=np.float32)
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE  # <-- important
        )

        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', qos_profile)

        self.get_logger().info(" Gazebo RL Environment initialized.")

    def publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_position[0]
        marker.pose.position.y = self.goal_position[1]
        marker.pose.position.z = 0.1
        marker.scale.x = 1.3
        marker.scale.y = 1.3
        marker.scale.z = 1.3
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_marker_pub.publish(marker)
    # ----------------------------
    # ROS2 Callbacks
    # ----------------------------
    def laser_callback(self, msg):
        data = np.array(msg.ranges, dtype=np.float32)
        data = np.nan_to_num(data, nan=10.0, posinf=10.0, neginf=0.0)
        self.laser_data = np.clip(data, 0.0, 10.0)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_position = np.array([pos.x, pos.y])
        self.odom = msg

    # ----------------------------
    # Utility Methods
    # ----------------------------
    def _get_distance_to_goal(self):
        return np.linalg.norm(self.current_position - self.goal_position)

    def _send_action(self, linear, angular):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

    def _quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_obs(self):
        """Get current observation as a 1D array (laser + position + orientation)."""
        scan = np.array(self.laser_data, dtype=np.float32)
        return scan

    # ----------------------------
    # Gymnasium API
    # ----------------------------
    def step(self, action):
        linear, angular = action
        self._send_action(linear, angular)

        rclpy.spin_once(self, timeout_sec=0.1)

        obs = self._get_obs()
        reward = 0.0
        done = False

        #  Collision detection
        if np.min(obs) < self.collision_threshold:
            reward -= 10.0
            done = True
            self.get_logger().info("[INFO] Collision detected — resetting environment.")

        #  Progress reward
        distance_to_goal = self._get_distance_to_goal()
        progress = np.linalg.norm(self.prev_position - self.current_position)
        reward += 2.0 * progress

        #  Small penalty if still
        if progress < 0.005:
            reward -= 0.05
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        #  Stuck penalty
        #if self.stuck_steps > 20:
        #    reward -= 5.0
        #    done = True
        #    self.get_logger().info("[INFO] Robot stuck — resetting environment.")

        #  Goal reached
        if distance_to_goal < 0.3:
            reward += 20.0
            done = True
            self.get_logger().info("[INFO] Goal reached!")

        self.prev_position = np.copy(self.current_position)
        info = {}
        self.publish_goal_marker()

        return obs, reward, done, False, info  # Gymnasium 0.29+ return format

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation in Gazebo
        req = Empty.Request()
        if self.reset_client.service_is_ready():
            self.reset_client.call_async(req)
            time.sleep(1.0)

        # Reset state
        self.laser_data = np.ones(360, dtype=np.float32) * 10.0
        self.current_position = np.array([0.0, 0.0])
        self.prev_position = np.array([0.0, 0.0])
        self.stuck_steps = 0

        obs = self._get_obs()
        info = {}

        self.publish_goal_marker()
        return obs, info

    def close(self):
        self.get_logger().info("[INFO] Shutting down GazeboRLEnv.")
        self._send_action(0.0, 0.0)
        self.destroy_node()
        rclpy.shutdown()

