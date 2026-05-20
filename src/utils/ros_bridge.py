# src/utils/ros_bridge.py
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Image
    from geometry_msgs.msg import Twist
    ROS_AVAILABLE = True
    print("[RosBridge] ROS 2 détecté ✓")
except ImportError:
    ROS_AVAILABLE = False
    print("[RosBridge] ROS 2 non disponible — mode simulation seule")


class RosBridge:

    def __init__(self, node_name="isaac_sim_bridge"):
        self.node_name = node_name
        self.enabled = ROS_AVAILABLE
        self.node = None
        self._publishers = {}
        self._subscribers = {}
        self.last_cmd_vel = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}

        if self.enabled:
            self._init_ros()

    def _init_ros(self):
        try:
            rclpy.init()
            self.node = Node(self.node_name)
            self._publishers["joint_states"] = self.node.create_publisher(
                JointState, "/joint_states", 10
            )
            self._publishers["rgb_image"] = self.node.create_publisher(
                Image, "/camera/rgb/image_raw", 10
            )
            self._publishers["depth_image"] = self.node.create_publisher(
                Image, "/camera/depth/image_raw", 10
            )
            if ROS_AVAILABLE:
                self._subscribers["cmd_vel"] = self.node.create_subscription(
                    Twist, "/cmd_vel", self._cmd_vel_callback, 10
                )
            print(f"[RosBridge] Noeud '{self.node_name}' initialisé ✓")
        except Exception as e:
            print(f"[RosBridge] Erreur : {e}")
            self.enabled = False

    def _now_msg(self):
        """Return a valid ROS Time message. Fallback to zero-time if node/clock unavailable."""
        try:
            # prefer node clock when available
            if self.node is not None and hasattr(self.node, "get_clock"):
                return self.node.get_clock().now().to_msg()
        except Exception:
            pass
        # fallback to a zeroed builtin Time
        try:
            from builtin_interfaces.msg import Time as _Time

            return _Time()
        except Exception:
            return None

    def _cmd_vel_callback(self, msg):
        self.last_cmd_vel = {
            "linear_x":  msg.linear.x,
            "linear_y":  msg.linear.y,
            "angular_z": msg.angular.z
        }

    def publish_joint_states(self, joint_names, positions, velocities=None, efforts=None):
        if not self.enabled:
            return
        msg = JointState()
        stamp = self._now_msg()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.name = joint_names
        msg.position = list(positions)
        if velocities is not None:
            msg.velocity = list(velocities)
        if efforts is not None:
            msg.effort = list(efforts)
        self._publishers["joint_states"].publish(msg)

    def publish_rgb_image(self, image_array):
        if not self.enabled:
            return
        msg = Image()
        stamp = self._now_msg()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.height   = image_array.shape[0]
        msg.width    = image_array.shape[1]
        msg.encoding = "rgb8"
        msg.step     = image_array.shape[1] * 3
        msg.data     = image_array.tobytes()
        self._publishers["rgb_image"].publish(msg)

    def publish_depth_image(self, depth_array):
        if not self.enabled:
            return
        msg = Image()
        stamp = self._now_msg()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.height   = depth_array.shape[0]
        msg.width    = depth_array.shape[1]
        msg.encoding = "32FC1"
        msg.step     = depth_array.shape[1] * 4
        msg.data     = depth_array.astype(np.float32).tobytes()
        self._publishers["depth_image"].publish(msg)

    def spin_once(self):
        if self.enabled and self.node:
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        if self.enabled and self.node:
            self.node.destroy_node()
            rclpy.shutdown()
            print("[RosBridge] Noeud ROS arrêté ✓")


if __name__ == "__main__":
    print("=== Test ros_bridge ===")
    bridge = RosBridge(node_name="test_node")
    print(f"ROS disponible : {bridge.enabled}")

    joint_names = ["panda_joint1", "panda_joint2"]
    positions   = [0.0, -0.785]
    bridge.publish_joint_states(joint_names, positions)
    print("publish_joint_states() → OK")

    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
    bridge.publish_rgb_image(fake_image)
    print("publish_rgb_image() → OK")

    bridge.shutdown()
    print("=== Test terminé ✓ ===")
