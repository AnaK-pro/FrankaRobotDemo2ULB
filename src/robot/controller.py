# src/robot/controller.py
import numpy as np
import yaml
from typing import Optional

try:
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.types import ArticulationAction
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    class Robot: pass
    class ArticulationAction: pass


class PandaController:
    def __init__(self, config_path="config/robot.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["robot"]
        
        self.joint_names = self.config["joint_names"]
        self.num_dof = self.config["dof"]
        self.default_positions = np.array(self.config["default_joint_positions"], dtype=np.float32)
        
        limits = self.config["joint_limits"]
        self.joint_limits = np.array([limits[name] for name in self.joint_names], dtype=np.float32)
        
        self.robot: Optional[Robot] = None
        self.current_positions = self.default_positions.copy()
        self.target_positions = self.default_positions.copy()  # pour le lissage

        print(f"[Controller] ✅ PandaController initialisé avec lissage")

    def set_robot(self, robot):
        self.robot = robot
        if ISAAC_AVAILABLE and robot:
            print(f"[Controller] ✅ Robot '{getattr(robot, 'name', 'Franka')}' lié")

    def set_joint_positions(self, positions, smooth_factor=0.6):
        positions = np.asarray(positions, dtype=np.float32).flatten()
        positions = np.clip(positions, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # LISSAGE des commandes (très important pour éviter les tremblements)
        self.target_positions = self.target_positions * (1 - smooth_factor) + positions * smooth_factor

        if ISAAC_AVAILABLE and self.robot:
            action = ArticulationAction(joint_positions=self.target_positions)  # type: ignore
            self.robot.apply_action(action)  # type: ignore

        self.current_positions = self.target_positions.copy()
        return self.current_positions

    def set_arm_positions(self, arm_positions):
        current = self.get_joint_positions()
        full = np.concatenate([np.array(arm_positions)[:7], current[7:]])
        return self.set_joint_positions(full)

    def close_gripper(self, width: float = 0.0):
        width = float(np.clip(width, 0.0, 0.04))
        current = self.get_joint_positions()
        full = np.concatenate([current[:7], [width, width]])
        return self.set_joint_positions(full, smooth_factor=0.4)  # plus rapide pour la pince

    def get_joint_positions(self):
        if ISAAC_AVAILABLE and self.robot:
            try:
                state = self.robot.get_joints_state()  # type: ignore
                self.current_positions = np.array(state.positions, dtype=np.float32)
            except:
                pass
        return self.current_positions.copy()

    def go_to_default_position(self):
        self.target_positions = self.default_positions.copy()
        return self.set_joint_positions(self.default_positions)