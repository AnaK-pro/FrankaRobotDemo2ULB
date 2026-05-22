# src/env/base_env.py
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any


class BaseEnv(gym.Env):
    """
    Classe de base pour tous les environnements Isaac Sim avec Franka Panda.
    """

    def __init__(self, render_mode: Optional[str] = None, use_ros: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.use_ros = use_ros
        
        self.step_count = 0
        self.episode_count = 0
        self.max_steps = 1000
        
        self.scene = None
        self.controller = None
        self.world = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement (compatible Gymnasium)."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1

        # Import et création du monde seulement si nécessaire
        if self.world is None:
            from omni.isaac.core import World
            from src.scene.scene_builder import SceneBuilder
            from src.robot.controller import PandaController

            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0/60.0,
                rendering_dt=1.0/30.0
            )
            self.scene = SceneBuilder()
            self.scene.build(world=self.world)
            
            self.controller = PandaController()
            robot = self.scene.get_robot()
            if robot:
                self.controller.set_robot(robot)

        self.world.reset()
        if self.controller:
            self.controller.go_to_default_position()

        obs = np.zeros(39, dtype=np.float32)
        info: Dict[str, Any] = {"episode": self.episode_count}

        return obs, info

    def _sim_step(self):
        """Avance la simulation d'un pas."""
        if self.world:
            self.world.step(render=self.render_mode == "human")

    def _apply_action(self, action: np.ndarray):
        """Contrôle direct en espace articulaire (action = [dj0..dj6, gripper 0-1])."""
        if not self.controller:
            return
        joints = self.controller.get_joint_positions()
        new_joints = joints[:7] + np.array(action[:7], dtype=np.float32)
        self.controller.set_arm_positions(new_joints)
        # action[7] dans [0,1] : 0 = pince fermée, 1 = ouverte → [0, 0.04 m]
        self.controller.close_gripper((1.0 - float(action[7])) * 0.04)

    def close(self):
        if self.world:
            self.world.stop()
            print("[BaseEnv] Simulation fermée.")


if __name__ == "__main__":
    print("BaseEnv chargé avec succès.")