# src/env/task_env.py
# src/env/task_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

from src.env.base_env import BaseEnv
from src.tasks.pick_place import ShapeSortingTask
from src.utils.math_utils import distance_3d


class ShapeSortingEnv(BaseEnv):
    """
    Environnement RL pour le tri de formes avec Franka Panda.
    """

    def __init__(self, render_mode: Optional[str] = None, use_ros: bool = False):
        super().__init__(render_mode=render_mode, use_ros=use_ros)
        
        self.task = ShapeSortingTask()

        from src.robot.kinematics import PandaKinematics
        self._kin = PandaKinematics()

        # 27 (task obs) + 3 (TCP position) + 9 (joints normalisés) = 39
        self.observation_space = spaces.Box(
            low=-2.0 * np.ones(39, dtype=np.float32),
            high= 2.0 * np.ones(39, dtype=np.float32),
            dtype=np.float32
        )

        # Contrôle articulaire : [dj0..dj6 (±0.1 rad/step), gripper 0=fermé 1=ouvert]
        self.action_space = spaces.Box(
            low =np.array([-0.1]*7 + [0.0], dtype=np.float32),
            high=np.array([ 0.1]*7 + [1.0], dtype=np.float32),
        )

        print("[ShapeSortingEnv] ✅ Environnement RL initialisé avec succès")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed, options=options)
        self.task.reset()
        obs = self._get_observation()
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)
        self._sim_step()

        tcp_pos = None
        if self.controller:
            joints = self.controller.get_joint_positions()
            tcp_pos = self._kin.forward_kinematics(joints[:7])["position"]

        if self.scene is not None:
            _, task_reward, task_done, task_info = self.task.step(
                action, self.scene, tcp_pos=tcp_pos
            )
        else:
            task_reward, task_done, task_info = 0.0, False, {}

        obs = self._get_observation()
        reward = self._compute_reward(action, task_reward)

        terminated = task_done or self._check_terminated()
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info.update(task_info)

        # Ajout du résumé d'épisode pour le callback TensorBoard
        if terminated or truncated:
            info["episode_summary"] = self.task.get_episode_summary()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        shapes_info = self.scene.get_all_shapes_info() if self.scene else None
        task_obs = self.task._get_observation(shapes_info)  # 27 valeurs

        tcp_pos      = np.zeros(3, dtype=np.float32)
        joints_norm  = np.zeros(9, dtype=np.float32)

        if self.controller:
            joints = self.controller.get_joint_positions()  # 9 DOF
            fk = self._kin.forward_kinematics(joints[:7])
            tcp_pos = fk["position"].astype(np.float32)
            joints_norm = np.concatenate([
                joints[:7] / np.pi,       # articulaire → [-1, 1] approximatif
                joints[7:9] / 0.04,       # pince → [0, 1]
            ]).astype(np.float32)

        obs = np.concatenate([task_obs, tcp_pos, joints_norm])  # 39 valeurs
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    def _compute_reward(self, action, task_reward: float) -> float:
        return float(task_reward - 0.005)

    def _check_terminated(self) -> bool:
        return all(self.task.sorted_shapes.values())

    def _get_info(self) -> Dict:
        return {
            "step": self.step_count,
            "sorted": sum(self.task.sorted_shapes.values()),
            "is_holding": self.task.is_holding
        }


if __name__ == "__main__":
    env = ShapeSortingEnv(render_mode=None)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("ShapeSortingEnv prêt pour l'entraînement RL ✓")