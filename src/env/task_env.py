# src/env/task_env.py
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict

from src.env.base_env import BaseEnv
from src.tasks.pick_place import ShapeSortingTask
from src.utils.math_utils import distance_3d


class ShapeSortingEnv(BaseEnv):
    """
    Environnement RL pour le tri de formes avec Franka Panda.
    Reward shaping en 4 zones selon la distance TCP → forme cible.
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
        self.action_space: spaces.Box = spaces.Box(
            low =np.array([-0.1]*7 + [0.0], dtype=np.float32),
            high=np.array([ 0.1]*7 + [1.0], dtype=np.float32),
        )

        # État pour le reward différentiel (réinitialisé à chaque reset)
        self.prev_dist: Optional[float] = None
        self.steps_without_progress: int = 0

        print("[ShapeSortingEnv] ✅ Environnement RL initialisé avec succès")

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed, options=options)
        self.task.reset()
        self.prev_dist = None
        self.steps_without_progress = 0
        obs = self._get_observation()
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)
        self._sim_step()

        tcp_pos = None
        if self.controller:
            joints  = self.controller.get_joint_positions()
            tcp_pos = self._kin.forward_kinematics(joints[:7])["position"]

        if self.scene is not None:
            _, task_reward, task_done, task_info = self.task.step(
                action, self.scene, tcp_pos=tcp_pos
            )
        else:
            task_reward, task_done, task_info = 0.0, False, {}

        # dist fournie par pick_place (None si tcp_pos indisponible)
        dist = task_info.get("dist_tcp_shape")

        obs = self._get_observation()
        reward, reward_info = self._compute_reward(action, task_reward, dist=dist)

        terminated = task_done or self._check_terminated()
        truncated  = self.step_count >= self.max_steps

        info = self._get_info()
        info.update(task_info)
        info.update(reward_info)   # proximity_reward, delta_reward, …

        if terminated or truncated:
            info["episode_summary"] = self.task.get_episode_summary()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation

    def _get_observation(self) -> np.ndarray:
        shapes_info = self.scene.get_all_shapes_info() if self.scene else None
        task_obs    = self.task._get_observation(shapes_info)  # 27 valeurs

        tcp_pos     = np.zeros(3, dtype=np.float32)
        joints_norm = np.zeros(9, dtype=np.float32)

        if self.controller:
            joints = self.controller.get_joint_positions()   # 9 DOF
            fk     = self._kin.forward_kinematics(joints[:7])
            tcp_pos = fk["position"].astype(np.float32)
            joints_norm = np.concatenate([
                joints[:7] / np.pi,   # ≈ [-1, 1]
                joints[7:9] / 0.04,   # [0, 1]
            ]).astype(np.float32)

        obs = np.concatenate([task_obs, tcp_pos, joints_norm])  # 39 valeurs
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Reward shaping en zones

    def _compute_reward(
        self,
        action: np.ndarray,
        task_reward: float,
        dist: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """
        Reward shaping progressif selon la distance TCP → forme cible.

        Zones :
          ZONE 1 – loin         dist > 0.5 m   → signal faible, survie -0.001
          ZONE 2 – proche       0.2 – 0.5 m    → bonus intermédiaire, survie -0.0005
          ZONE 3 – très proche  < 0.2 m        → fort bonus, survie -0.0001
          ZONE 4 – succès       via task_reward → grasp +1, sort +10, all +30

        Pourquoi ces seuils ?
          0.5 m : distance typique de départ sur la table 80×50 cm
          0.2 m : "zone d'engagement" où la préhension devient réaliste
          0.1 m : quasi-contact, signal critique pour déclencher la fermeture
          0.15 m : seuil de saisie dans pick_place.py (_evaluate_state)

        task_reward inclut déjà :
          - step_penalty (-0.01/step)   → signal de base dans tous les cas
          - delta × 10 (REACHING/CARRYING)
          - grasp_bonus, correct_bin, all_sorted_bonus
        Ici on ajoute un shaping ADDITIONNEL sans remplacer.
        """
        proximity_reward = 0.0
        delta_reward     = 0.0
        survival_penalty = 0.0
        # contact_reward = task_reward (bonuses grasp/sort inclus dans task_reward)

        valid_dist = dist is not None and float(dist) < 900.0

        if valid_dist:
            d = float(dist)

            # ── Survie adaptative : moins punitive quand l'agent est proche ──
            if d > 0.5:
                survival_penalty = -0.001     # standard
            elif d > 0.2:
                survival_penalty = -0.0005    # réduit : l'effort est récompensé
            else:
                survival_penalty = -0.0001    # quasi-nulle : on est dans la zone clé

            # ── Bonus de proximité par zone ───────────────────────────────────
            # Zone 2 : petit bonus pour maintenir la pression
            if d <= 0.5:
                proximity_reward = 0.002

            # Zone 3 : signal fort pour déclencher la fermeture de pince
            if d < 0.2:
                proximity_reward = 0.01       # ×5 vs zone 2
            if d < 0.1:
                proximity_reward += 0.05      # bonus critique : on est à 10 cm

            # ── Delta distance : amplifier la progression selon la zone ──────
            # task_reward contient déjà delta×10 ; on ajoute un multiplicateur
            # supplémentaire qui grandit quand l'agent est proche pour
            # "tirer" la politique vers la forme pendant la dernière approche.
            if self.prev_dist is not None:
                delta = self.prev_dist - d    # > 0 si rapprochement

                # Multiplicateur croissant avec la proximité
                if d > 0.5:
                    multiplier = 1.0    # faible : delta×10 dans task_reward suffit
                elif d > 0.2:
                    multiplier = 3.0    # zone 2 : ×3 en plus
                else:
                    multiplier = 8.0    # zone 3 : signal très fort pour la dernière ligne

                delta_reward = delta * multiplier

                # Compteur d'immobilité (delta < 0.5 mm → pas de progrès)
                if abs(delta) < 0.0005:
                    self.steps_without_progress += 1
                else:
                    self.steps_without_progress = 0

            self.prev_dist = d

        else:
            # TCP indisponible (début d'épisode ou FK échoué)
            survival_penalty = -0.001

        # ── Pénalité d'immobilité croissante ─────────────────────────────────
        # Déclenche après 30 steps sans progress (~0.5 s à 60 Hz).
        # Montée en 2 paliers pour éviter un collapse catastrophique.
        if self.steps_without_progress > 30:
            palier = min(self.steps_without_progress // 30, 2)
            survival_penalty -= 0.0005 * palier

        # ── Pénalité si action quasi-nulle (robot volontairement immobile) ───
        # Évite l'optima local "ne rien faire = limiter les pénalités".
        if np.linalg.norm(action[:7]) < 0.005:
            survival_penalty -= 0.001

        reward = float(task_reward + survival_penalty + proximity_reward + delta_reward)

        reward_info: dict = {
            "proximity_reward": round(float(proximity_reward), 5),
            "delta_reward":     round(float(delta_reward),     5),
            "survival_penalty": round(float(survival_penalty), 5),
            "contact_reward":   round(float(task_reward),      5),
        }

        return reward, reward_info

    # ──────────────────────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        active = list(self.task.sorted_shapes.keys())[:self.task.max_shapes]
        return all(self.task.sorted_shapes[s] for s in active)

    def _get_info(self) -> Dict:
        return {
            "step":       self.step_count,
            "sorted":     sum(self.task.sorted_shapes.values()),
            "is_holding": self.task.is_holding,
        }


if __name__ == "__main__":
    env = ShapeSortingEnv(render_mode=None)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("ShapeSortingEnv prêt pour l'entraînement RL ✓")
