# src/env/task_env.py
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict

from src.env.base_env import BaseEnv
from src.tasks.pick_place import ShapeSortingTask

# Workspace Franka en repère robot-local (mètres)
_WS_TCP_MAX_NORM = 1.2   # reach max ~0.85m + marge
_DIST_MAX_CLAMP  = 1.5   # distance max admissible TCP→forme


class ShapeSortingEnv(BaseEnv):
    """
    Environnement RL pour le tri de formes avec Franka Panda.
    Reward shaping en 4 zones selon la distance TCP → forme cible.
    """

    action_space: spaces.Box  # narrowed from Space — Pyright sees mutable invariance

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
        self.action_space = spaces.Box(  # type: ignore[assignment]
            low =np.array([-0.1]*7 + [0.0], dtype=np.float32),
            high=np.array([ 0.1]*7 + [1.0], dtype=np.float32),
        )

        # État pour le reward shaping (réinitialisé à chaque reset)
        self.prev_dist: Optional[float] = None
        self.steps_without_progress: int = 0

        self._sanity_check()
        print("[ShapeSortingEnv] ✅ Environnement RL initialisé avec succès")

    # ──────────────────────────────────────────────────────────────────────────
    # Sanity check au démarrage

    def _sanity_check(self) -> None:
        """Vérifie le workspace et la FK au lancement."""
        default_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        fk = self._kin.forward_kinematics(default_joints)
        pos = fk["position"]

        if np.any(~np.isfinite(pos)):
            print(f"[SanityCheck] ❌ FK NaN/Inf à la position par défaut: {pos}")
        elif float(np.linalg.norm(pos)) > _WS_TCP_MAX_NORM:
            print(f"[SanityCheck] ⚠ FK hors workspace: {pos.round(3)} (norm={np.linalg.norm(pos):.3f})")
        else:
            print(f"[SanityCheck] ✓ FK défaut OK: TCP = {pos.round(3)}")

        for s in self.task.shapes_config:
            sp = np.array(s["spawn_position"])
            if np.linalg.norm(sp) > 1.5 or sp[2] < 0:
                print(f"[SanityCheck] ⚠ Shape '{s['name']}' spawn hors workspace: {sp}")

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

        # Valider et récupérer la distance proprement
        raw_dist = task_info.get("dist_tcp_shape")
        dist = self._validate_dist(raw_dist)

        obs = self._get_observation()
        is_holding = bool(task_info.get("is_holding", False))
        reward, reward_info = self._compute_reward(
            action, task_reward,
            dist=dist,
            contact_bonus=float(task_info.get("contact_bonus", 0.0)),
            is_holding=is_holding,
        )

        terminated = task_done or self._check_terminated()
        truncated  = self.step_count >= self.max_steps

        info = self._get_info()
        info.update(task_info)
        info.update(reward_info)

        if terminated or truncated:
            info["episode_summary"] = self.task.get_episode_summary()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Validation distance

    def _validate_and_compute_distance(
        self,
        tcp_pos: np.ndarray,
        shape_pos: np.ndarray,
    ) -> Optional[float]:
        """
        Calcule la distance TCP→forme avec validation complète.

        Retourne None si les positions sont suspectes (NaN, hors workspace).
        Retourne la distance clampée à _DIST_MAX_CLAMP sinon.

        Pourquoi norm(tcp_pos) > 1.2 → invalide ?
        Le reach max du Franka Panda est ~855mm. Une norme > 1.2m en repère
        robot-local signifie que le vecteur offset a été mal soustrait
        (bug typique: end_effector.get_world_pose() retourne (0,0,0) quand
        la physique n'est pas encore initialisée → pos - offset = -offset,
        norm = 4.2m pour un env en grille à 3m).
        """
        if tcp_pos is None or np.any(~np.isfinite(tcp_pos)):
            return None
        if shape_pos is None or np.any(~np.isfinite(shape_pos)):
            return None

        if float(np.linalg.norm(tcp_pos)) > _WS_TCP_MAX_NORM:
            return None  # position TCP invalide (mauvais référentiel)

        dist = float(np.linalg.norm(
            np.asarray(tcp_pos, dtype=np.float32) - np.asarray(shape_pos, dtype=np.float32)
        ))
        return min(dist, _DIST_MAX_CLAMP)

    def _validate_dist(self, raw_dist: Optional[float]) -> Optional[float]:
        """Filtre les distances aberrantes fournies par pick_place."""
        if raw_dist is None:
            return None
        d = float(raw_dist)
        if d >= 900.0 or not np.isfinite(d):
            return None
        if d > _DIST_MAX_CLAMP:
            return _DIST_MAX_CLAMP  # clamp sans invalider
        return d

    # ──────────────────────────────────────────────────────────────────────────
    # Reward shaping

    def _compute_reward(
        self,
        action: np.ndarray,
        task_reward: float,
        dist: Optional[float] = None,
        contact_bonus: float = 0.0,
        is_holding: bool = False,
    ) -> Tuple[float, dict]:
        """
        Reward shaping état-dépendant : REACHING vs CARRYING.

        REACHING (is_holding=False) :
          - zones survival/proximity adaptatives
          - approach_bonus progressif (distance TCP→forme)
          - grip_incentive : bonus proportionnel à la fermeture dans la zone (<0.20m)
            → gradient explicite vers action[7]→0 ; avec gripper_closed<0.5
              P(trigger grasp) passe de 16% à >50% après quelques updates

        CARRYING (is_holding=True) :
          - hold_bonus   = +0.02/step si pince fermée (gripper < 0.5)
          - hold_penalty = −0.05/step si pince ouverte (gripper ≥ 0.5)
          → empêche les lâchers accidentels sur le trajet vers le bac ;
            sans ça l'agent exploite approach_bonus en survol plutôt que de saisir
        """
        survival      = 0.0
        proximity     = 0.0
        approach_bonus = 0.0

        if not is_holding:
            # ── REACHING ──────────────────────────────────────────────────
            if dist is not None:
                if dist > 0.5:
                    survival = -0.001
                elif dist > 0.2:
                    survival  = -0.0005
                    proximity = 0.002
                else:
                    survival  = -0.0001
                    proximity = 0.01
                    if dist < 0.1:
                        proximity += 0.05

                if dist < 0.05:    approach_bonus = 0.10
                elif dist < 0.10:  approach_bonus = 0.05
                elif dist < 0.25:  approach_bonus = 0.01

                # Grip incentive signé : [-0.08, +0.08]
                # CRUCIAL : pas de max(0,...) — pénalise gripper>0.5 dans la zone.
                # Sans ça, pour gripper=0.9 : max(0, 1-1.8)=0 → gradient nul.
                # La politique converge à gripper≈0.9 pour éviter CARRYING et
                # le gradient est mort. Avec la version signée :
                #   gripper=0   → +0.08 (bonus fermé)
                #   gripper=0.5 → 0     (neutre)
                #   gripper=0.9 → -0.07 (pénalité ouvert → gradient vers fermeture)
                if dist < 0.25:
                    gripper_val    = float(action[7])
                    grip_incentive = 0.08 * (1.0 - gripper_val * 2.0)
                    approach_bonus += grip_incentive

                if self.prev_dist is not None:
                    if abs(self.prev_dist - dist) < 0.0005:
                        self.steps_without_progress += 1
                    else:
                        self.steps_without_progress = 0
                self.prev_dist = dist
            else:
                survival = -0.001

            if self.steps_without_progress > 30:
                palier = min(self.steps_without_progress // 30, 2)
                survival -= 0.0005 * palier

            if float(np.linalg.norm(action[:7])) < 0.005:
                survival -= 0.001

        else:
            # ── CARRYING : hold bonus ──────────────────────────────────────
            gripper_val    = float(action[7])
            approach_bonus = 0.02 if gripper_val < 0.5 else -0.05

        total = np.clip(
            float(task_reward) + survival + proximity + approach_bonus,
            -1.0, 5.0
        )

        reward_info: dict = {
            "survival_penalty": round(float(survival),       5),
            "proximity_reward": round(float(proximity),      5),
            "approach_bonus":   round(float(approach_bonus), 5),
            "contact_reward":   round(float(contact_bonus),  5),
            "task_signal":      round(float(task_reward),    5),
        }

        return total, reward_info

    def _debug_reward_components(self, dist: Optional[float], components: dict) -> None:
        """Alerte console si les composantes de reward semblent anormales."""
        if self.step_count % 200 != 0:
            return
        if dist is None:
            return
        ab = components.get("approach_bonus", 0.0)
        cr = components.get("contact_reward", 0.0)
        if dist < 0.20:
            zone = "SAISIE" if dist < 0.20 else "approche"
            print(
                f"[RewardDebug|ep={self.episode_count} step={self.step_count:4d}] "
                f"dist={dist:.3f}  approach={ab:.4f}  contact={cr:.4f}  "
                f"stagnation={self.steps_without_progress}  [{zone}]"
            )

    # ──────────────────────────────────────────────────────────────────────────

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
