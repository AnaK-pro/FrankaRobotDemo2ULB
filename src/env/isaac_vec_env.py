# src/env/isaac_vec_env.py
"""
Environnement vectorisé Isaac Sim : N robots Franka en parallèle (grille explicite).
Compatible SB3 PPO via l'interface VecEnv.

IMPORTANT : SimulationApp doit être créé AVANT tout import omni/isaacsim.
            Ce fichier ne crée pas la SimulationApp — c'est train_vec.py qui s'en charge.

Observations (39D) : 27 task + 3 TCP + 9 joints normalisés — toutes en repère robot-local.
Actions      (8D)  : [dj0..dj6 ±0.1 rad/step, gripper 0=fermé 1=ouvert]
"""

import numpy as np
import yaml
from typing import Any, Dict, List, Optional, Tuple

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class IsaacVecEnv(VecEnv):
    # Surcharge des annotations de VecEnv → Pyright connaît le type concret
    action_space: spaces.Box
    observation_space: spaces.Box

    OBS_DIM = 39   # 27 (task) + 3 (TCP local) + 9 (joints normalisés)
    ACT_DIM = 8    # [dj0..dj6 ±0.1 rad/step, gripper 0=fermé 1=ouvert]

    def __init__(
        self,
        num_envs: int = 40,
        max_steps: int = 1000,
        env_spacing: float = 3.0,
        render: bool = True,
    ):
        obs_space = spaces.Box(
            low=-2.0 * np.ones(self.OBS_DIM, dtype=np.float32),
            high= 2.0 * np.ones(self.OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        act_space = spaces.Box(
            low =np.array([-0.1]*7 + [0.0], dtype=np.float32),
            high=np.array([ 0.1]*7 + [1.0], dtype=np.float32),
        )
        super().__init__(num_envs, obs_space, act_space)

        self._max_steps = max_steps
        self._render = render
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._pending_actions: Optional[np.ndarray] = None
        # État par env pour le reward différentiel (réinitialisé au reset/reset_single)
        self._prev_dists = np.full(num_envs, np.nan, dtype=np.float32)
        self._steps_no_progress = np.zeros(num_envs, dtype=np.int32)

        with open("config/task.yaml") as f:
            self._task_cfg = yaml.safe_load(f)["task"]

        self._setup_isaac(num_envs, env_spacing)

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation Isaac Sim
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_isaac(self, num_envs: int, spacing: float) -> None:
        import math

        try:
            from isaacsim.core.api import World
            from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        except ImportError:
            from omni.isaac.core import World
            from omni.isaac.core.objects import DynamicCuboid, FixedCuboid

        from omni.isaac.franka import Franka
        from src.robot.controller import PandaController
        from src.robot.kinematics import PandaKinematics
        from src.tasks.pick_place import ShapeSortingTask

        shapes_cfg = self._task_cfg["shapes"]
        bins_cfg   = self._task_cfg["bins"]
        table_cfg  = self._task_cfg["table"]

        # ── Monde physique ────────────────────────────────────────────────────
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 30.0,
        )
        self._world.scene.add_default_ground_plane()

        # ── Grille : calcul des offsets pour chaque env ───────────────────────
        num_cols = int(math.ceil(math.sqrt(num_envs)))
        self._offsets: List[np.ndarray] = []
        for i in range(num_envs):
            col = i % num_cols
            row = i // num_cols
            self._offsets.append(np.array([col * spacing, row * spacing, 0.0], dtype=np.float32))
        offsets = self._offsets

        # ── Construction explicite de chaque env à sa position dans la grille ─
        # Chaque objet est créé une fois avec ses coordonnées monde absolues.
        # Pas de cloner : plus de magie USD, comportement garanti.
        self._robots: List[Any] = []
        self._shapes_objs: List[Dict[str, Any]] = []
        self._bins_objs:   List[Dict[str, Any]] = []

        for i in range(num_envs):
            off = offsets[i]

            # Robot
            robot = self._world.scene.add(
                Franka(
                    prim_path=f"/World/envs/env_{i}/robot",
                    name=f"robot_{i}",
                    position=off + np.array([0.0, 0.0, 0.0]),
                )
            )
            self._robots.append(robot)

            # Table (FixedCuboid = statique, ne tombe pas)
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/envs/env_{i}/table",
                    name=f"table_{i}",
                    position=off + np.array(table_cfg["position"]),
                    scale=np.array(table_cfg["size"]),
                    color=np.array([0.6, 0.4, 0.2]),
                )
            )

            # Formes (DynamicCuboid = saisies par le robot)
            env_shapes: Dict[str, Any] = {}
            for s in shapes_cfg:
                obj = self._world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/envs/env_{i}/shape_{s['name']}",
                        name=f"shape_{s['name']}_{i}",
                        position=off + np.array(s["spawn_position"]),
                        scale=np.array(s["size"]),
                        color=np.array(s["color"]),
                        mass=float(s["mass"]),
                    )
                )
                env_shapes[s["name"]] = obj
            self._shapes_objs.append(env_shapes)

            # Bacs (FixedCuboid = marqueurs de dépôt, statiques)
            env_bins: Dict[str, Any] = {}
            for b in bins_cfg:
                obj = self._world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/envs/env_{i}/bin_{b['name']}",
                        name=f"bin_{b['name']}_{i}",
                        position=off + np.array(b["position"]),
                        scale=np.array([b["size"][0], b["size"][1], 0.01]),
                        color=np.array(b["color"]),
                    )
                )
                env_bins[b["name"]] = obj
            self._bins_objs.append(env_bins)

            if (i + 1) % 10 == 0 or i == num_envs - 1:
                print(f"[IsaacVecEnv] Env {i+1}/{num_envs} construits...")

        # ── Cinématique + contrôleurs + tâches (un par env) ──────────────────
        self._kin = PandaKinematics()
        self._controllers: List[PandaController] = []
        self._tasks: List[ShapeSortingTask] = []

        for i in range(num_envs):
            ctrl = PandaController()
            ctrl.set_robot(self._robots[i])
            self._controllers.append(ctrl)
            self._tasks.append(ShapeSortingTask(max_shapes=1))  # curriculum débute à 1 forme

        self._world.reset()
        print(f"[IsaacVecEnv] ✅ {num_envs} environnements prêts — grille {num_cols}×{math.ceil(num_envs/num_cols)}, espacement {spacing}m")

    # ──────────────────────────────────────────────────────────────────────────
    # Interface VecEnv (SB3)
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._world.reset()
        self._step_counts[:] = 0
        self._prev_dists[:] = np.nan
        self._steps_no_progress[:] = 0
        for i in range(self.num_envs):
            self._tasks[i].reset()
            self._controllers[i].go_to_default_position()
        return self._get_all_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.clip(
            actions, self.action_space.low, self.action_space.high
        )

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        actions = self._pending_actions
        n = self.num_envs

        # Appliquer les actions (contrôle articulaire direct)
        for i in range(n):
            self._apply_action(i, actions[i])

        # Un seul pas physique — tous les envs avancent ensemble
        self._world.step(render=self._render)

        # Fake grasping : téléporter la forme tenue au TCP pour un signal de reward fiable.
        # _get_tcp_pos retourne une position locale → on rajoute l'offset pour set_world_pose.
        for i in range(n):
            if self._tasks[i].is_holding and self._tasks[i].held_shape:
                tcp_local = self._get_tcp_pos(i)
                obj = self._shapes_objs[i].get(self._tasks[i].held_shape)
                if obj is not None:
                    obj.set_world_pose(
                        position=tcp_local + self._offsets[i],
                        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                    )
        self._step_counts += 1

        # Collecter obs, rewards, dones
        obs     = self._get_all_obs()
        rewards = np.zeros(n, dtype=np.float32)
        dones   = np.zeros(n, dtype=bool)
        infos: List[dict] = [{} for _ in range(n)]

        for i in range(n):
            tcp_pos     = self._get_tcp_pos(i)
            shapes_info = self._compute_shapes_info(i)

            _, r, done_task, task_info = self._tasks[i].step(
                actions[i], _SceneAdapter(shapes_info), tcp_pos=tcp_pos
            )

            # ── Reward shaping en zones (distance TCP → forme cible) ──────
            dist = task_info.get("dist_tcp_shape")
            proximity_reward = 0.0
            delta_reward     = 0.0
            survival_penalty = 0.0

            if dist is not None and float(dist) < 900.0:
                d = float(dist)
                if d > 0.5:
                    survival_penalty = -0.001
                elif d > 0.2:
                    survival_penalty = -0.0005
                    proximity_reward = 0.002
                else:
                    survival_penalty = -0.0001
                    proximity_reward = 0.01
                    if d < 0.1:
                        proximity_reward += 0.05

                prev = self._prev_dists[i]
                if not np.isnan(prev):
                    delta = float(prev) - d
                    if d > 0.5:
                        multiplier = 1.0
                    elif d > 0.2:
                        multiplier = 3.0
                    else:
                        multiplier = 8.0
                    delta_reward = delta * multiplier
                    if abs(delta) < 0.0005:
                        self._steps_no_progress[i] += 1
                    else:
                        self._steps_no_progress[i] = 0
                self._prev_dists[i] = d
            else:
                survival_penalty = -0.001

            # Pénalité d'immobilité croissante (paliers de 30 steps)
            if self._steps_no_progress[i] > 30:
                palier = min(int(self._steps_no_progress[i]) // 30, 2)
                survival_penalty -= 0.0005 * palier

            # Pénalité si action quasi-nulle
            if float(np.linalg.norm(actions[i][:7])) < 0.005:
                survival_penalty -= 0.001

            rewards[i] = float(r) + survival_penalty + proximity_reward + delta_reward

            task_info.update({
                "proximity_reward": round(float(proximity_reward), 5),
                "delta_reward":     round(float(delta_reward),     5),
                "survival_penalty": round(float(survival_penalty), 5),
                "contact_reward":   round(float(r),                5),
            })

            active    = list(self._tasks[i].sorted_shapes.keys())[:self._tasks[i].max_shapes]
            terminated = done_task or all(self._tasks[i].sorted_shapes[s] for s in active)
            truncated   = int(self._step_counts[i]) >= self._max_steps
            dones[i]    = terminated or truncated
            infos[i]    = task_info

            if dones[i]:
                infos[i]["episode_summary"] = self._tasks[i].get_episode_summary()
                self._reset_single_env(i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        if hasattr(self, "_world") and self._world:
            self._world.stop()

    def seed(self, seed=None):
        return [seed] * self.num_envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._tasks[i], method_name)(*args, **kwargs) for i in indices]

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._tasks[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self._tasks[i], attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [False] * len(list(indices))

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_action(self, env_idx: int, action: np.ndarray) -> None:
        ctrl   = self._controllers[env_idx]
        joints = ctrl.get_joint_positions()
        new_arm      = joints[:7] + np.array(action[:7], dtype=np.float32)
        gripper_w    = (1.0 - float(action[7])) * 0.04  # 0=fermé, 1=ouvert
        full_joints  = np.concatenate([new_arm, [gripper_w, gripper_w]])
        ctrl.set_joint_positions(full_joints)

    def _get_tcp_pos(self, env_idx: int) -> np.ndarray:
        """Retourne la position TCP dans le repère local du robot (offset soustrait)."""
        try:
            pos, _ = self._robots[env_idx].end_effector.get_world_pose()
            return np.array(pos, dtype=np.float32) - self._offsets[env_idx]
        except Exception:
            joints = self._controllers[env_idx].get_joint_positions()
            return self._kin.forward_kinematics(joints[:7])["position"]

    def _compute_shapes_info(self, env_idx: int) -> dict:
        tol      = self._task_cfg["placement_tolerance"]
        info     = {}
        bins_cfg = self._task_cfg["bins"]

        off = self._offsets[env_idx]

        # Positions des bacs dans le repère local du robot (offset soustrait)
        bin_local_pos: dict = {}
        for b in bins_cfg:
            bin_obj = self._bins_objs[env_idx].get(b["name"])
            if bin_obj is not None:
                p, _ = bin_obj.get_world_pose()
                bin_local_pos[b["name"]] = np.array(p, dtype=np.float32) - off
            else:
                bin_local_pos[b["name"]] = np.array(b["position"], dtype=np.float32)

        for s in self._task_cfg["shapes"]:
            name    = s["name"]
            obj     = self._shapes_objs[env_idx].get(name)
            if obj is not None:
                p, _  = obj.get_world_pose()
                pos   = np.array(p, dtype=np.float32) - off
            else:
                pos = np.array(s["spawn_position"], dtype=np.float32)

            bin_name = next(
                (b["name"] for b in bins_cfg if b.get("shape") == name), None
            )
            bin_pos  = bin_local_pos.get(bin_name, np.zeros(3, dtype=np.float32))
            dist     = float(np.linalg.norm(pos - bin_pos))
            info[name] = {
                "position":        pos.tolist(),
                "target_bin":      bin_name,
                "bin_position":    bin_pos.tolist(),
                "distance_to_bin": round(dist, 3),
                "is_sorted":       dist < tol,
            }
        return info

    def _get_all_obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.OBS_DIM), dtype=np.float32)
        for i in range(self.num_envs):
            shapes_info  = self._compute_shapes_info(i)
            task_obs     = self._tasks[i]._get_observation(shapes_info)  # 27
            tcp          = self._get_tcp_pos(i)                          # 3
            joints       = self._controllers[i].get_joint_positions()    # 9
            joints_norm  = np.concatenate([
                joints[:7] / np.pi,   # ≈ [-1, 1]
                joints[7:9] / 0.04,   # [0, 1]
            ]).astype(np.float32)
            obs[i] = np.clip(np.concatenate([task_obs, tcp, joints_norm]), -2.0, 2.0)
        return obs

    def _reset_single_env(self, env_idx: int) -> None:
        """Reset d'un seul env sans toucher aux autres."""
        off = self._offsets[env_idx]
        dr  = self._task_cfg["domain_randomization"]
        for s in self._task_cfg["shapes"]:
            name = s["name"]
            pos  = off + np.array(s["spawn_position"], dtype=np.float32)
            if dr["enabled"]:
                pos[0] += np.random.uniform(-0.03, 0.03)
                pos[1] += np.random.uniform(-0.03, 0.03)
            obj = self._shapes_objs[env_idx].get(name)
            if obj is not None:
                obj.set_world_pose(
                    position=pos,
                    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                )
        self._controllers[env_idx].go_to_default_position()
        self._tasks[env_idx].reset()
        self._step_counts[env_idx] = 0
        self._prev_dists[env_idx] = np.nan
        self._steps_no_progress[env_idx] = 0


class _SceneAdapter:
    """
    Adaptateur minimal : expose get_all_shapes_info() comme SceneBuilder,
    mais à partir d'un dict déjà calculé (évite un double appel USD).
    """

    def __init__(self, shapes_info: dict):
        self._info = shapes_info

    def get_all_shapes_info(self) -> dict:
        return self._info
