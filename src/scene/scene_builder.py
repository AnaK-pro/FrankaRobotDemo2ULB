import numpy as np
import yaml
import os
from typing import Any

try:
    import isaacsim
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    # ON REVIENT SUR LA CLASSE FRANKA OFFICIELLE ET STABLE
    from omni.isaac.franka import Franka
    ISAAC_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid, GroundPlane
        from omni.isaac.franka import Franka
        ISAAC_AVAILABLE = True
    except ImportError:
        class World:
            def __init__(self, **kwargs: Any) -> None: ...
            scene: Any = None
            def reset(self) -> None: ...
        class DynamicCuboid:
            def __init__(self, **kwargs: Any) -> None: ...
        class GroundPlane:
            def __init__(self, **kwargs: Any) -> None: ...
        class Franka:
            def __init__(self, **kwargs: Any) -> None: ...
        ISAAC_AVAILABLE = False
        print("[SceneBuilder] Isaac Sim non disponible — mode test seul")


class SceneBuilder:
    def __init__(
        self,
        robot_config_path="config/robot.yaml",
        task_config_path="config/task.yaml",
        sim_config_path="config/simulation.yaml"
    ):
        self.robot_config = self._load_yaml(robot_config_path)
        self.task_config  = self._load_yaml(task_config_path)
        self.sim_config   = self._load_yaml(sim_config_path)

        self.world:    Any = None
        self.robot:    Any = None
        self.shapes:   dict[str, Any] = {}
        self.bins:     dict[str, Any] = {}
        self.is_built: bool = False
        print("[SceneBuilder] Initialisé ✓")

    def _load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config introuvable : {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def build(self, world=None):
        if not ISAAC_AVAILABLE:
            print("[SceneBuilder] Mode test — pas de scène 3D")
            self.is_built = True
            return None

        if world is None:
            sim_cfg = self.sim_config["simulation"]
            dt = sim_cfg.get("dt", sim_cfg.get("time_step", 1.0 / 60.0))
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=dt,
                rendering_dt=dt * 2
            )
        else:
            self.world = world

        print("[SceneBuilder] Construction de la scène...")
        self._add_ground()
        self._add_table()
        self._add_robot()
        self._add_shapes()
        self._add_bins()

        self.is_built = True
        print("[SceneBuilder] Scène construite ✓")
        return self.world

    def _add_ground(self):
        if not ISAAC_AVAILABLE:
            return
        self.world.scene.add_default_ground_plane()
        print("[SceneBuilder]   Sol ✓")

    def _add_table(self):
        if not ISAAC_AVAILABLE:
            return
        cfg = self.task_config["task"]["table"]
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array(cfg["position"]),
                scale=np.array(cfg["size"]),
                color=np.array([0.6, 0.4, 0.2]),
                mass=0.0
            )
        )
        print("[SceneBuilder]   Table ✓")

    def _add_robot(self):
        if not ISAAC_AVAILABLE:
            return
        robot_cfg = self.robot_config["robot"]
        prim_path = "/World/panda"

        # Utilisation directe de la classe Franka d'Isaac Sim.
        # Elle gère elle-même son Stage Reference sans surcouche risquée.
        self.robot = self.world.scene.add(
            Franka(
                prim_path=prim_path,
                name=robot_cfg["name"],
                position=np.array([0.0, 0.0, 0.0])
            )
        )
        print("[SceneBuilder]   Robot Panda (Franka Asset chargé) ✓")
        
    def _add_shapes(self):
        if not ISAAC_AVAILABLE:
            return
        shapes_cfg = self.task_config["task"]["shapes"]
        for shape_cfg in shapes_cfg:
            name     = shape_cfg["name"]
            prim     = f"/World/shape_{name}"
            position = np.array(shape_cfg["spawn_position"])
            size     = np.array(shape_cfg["size"])
            color    = np.array(shape_cfg["color"])
            mass     = shape_cfg["mass"]

            obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=prim,
                    name=f"shape_{name}",
                    position=position,
                    scale=size,
                    color=color,
                    mass=mass
                )
            )
            self.shapes[name] = obj
            print(f"[SceneBuilder]   Forme '{name}' à {position} ✓")

    def _add_bins(self):
        if not ISAAC_AVAILABLE:
            return
        bins_cfg = self.task_config["task"]["bins"]
        for bin_cfg in bins_cfg:
            name     = bin_cfg["name"]
            prim     = f"/World/{name}"
            position = np.array(bin_cfg["position"])
            size     = np.array(bin_cfg["size"])
            color    = np.array(bin_cfg["color"])

            obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=prim,
                    name=name,
                    position=position,
                    scale=np.array([size[0], size[1], 0.01]),
                    color=color,
                    mass=0.0
                )
            )
            self.bins[name] = obj
            print(f"[SceneBuilder]   Bac '{name}' à {position} ✓")

    def reset(self):
        print("[SceneBuilder] Reset...")
        if ISAAC_AVAILABLE and self.world is not None:
            self.world.reset()
        self._reset_shapes()
        print("[SceneBuilder] Reset ✓")

    def _reset_shapes(self):
        shapes_cfg = self.task_config["task"]["shapes"]
        for shape_cfg in shapes_cfg:
            name = shape_cfg["name"]
            pos  = np.array(shape_cfg["spawn_position"])

            dr = self.task_config["task"]["domain_randomization"]
            if dr["enabled"]:
                pos[0] += np.random.uniform(-0.03, 0.03)
                pos[1] += np.random.uniform(-0.03, 0.03)

            if ISAAC_AVAILABLE and name in self.shapes:
                self.shapes[name].set_world_pose(
                    position=pos,
                    orientation=np.array([0.0, 0.0, 0.0, 1.0])
                )
            print(f"[SceneBuilder]   {name} → {pos.round(3)}")

    def get_shape_position(self, shape_name):
        if ISAAC_AVAILABLE and shape_name in self.shapes:
            pos, _ = self.shapes[shape_name].get_world_pose()
            return np.array(pos)
        for s in self.task_config["task"]["shapes"]:
            if s["name"] == shape_name:
                return np.array(s["spawn_position"])
        return np.zeros(3)

    def get_bin_position(self, bin_name):
        for b in self.task_config["task"]["bins"]:
            if b["name"] == bin_name:
                return np.array(b["position"])
        return np.zeros(3)

    def get_correct_bin_for_shape(self, shape_name):
        for b in self.task_config["task"]["bins"]:
            if b["shape"] == shape_name:
                return b["name"]
        return None

    def get_all_shapes_info(self):
        info = {}
        for shape_cfg in self.task_config["task"]["shapes"]:
            name = shape_cfg["name"]
            pos  = self.get_shape_position(name)
            bin_name = self.get_correct_bin_for_shape(name)
            bin_pos  = self.get_bin_position(bin_name)
            dist     = np.linalg.norm(pos - bin_pos)
            tol      = self.task_config["task"]["placement_tolerance"]

            info[name] = {
                "position":    pos.round(3).tolist(),
                "target_bin":  bin_name,
                "bin_position": bin_pos.round(3).tolist(),
                "distance_to_bin": round(float(dist), 3),
                "is_sorted":   dist < tol
            }
        return info

    def get_robot(self) -> Any:
        return self.robot

    def get_world(self) -> Any:
        return self.world

    def get_scene_info(self):
        shapes_info = self.get_all_shapes_info()
        nb_sorted   = sum(1 for s in shapes_info.values() if s["is_sorted"])
        return {
            "is_built":        self.is_built,
            "isaac_available": ISAAC_AVAILABLE,
            "robot_loaded":    self.robot is not None,
            "shapes_sorted":   f"{nb_sorted}/3",
            "shapes":          shapes_info
        }


if __name__ == "__main__":
    builder = SceneBuilder()
    builder.build()