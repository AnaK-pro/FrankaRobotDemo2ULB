# scripts/launch_sim.py
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Simulation App
try:
    from isaacsim.simulation_app import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from omni.isaac.core import World
from src.scene.scene_builder import SceneBuilder
from src.robot.controller import PandaController
from src.robot.kinematics import PandaKinematics
from src.tasks.pick_place import ShapeSortingTask


def launch():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    
    scene = SceneBuilder()
    scene.build(world=world)

    controller = PandaController()
    robot = scene.get_robot()
    if robot:
        controller.set_robot(robot)

    task = ShapeSortingTask()
    kin = PandaKinematics()

    world.reset()
    
    # Position de départ optimale
    start_angles = np.array([0.0, -0.785, 0.0, -2.0, 0.0, 2.0, 0.785])
    controller.set_arm_positions(start_angles)
    controller.close_gripper(0.04)

    print("\n" + "="*60)
    print("🚀 TRI AUTOMATIQUE DU PANDA - Backend Stabilisé")
    print("="*60)
    
    _run_shape_sorting(simulation_app, world, scene, controller, task, kin)


def _run_shape_sorting(sim_app, world, scene, controller, task, kin):
    step = 0
    current_shape_idx = 0
    phase = "approach"

    while sim_app.is_running():
        step += 1
        world.step(render=True)

        shapes_info = scene.get_all_shapes_info()
        if not shapes_info:
            continue

        # Forme actuelle
        shape_names = list(shapes_info.keys())
        shape_name = shape_names[current_shape_idx % 3]
        info = shapes_info[shape_name]

        shape_pos = np.array(info["position"])
        bin_pos = np.array(info["bin_position"])

        # Variables initialisées
        target = np.array([0.4, 0.0, 0.3])
        gripper = 0.04

        # === LOGIQUE DE PHASES ===
        if phase == "approach":
            target = shape_pos + [0.0, 0.0, 0.28]
            gripper = 0.04
        elif phase == "grasp":
            target = shape_pos + [0.0, 0.0, 0.055]
            gripper = 0.0
        elif phase == "lift":
            target = shape_pos + [0.0, 0.0, 0.40]
            gripper = 0.0
        elif phase == "transport":
            target = bin_pos + [0.0, 0.0, 0.40]
            gripper = 0.0
        elif phase == "place":
            target = bin_pos + [0.0, 0.0, 0.08]
            gripper = 0.04

        # IK + Orientation
        base_angle = np.arctan2(target[1], target[0])
        ik = kin.inverse_kinematics(target, initial_angles=controller.get_joint_positions())
        angles = ik["angles"].copy()
        angles[0] = base_angle

        controller.set_arm_positions(angles)
        controller.close_gripper(gripper)

        # Transitions
        if phase == "approach" and step % 150 == 0:
            phase = "grasp"
        elif phase == "grasp" and step % 100 == 0:
            phase = "lift"
        elif phase == "lift" and step % 120 == 0:
            phase = "transport"
        elif phase == "transport" and step % 150 == 0:
            phase = "place"
        elif phase == "place" and step % 110 == 0:
            print(f"✅ {info.get('target_bin')} → DÉPOSÉ")
            current_shape_idx += 1
            phase = "approach"

        if step % 80 == 0:
            print(f"Pas {step:4d} | Phase: {phase:10s} | Forme: {shape_name} | Err={ik['error']:.4f}")

        # Reset sécurité
        if step > 8000:
            print("\n🔄 RESET GLOBAL")
            world.reset()
            controller.set_arm_positions([0.0, -0.785, 0.0, -2.0, 0.0, 2.0, 0.785])
            current_shape_idx = 0
            phase = "approach"
            step = 0


if __name__ == "__main__":
    launch()