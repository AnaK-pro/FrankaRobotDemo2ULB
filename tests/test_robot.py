
# tests/test_robot.py
# Tests unitaires pour le contrôleur et la cinématique

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robot.controller import PandaController
from src.robot.kinematics import PandaKinematics
from src.utils.math_utils import rad_to_deg, deg_to_rad


def test_controller_init():
    """Test : le contrôleur s'initialise correctement."""
    ctrl = PandaController()
    assert ctrl.num_dof == 9
    assert len(ctrl.joint_names) == 9
    assert len(ctrl.default_positions) == 9
    print("✓ test_controller_init")


def test_default_position():
    """Test : la position par défaut est correcte."""
    ctrl = PandaController()
    ctrl.go_to_default_position()
    pos = ctrl.get_joint_positions()
    assert len(pos) == 9
    assert abs(pos[1] - (-0.785)) < 0.01  # joint2 = -45°
    assert abs(pos[3] - (-2.356)) < 0.01  # joint4 = -135°
    print("✓ test_default_position")


def test_gripper_open():
    """Test : ouverture de la pince."""
    ctrl = PandaController()
    ctrl.open_gripper()
    pos = ctrl.get_joint_positions()
    assert abs(pos[7] - 0.04) < 0.001  # doigt1 = 40mm
    assert abs(pos[8] - 0.04) < 0.001  # doigt2 = 40mm
    print("✓ test_gripper_open")


def test_gripper_close():
    """Test : fermeture de la pince."""
    ctrl = PandaController()
    ctrl.close_gripper(width=0.0)
    pos = ctrl.get_joint_positions()
    assert abs(pos[7] - 0.0) < 0.001
    assert abs(pos[8] - 0.0) < 0.001
    print("✓ test_gripper_close")


def test_gripper_partial():
    """Test : fermeture partielle de la pince."""
    ctrl = PandaController()
    ctrl.close_gripper(width=0.02)
    pos = ctrl.get_joint_positions()
    assert abs(pos[7] - 0.02) < 0.001
    print("✓ test_gripper_partial")


def test_joint_limits():
    """Test : les limites articulaires sont respectées."""
    ctrl = PandaController()
    # Envoie des angles hors limites
    extreme = np.array([10.0] * 7 + [1.0, 1.0])
    result = ctrl.set_joint_positions(extreme)
    # Vérifie que tout est dans les limites
    for i in range(7):
        assert result[i] <= ctrl.joint_limits[i, 1] + 0.001
        assert result[i] >= ctrl.joint_limits[i, 0] - 0.001
    print("✓ test_joint_limits")


def test_arm_positions():
    """Test : commande du bras seul."""
    ctrl = PandaController()
    ctrl.open_gripper()
    arm = np.array([0.1, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5])
    ctrl.set_arm_positions(arm)
    pos = ctrl.get_joint_positions()
    # La pince doit rester ouverte
    assert abs(pos[7] - 0.04) < 0.001
    print("✓ test_arm_positions")


def test_fk_home():
    """Test : FK à la position home donne une position cohérente."""
    kin = PandaKinematics()
    home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    pose = kin.forward_kinematics(home)

    # Le TCP doit être devant le robot (x > 0)
    assert pose["position"][0] > 0
    # Le TCP doit être au-dessus du sol (z > 0)
    assert pose["position"][2] > 0
    # Le quaternion doit être normalisé
    quat_norm = np.linalg.norm(pose["orientation"])
    assert abs(quat_norm - 1.0) < 0.01
    print(f"✓ test_fk_home — TCP: {pose['position'].round(3)}")


def test_fk_zero():
    """Test : FK avec tous les angles à zéro."""
    kin = PandaKinematics()
    zeros = [0.0] * 7
    pose = kin.forward_kinematics(zeros)
    assert pose["position"] is not None
    assert len(pose["position"]) == 3
    print(f"✓ test_fk_zero — TCP: {pose['position'].round(3)}")


def test_ik_reachable():
    """Test : IK converge vers une position atteignable."""
    kin = PandaKinematics()
    target = [0.4, 0.0, 0.5]
    result = kin.inverse_kinematics(
        target_position=target,
        max_iterations=1000,
        tolerance=0.01
    )
    # Vérifie que l'erreur est raisonnable
    assert result["error"] < 0.1  # moins de 10cm
    assert len(result["angles"]) == 7
    print(f"✓ test_ik_reachable — erreur: {result['error']*1000:.1f}mm")


def test_ik_unreachable():
    """Test : IK échoue proprement pour position inatteignable."""
    kin = PandaKinematics()
    target = [5.0, 0.0, 0.0]  # 5m = hors portée
    result = kin.inverse_kinematics(
        target_position=target,
        max_iterations=100
    )
    assert not result["success"]
    print(f"✓ test_ik_unreachable — succès: {result['success']}")


def test_reachability():
    """Test : détection correcte des zones atteignables."""
    kin = PandaKinematics()
    assert kin.is_reachable([0.4, 0.0, 0.4])      # atteignable
    assert not kin.is_reachable([2.0, 0.0, 0.0])  # trop loin
    assert not kin.is_reachable([0.0, 0.0, 0.0])  # trop près
    print("✓ test_reachability")


def test_all_joint_positions():
    """Test : positions de tous les joints."""
    kin = PandaKinematics()
    home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    positions = kin.get_all_joint_positions(home)
    assert len(positions) == 9  # base + 7 joints + TCP
    print(f"✓ test_all_joint_positions — {len(positions)} points")


def run_all_tests():
    """Lance tous les tests."""
    print("\n" + "="*50)
    print("  TESTS UNITAIRES — Robot")
    print("="*50)

    tests = [
        test_controller_init,
        test_default_position,
        test_gripper_open,
        test_gripper_close,
        test_gripper_partial,
        test_joint_limits,
        test_arm_positions,
        test_fk_home,
        test_fk_zero,
        test_ik_reachable,
        test_ik_unreachable,
        test_reachability,
        test_all_joint_positions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} — ÉCHEC : {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} — ERREUR : {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Résultat : {passed}/{len(tests)} tests passés")
    if failed == 0:
        print("  ✓ Tous les tests passent !")
    else:
        print(f"  ✗ {failed} test(s) échoué(s)")
    print("="*50)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)