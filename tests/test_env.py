# tests/test_env.py
# Tests d'intégration pour l'environnement complet

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.task_env import ShapeSortingEnv
from src.scene.scene_builder import SceneBuilder
from src.tasks.pick_place import ShapeSortingTask


def test_env_init():
    """Test : l'environnement s'initialise correctement."""
    env = ShapeSortingEnv()
    assert env.action_space is not None
    assert env.observation_space is not None
    assert env.max_steps == 1000
    env.close()
    print("✓ test_env_init")


def test_action_space():
    """Test : l'espace d'action est correct."""
    env = ShapeSortingEnv()
    assert env.action_space.shape == (4,)
    assert env.action_space.low[3] == 0.0   # pince min
    assert env.action_space.high[3] == 1.0  # pince max
    env.close()
    print("✓ test_action_space")


def test_observation_space():
    """Test : l'espace d'observation est correct."""
    env = ShapeSortingEnv()
    assert env.observation_space.shape == (40,)
    env.close()
    print("✓ test_observation_space")


def test_reset():
    """Test : reset retourne une observation valide."""
    env = ShapeSortingEnv()
    obs, info = env.reset()

    assert obs.shape == (40,)
    assert obs.dtype == np.float32
    assert "episode" in info or "ep_num" in info
    assert not np.any(np.isnan(obs))
    assert not np.any(np.isinf(obs))
    env.close()
    print("✓ test_reset")


def test_step_output():
    """Test : step retourne les bonnes valeurs."""
    env = ShapeSortingEnv()
    env.reset()

    action = np.array([0.01, 0.0, 0.0, 1.0])
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (40,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert not np.any(np.isnan(obs))
    env.close()
    print("✓ test_step_output")


def test_action_clipping():
    """Test : les actions hors limites sont clippées."""
    env = ShapeSortingEnv()
    env.reset()

    # Action hors limites
    extreme_action = np.array([10.0, -10.0, 10.0, 5.0])
    obs, reward, _, _, _ = env.step(extreme_action)

    assert obs.shape == (40,)
    assert not np.any(np.isnan(obs))
    env.close()
    print("✓ test_action_clipping")


def test_multiple_steps():
    """Test : plusieurs steps successifs sans crash."""
    env = ShapeSortingEnv()
    env.reset()

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs))
        if terminated or truncated:
            break

    env.close()
    print("✓ test_multiple_steps")


def test_truncation():
    """Test : l'épisode se termine après max_steps."""
    env = ShapeSortingEnv()
    # Réduit max_steps pour le test
    env.max_steps = 5
    env.task.max_steps = 5
    env.reset()

    done = False
    steps = 0
    while not done:
        action = np.array([0.0, 0.0, 0.0, 1.0])
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if steps > 10:
            break

    assert steps <= 6  # max_steps + 1 tolérance
    env.close()
    print(f"✓ test_truncation — terminé en {steps} pas")


def test_multiple_episodes():
    """Test : plusieurs épisodes consécutifs."""
    env = ShapeSortingEnv()

    for ep in range(3):
        obs, info = env.reset()
        assert obs.shape == (40,)

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    assert env.episode_count == 3
    env.close()
    print("✓ test_multiple_episodes")


def test_scene_builder():
    """Test : le SceneBuilder gère correctement les formes."""
    scene = SceneBuilder()
    scene.build()

    # Vérifie les 3 formes
    for name in ["cube", "cylinder", "pyramid"]:
        pos = scene.get_shape_position(name)
        assert len(pos) == 3
        assert not np.any(np.isnan(pos))

    # Vérifie les 3 bacs
    for name in ["bin_cube", "bin_cylinder", "bin_pyramid"]:
        pos = scene.get_bin_position(name)
        assert len(pos) == 3

    print("✓ test_scene_builder")


def test_correct_bin_mapping():
    """Test : chaque forme est correctement liée à son bac."""
    scene = SceneBuilder()
    scene.build()

    assert scene.get_correct_bin_for_shape("cube")     == "bin_cube"
    assert scene.get_correct_bin_for_shape("cylinder") == "bin_cylinder"
    assert scene.get_correct_bin_for_shape("pyramid")  == "bin_pyramid"
    print("✓ test_correct_bin_mapping")


def test_task_reset():
    """Test : la tâche se réinitialise correctement."""
    task = ShapeSortingTask()
    obs = task.reset()

    assert len(obs) == 27
    assert task.current_shape_idx == 0
    assert task.current_state == "idle"
    assert not task.is_holding
    assert task.step_count == 0
    assert all(not v for v in task.sorted_shapes.values())
    print("✓ test_task_reset")


def test_task_state_machine():
    """Test : la machine à états de la tâche fonctionne."""
    task = ShapeSortingTask()
    scene = SceneBuilder()
    scene.build()
    task.reset()

    # État initial
    assert task.current_state == "idle"

    # Un pas → reaching
    action = np.array([0.0, 0.0, 0.0, 1.0])
    task.step(action, scene)
    assert task.current_state == "reaching"

    # Force grasping
    task.set_state("grasping")
    action_grasp = np.array([0.0, 0.0, 0.0, 0.0])
    task.step(action_grasp, scene)
    assert task.is_holding
    assert task.current_state == "carrying"

    print("✓ test_task_state_machine")


def test_reward_range():
    """Test : les récompenses sont dans une plage raisonnable."""
    env = ShapeSortingEnv()
    env.reset()

    rewards = []
    for _ in range(50):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            env.reset()

    # Récompenses pas trop extrêmes
    assert max(rewards) < 20.0
    assert min(rewards) > -10.0
    env.close()
    print(f"✓ test_reward_range — min={min(rewards):.2f} max={max(rewards):.2f}")


def run_all_tests():
    """Lance tous les tests d'intégration."""
    print("\n" + "="*50)
    print("  TESTS D'INTÉGRATION — Environnement")
    print("="*50)

    tests = [
        test_env_init,
        test_action_space,
        test_observation_space,
        test_reset,
        test_step_output,
        test_action_clipping,
        test_multiple_steps,
        test_truncation,
        test_multiple_episodes,
        test_scene_builder,
        test_correct_bin_mapping,
        test_task_reset,
        test_task_state_machine,
        test_reward_range,
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