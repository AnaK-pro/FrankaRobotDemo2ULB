# scripts/visualize.py
# Lance Isaac Sim en mode graphique avec rendu RTX
# Permet de visualiser le robot et la scène en 3D

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualisation Isaac Sim"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin vers un modèle entraîné à visualiser"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Nombre d'épisodes à visualiser (défaut: 5)"
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Ralentit la simulation pour mieux voir"
    )
    return parser.parse_args()


def run_random_policy(env, n_episodes=3):
    """
    Fait tourner une politique aléatoire dans la scène.
    Utile pour vérifier que la simulation fonctionne.
    """
    print("\n[Visualize] Politique aléatoire...")

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n  Épisode {ep+1}/{n_episodes}")
        total_reward = 0.0
        step = 0

        while True:
            # Action aléatoire douce (petits déplacements)
            action = np.array([
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.01, 0.01),
                1.0  # pince ouverte
            ])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 50 == 0:
                print(
                    f"    Pas {step:4d} : "
                    f"reward={reward:+.3f} "
                    f"état={info.get('task_state','?')}"
                )

            if terminated or truncated:
                print(f"    → Terminé : {total_reward:.2f} reward")
                break


def run_trained_policy(env, checkpoint_path, n_episodes=5):
    """
    Charge un modèle entraîné et le visualise.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[Visualize] stable-baselines3 requis")
        return

    print(f"\n[Visualize] Chargement : {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
    print("[Visualize] Modèle chargé ✓")

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n  Épisode {ep+1}/{n_episodes}")
        total_reward = 0.0
        step = 0

        while True:
            # Prédit l'action avec le modèle entraîné
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 20 == 0:
                print(
                    f"    Pas {step:4d} : "
                    f"reward={reward:+.3f} "
                    f"état={info.get('task_state','?')} "
                    f"triées={info.get('sorted', 0)}/3"
                )

            if terminated or truncated:
                summary = info.get("episode_summary", {})
                success = summary.get("success", False)
                print(
                    f"    → {'✓ SUCCÈS' if success else '✗ Échec'} — "
                    f"reward: {total_reward:.2f} — "
                    f"steps: {step}"
                )
                break


def visualize(args):
    """Lance la visualisation."""
    from src.env.task_env import ShapeSortingEnv

    print("\n" + "="*50)
    print("  VISUALISATION Isaac Sim")
    print("="*50)
    if args.checkpoint:
        print(f"  Modèle    : {args.checkpoint}")
    else:
        print(f"  Mode      : Politique aléatoire")
    print(f"  Épisodes  : {args.episodes}")
    print("="*50 + "\n")

    # Crée l'environnement avec rendu graphique
    env = ShapeSortingEnv(render_mode="human")

    try:
        if args.checkpoint and os.path.exists(args.checkpoint):
            run_trained_policy(env, args.checkpoint, args.episodes)
        else:
            if args.checkpoint:
                print(f"[Visualize] Checkpoint non trouvé : {args.checkpoint}")
                print("[Visualize] Lancement en mode aléatoire...")
            run_random_policy(env, args.episodes)

    except KeyboardInterrupt:
        print("\n[Visualize] Arrêté par l'utilisateur")

    finally:
        env.close()
        print("[Visualize] Fermé ✓")


if __name__ == "__main__":
    args = parse_args()
    visualize(args)