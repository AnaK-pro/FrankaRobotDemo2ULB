# scripts/train.py
# Script de lancement de l'entraînement RL
# Lance l'algorithme PPO sur la tâche de tri de formes

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import Any

# Ajoute la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.task_env import ShapeSortingEnv

# TensorBoard pour visualiser l'entraînement
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[Train] TensorBoard non disponible")

# Stable Baselines 3 — bibliothèque d'algorithmes RL
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        BaseCallback  # type: ignore[assignment]
    )
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Train] Stable Baselines 3 non disponible")
    print("        Installe avec : pip install stable-baselines3")

    class BaseCallback:
        def __init__(self, verbose: int = 0) -> None:
            self.verbose  = verbose
            self.locals:  dict = {}
            self.logger:  Any  = None

    PPO:                Any = None
    CheckpointCallback: Any = None
    EvalCallback:       Any = None
    Monitor:            Any = None


class TrainingMetricsCallback(BaseCallback):
    """
    Callback personnalisé pour logger les métriques
    spécifiques à notre tâche de tri de formes.

    Un callback est une fonction appelée automatiquement
    pendant l'entraînement à des moments précis
    (après chaque pas, après chaque épisode, etc.)
    """

    def __init__(self, log_dir, verbose=1):
        super().__init__(verbose)
        self.log_dir    = log_dir
        self.episode_rewards  = []
        self.episode_lengths  = []
        self.shapes_sorted    = []

    def _on_step(self):
        """Appelé à chaque pas de simulation."""

        # Récupère les infos de fin d'épisode
        for info in self.locals.get("infos", []):
            if "episode_summary" in info:
                summary = info["episode_summary"]

                # Log dans TensorBoard
                if TENSORBOARD_AVAILABLE and self.logger:
                    self.logger.record(
                        "task/shapes_sorted",
                        summary["nb_sorted"]
                    )
                    self.logger.record(
                        "task/success",
                        float(summary["success"])
                    )
                    self.logger.record(
                        "task/episode_reward",
                        summary["total_reward"]
                    )

                if self.verbose > 0:
                    print(
                        f"[Callback] Épisode terminé — "
                        f"triées: {summary['nb_sorted']}/3 "
                        f"succès: {summary['success']} "
                        f"reward: {summary['total_reward']:.2f}"
                    )

        return True  # True = continuer l'entraînement


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Entraînement RL pour le tri de formes"
    )
    parser.add_argument(
        "--total-steps", type=int, default=500_000,
        help="Nombre total de pas d'entraînement (défaut: 500000)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10_000,
        help="Fréquence de sauvegarde des checkpoints"
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None,
        help="Chemin vers un checkpoint à charger pour reprendre"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Active le rendu graphique (plus lent)"
    )
    parser.add_argument(
        "--use-ros", action="store_true",
        help="Active le bridge ROS 2"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Taux d'apprentissage du réseau (défaut: 0.0003)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Pas par mise à jour PPO (défaut: 2048)"
    )
    return parser.parse_args()


def make_env(args):
    """
    Crée et configure l'environnement d'entraînement.

    Monitor = wrapper Gym qui enregistre automatiquement
    les récompenses et longueurs d'épisodes dans un fichier CSV.
    """
    render_mode = "human" if args.render else None

    env = ShapeSortingEnv(
        render_mode=render_mode,
        use_ros=args.use_ros
    )

    if SB3_AVAILABLE:
        # Monitor enregistre les stats dans logs/
        log_path = os.path.join("logs", "monitor")
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, log_path)

    return env


def train(args):
    """
    Lance l'entraînement avec l'algorithme PPO.

    PPO = Proximal Policy Optimization
    C'est l'algorithme RL le plus utilisé en robotique.
    Il apprend en alternant :
    1. Collecte d'expériences (le robot agit)
    2. Mise à jour du réseau (on améliore la policy)
    """
    # Dossiers de logs et checkpoints
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"shape_sorting_{timestamp}"
    log_dir    = os.path.join("logs", "tensorboard", run_name)
    ckpt_dir   = os.path.join("logs", "checkpoints", run_name)

    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "="*50)
    print("  ENTRAÎNEMENT RL — Tri de formes")
    print("="*50)
    print(f"  Run          : {run_name}")
    print(f"  Total steps  : {args.total_steps:,}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Checkpoints  : {ckpt_dir}")
    print(f"  TensorBoard  : {log_dir}")
    print("="*50 + "\n")

    # Crée l'environnement
    env = make_env(args)

    if not SB3_AVAILABLE:
        print("[Train] Stable Baselines 3 requis pour l'entraînement.")
        print("        Lance : pip install stable-baselines3")
        print("\n[Train] Test de l'environnement sans RL...")
        _test_env_loop(env, n_steps=20)
        env.close()
        return

    # Crée ou charge le modèle PPO
    if args.load_checkpoint:
        print(f"[Train] Chargement checkpoint : {args.load_checkpoint}")
        model = PPO.load(
            args.load_checkpoint,
            env=env,
            tensorboard_log=log_dir
        )
    else:
        # Architecture du réseau de neurones :
        # policy="MlpPolicy" = réseau dense (MLP)
        # net_arch = taille des couches cachées
        # [256, 256] = 2 couches de 256 neurones chacune
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,           # facteur de décompte des récompenses futures
            gae_lambda=0.95,      # GAE lambda pour l'estimation d'avantage
            clip_range=0.15,       # clip PPO (stabilité)
            ent_coef=0.01,        # coefficient d'entropie (exploration)
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs={
                "net_arch": [256, 256]  # 2 couches cachées de 128 neurones
            }
        )

    # Callbacks
    callbacks = []

    # 1. Sauvegarde automatique des checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="panda_shape_sorting",
        verbose=1
    )
    callbacks.append(checkpoint_cb)

    # 2. Métriques custom
    metrics_cb = TrainingMetricsCallback(
        log_dir=log_dir,
        verbose=1
    )
    callbacks.append(metrics_cb)

    # Lance l'entraînement
    print("[Train] Démarrage de l'entraînement...")
    print("[Train] Lance TensorBoard avec :")
    print(f"        tensorboard --logdir {log_dir}\n")

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            reset_num_timesteps=args.load_checkpoint is None,
            tb_log_name=run_name
        )
    except KeyboardInterrupt:
        print("\n[Train] Entraînement interrompu par l'utilisateur")

    # Sauvegarde finale
    final_path = os.path.join(ckpt_dir, "final_model")
    model.save(final_path)
    print(f"\n[Train] Modèle final sauvegardé : {final_path}")

    env.close()
    print("[Train] Terminé ✓")


def _test_env_loop(env, n_steps=20):
    """
    Test de l'environnement sans algorithme RL.
    Utile pour vérifier que tout fonctionne.
    """
    print(f"\n[Train] Test de {n_steps} pas aléatoires...")
    obs, info = env.reset()
    total_reward = 0.0

    for i in range(n_steps):
        # Action aléatoire dans l'espace d'action
        if hasattr(env, 'action_space'):
            action = env.action_space.sample()
        else:
            action = np.array([0.01, 0.0, 0.0, 1.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"  Pas {i+1:2d} : "
            f"reward={reward:+.3f} "
            f"état={info.get('task_state','?'):10s} "
            f"triées={info.get('sorted', {})}"
        )

        if terminated or truncated:
            print(f"  → Épisode terminé (reward total: {total_reward:.3f})")
            obs, info = env.reset()
            total_reward = 0.0

    print(f"[Train] Test terminé — reward total : {total_reward:.3f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)