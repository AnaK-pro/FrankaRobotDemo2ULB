"""
scripts/train_vec.py
Entraînement PPO vectorisé — N robots Franka en parallèle dans Isaac Sim.

Isaac Sim s'ouvre automatiquement. Pour lancer :

    conda activate isaac
    cd /home/anakpro/stage/Demo2
    python scripts/train_vec.py                        # 40 robots, GUI
    python scripts/train_vec.py --num-envs 8           # 8 robots (debug)
    python scripts/train_vec.py --headless             # sans GUI (plus rapide)
    python scripts/train_vec.py --load-checkpoint logs/checkpoints/.../model

⚠ SimulationApp DOIT être créée ici, avant tout import omni/isaacsim.
  Ne pas déplacer les lignes SimulationApp plus bas dans le fichier.
"""

import os
import sys
import argparse
from datetime import datetime

# ── Path projet ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 1. SimulationApp en PREMIER — avant tout import omni/isaacsim ─────────────
try:
    from isaacsim.simulation_app import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement RL vectorisé — N robots Franka dans Isaac Sim"
    )
    parser.add_argument("--num-envs",        type=int,   default=40,
                        help="Nombre de robots en parallèle (défaut: 40)")
    parser.add_argument("--total-steps",     type=int,   default=2_000_000,
                        help="Pas d'entraînement total (défaut: 2 000 000)")
    parser.add_argument("--env-spacing",     type=float, default=3.0,
                        help="Espacement entre envs en mètres (défaut: 3.0)")
    parser.add_argument("--headless",        action="store_true",
                        help="Pas d'interface graphique (plus rapide)")
    parser.add_argument("--checkpoint-freq", type=int,   default=20_000,
                        help="Fréquence de sauvegarde des checkpoints")
    parser.add_argument("--load-checkpoint", type=str,   default=None,
                        help="Chemin d'un checkpoint à reprendre")
    parser.add_argument("--learning-rate",   type=float, default=3e-4)
    parser.add_argument("--n-steps",         type=int,   default=128,
                        help="Pas collectés par env avant chaque update PPO")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Ouvrir Isaac Sim ──────────────────────────────────────────────────────
    sim_app = SimulationApp({
        "headless":       args.headless,
        "width":          1920,
        "height":         1080,
        "anti_aliasing":  1,
    })
    print(f"[TrainVec] Isaac Sim {'headless' if args.headless else 'GUI'} ouvert ✓")

    # ── 2. Imports après SimulationApp ────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from src.env.isaac_vec_env import IsaacVecEnv

    try:
        from torch.utils.tensorboard import SummaryWriter
        TB_AVAILABLE = True
    except ImportError:
        TB_AVAILABLE = False

    # ── Dossiers de logs ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"shape_sorting_vec{args.num_envs}_{timestamp}"
    log_dir   = os.path.join("logs", "tensorboard", run_name)
    ckpt_dir  = os.path.join("logs", "checkpoints", run_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT RL VECTORISÉ — Tri de formes")
    print("=" * 60)
    print(f"  Robots en parallèle : {args.num_envs}")
    print(f"  Total steps         : {args.total_steps:,}")
    print(f"  Espacement grille   : {args.env_spacing} m")
    print(f"  Mode rendu          : {'headless' if args.headless else 'GUI'}")
    print(f"  TensorBoard         : {log_dir}")
    print(f"  Checkpoints         : {ckpt_dir}")
    print("=" * 60 + "\n")

    # ── Créer l'environnement vectorisé ───────────────────────────────────────
    env = IsaacVecEnv(
        num_envs    = args.num_envs,
        max_steps   = 1000,
        env_spacing = args.env_spacing,
        render      = not args.headless,
    )
    # VecNormalize : normalise obs et rewards en ligne — indispensable pour PPO
    norm_path = os.path.join(ckpt_dir, "vecnorm.pkl")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Modèle PPO ────────────────────────────────────────────────────────────
    if args.load_checkpoint:
        print(f"[TrainVec] Reprise depuis : {args.load_checkpoint}")
        model = PPO.load(args.load_checkpoint, env=env, tensorboard_log=log_dir)
    else:
        # n_steps=128 × 40 envs = 5 120 pas/update (court → gradients plus fréquents)
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            device          = "cpu",
            learning_rate   = 3e-4,
            n_steps         = 128,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.05,   # exploration plus agressive au début
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            verbose         = 1,
            tensorboard_log = log_dir,
            policy_kwargs   = {"net_arch": [256, 256, 128]},
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    class TaskMetricsCallback(BaseCallback):
        """Log les métriques de tri dans TensorBoard."""
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode_summary" in info:
                    s = info["episode_summary"]
                    self.logger.record("task/shapes_sorted",   s["nb_sorted"])
                    self.logger.record("task/success",         float(s["success"]))
                    self.logger.record("task/episode_reward",  s["total_reward"])
            return True

    callbacks = [
        CheckpointCallback(
            save_freq   = max(args.checkpoint_freq // args.num_envs, 1),
            save_path   = ckpt_dir,
            name_prefix = "panda_vec",
            verbose     = 1,
        ),
        TaskMetricsCallback(verbose=0),
    ]

    # ── Entraînement ──────────────────────────────────────────────────────────
    print("[TrainVec] Démarrage de l'entraînement...")
    print(f"[TrainVec] Lance TensorBoard : tensorboard --logdir {log_dir}\n")

    try:
        model.learn(
            total_timesteps      = args.total_steps,
            callback             = callbacks,
            reset_num_timesteps  = args.load_checkpoint is None,
            tb_log_name          = run_name,
        )
    except KeyboardInterrupt:
        print("\n[TrainVec] Interrompu par l'utilisateur")

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_model")
    model.save(final_path)
    env.save(norm_path)  # sauvegarde les stats VecNormalize
    print(f"\n[TrainVec] Modèle sauvegardé : {final_path}")
    print(f"[TrainVec] Stats normalization : {norm_path}")

    env.close()
    sim_app.close()
    print("[TrainVec] Terminé ✓")


if __name__ == "__main__":
    main()
