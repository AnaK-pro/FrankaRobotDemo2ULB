# Demo2 — Simulation Franka Panda avec Isaac Sim

Simulation d'un bras robotique Franka Panda apprenant à attraper
et poser un cube, en utilisant NVIDIA Isaac Sim et le RL.

## Prérequis
- Ubuntu 22.04
- NVIDIA Isaac Sim 4.x
- Conda (environnement `isaac`)

## Installation

```bash
conda activate isaac
pip install -r requirements.txt
pip install -e .
```

## Lancer l'entraînement

```bash
python scripts/train.py
```

## Lancer la visualisation

```bash
python scripts/visualize.py
```

## Structure du projet

```
Demo2/
├── config/          # Fichiers de configuration YAML
├── assets/          # Modèles 3D, scènes, matériaux
├── src/             # Code source principal
│   ├── robot/       # Contrôleur et cinématique
│   ├── env/         # Environnements Gym
│   ├── sensors/     # Caméra et LiDAR
│   ├── tasks/       # Tâches robotiques
│   ├── rl/          # Policy et récompenses
│   ├── scene/       # Construction de scène
│   └── utils/       # Utilitaires
├── scripts/         # Points d'entrée
├── tests/           # Tests unitaires
└── logs/            # Checkpoints et TensorBoard
```