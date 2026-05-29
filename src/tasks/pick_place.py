# src/tasks/pick_place.py
# Logique de la tâche de tri de formes
# Gère : quel objet saisir, où le déposer, et comment noter le robot

import numpy as np
import yaml
import os


class ShapeSortingTask:
    """
    Tâche de tri de formes pour le Franka Panda.

    Le robot doit :
    1. Identifier chaque forme sur la table
    2. La saisir avec la pince
    3. La déposer dans le bon bac
    4. Répéter pour les 3 formes

    États :
    - REACHING : se déplacer vers la forme cible
    - CARRYING : transporter la forme vers le bon bac
    - DONE     : toutes les formes sont triées
    """

    # États de la tâche (machine simplifiée à 2 états)
    STATE_REACHING  = "reaching"
    STATE_CARRYING  = "carrying"
    STATE_DONE      = "done"

    def __init__(self, task_config_path="config/task.yaml", max_shapes: int = 3):
        self.config = self._load_yaml(task_config_path)
        cfg         = self.config["task"]

        # Paramètres de récompense
        self.reward_correct_bin    = cfg["reward"]["correct_bin"]
        self.reward_wrong_bin      = cfg["reward"]["wrong_bin"]
        self.reward_drop           = cfg["reward"]["drop_penalty"]
        self.reward_step           = cfg["reward"]["step_penalty"]
        self.reward_grasp_bonus    = cfg["reward"]["grasp_bonus"]
        self.reward_all_sorted     = cfg["reward"]["all_sorted_bonus"]
        self.placement_tolerance   = cfg["placement_tolerance"]
        self.max_steps             = cfg["max_episode_steps"]

        # Liste des formes à trier
        self.shapes_config = cfg["shapes"]
        self.bins_config   = cfg["bins"]

        # Curriculum : nombre de formes actives (1 → 2 → 3)
        self.max_shapes: int = min(max(1, max_shapes), len(self.shapes_config))

        # État interne de l'épisode
        self._reset_state()

    def _load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config introuvable : {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _reset_state(self):
        """Remet l'état interne à zéro pour un nouvel épisode."""

        # Quelle forme est en cours de traitement
        self.current_shape_idx = 0
        self.current_state     = self.STATE_REACHING

        # Suivi des formes
        # sorted_shapes : dict nom → True/False
        self.sorted_shapes = {
            s["name"]: False for s in self.shapes_config
        }

        # Est-ce que le robot tient une forme ?
        self.is_holding     = False
        self.held_shape     = None

        # Distances précédentes pour le reward différentiel
        self.prev_dist_tcp_shape = None
        self.prev_dist_to_bin    = None

        # Bonus du step courant (saisie + tri) — toujours ≥ 0
        # Réinitialisé au début de chaque _evaluate_state
        self._last_bonus: float = 0.0

        # Nombre de steps depuis le dernier grasp (évite les lâchers accidentels)
        self._carry_steps: int = 0

        # Compteurs
        self.step_count     = 0
        self.total_reward   = 0.0
        self.episode_done   = False

        # Historique pour le débogage
        self.history = []

    # ─────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────

    def reset(self):
        self._reset_state()
        return self._get_observation()

    # ─────────────────────────────────────────────
    # STEP — avance d'un pas
    # ─────────────────────────────────────────────

    def step(self, action, scene_builder, tcp_pos=None):
        """
        Avance la tâche d'un pas de simulation.

        action        : [dx, dy, dz, gripper]  gripper 0=fermé 1=ouvert
        scene_builder : SceneBuilder pour lire les positions des objets
        tcp_pos       : position 3D du TCP (optionnel) — utilisée pour le
                        reward de rapprochement en phase REACHING

        Retourne (observation, reward, done, info)
        """
        self.step_count += 1
        reward = self.reward_step  # malus par défaut à chaque pas

        # Récupère les positions actuelles
        shapes_info = scene_builder.get_all_shapes_info()

        # Calcule les récompenses selon l'état
        reward += self._evaluate_state(action, shapes_info, tcp_pos=tcp_pos)

        # Vérifie si l'épisode est terminé
        done = self._check_done()

        # Observation
        obs = self._get_observation(shapes_info)

        self.total_reward += reward
        self.history.append({
            "step":   self.step_count,
            "state":  self.current_state,
            "reward": round(reward, 3)
        })

        info = {
            "state":           self.current_state,
            "sorted":          self.sorted_shapes.copy(),
            "nb_sorted":       sum(self.sorted_shapes.values()),
            "is_holding":      self.is_holding,
            "held_shape":      self.held_shape,
            "step":            self.step_count,
            "total_reward":    round(self.total_reward, 3),
            "dist_tcp_shape":  getattr(self, "_last_dist_tcp_shape", None),
            "contact_bonus":   round(float(self._last_bonus), 4),  # ≥ 0 toujours
        }

        return obs, reward, done, info

    # ─────────────────────────────────────────────
    # LOGIQUE D'ÉVALUATION
    # ─────────────────────────────────────────────

    def _evaluate_state(self, action, shapes_info, tcp_pos=None):
        """
        Machine d'états simplifiée : REACHING → CARRYING → (succès ou retour REACHING).

        REACHING : reward dense sur distance TCP→forme.
                   Transition → CARRYING si pince fermée ET TCP proche (<15 cm).
        CARRYING : reward dense sur distance forme→bac (fake-grasping assure que la
                   forme suit le TCP).
                   Transition → succès si pince ouverte ET forme dans bac.
                   Transition → REACHING si pince ouverte hors bac (pénalité drop).
        """
        reward = 0.0
        self._last_bonus = 0.0  # réinitialisé chaque step : ne track que les bonus du step courant

        active = list(self.sorted_shapes.keys())[:self.max_shapes]
        if all(self.sorted_shapes[s] for s in active):
            self.current_state = self.STATE_DONE
            return 0.0

        if self.current_shape_idx >= self.max_shapes:
            return 0.0

        current_shape = self.shapes_config[self.current_shape_idx]["name"]
        shape_data    = shapes_info.get(current_shape, {})

        if not shape_data:
            return reward

        dist_to_bin = float(shape_data["distance_to_bin"])
        is_sorted   = shape_data.get("is_sorted", False)
        shape_pos   = np.array(shape_data["position"], dtype=np.float32)

        gripper        = float(action[-1])
        gripper_closed = gripper < 0.5   # 50% de chance au départ (mean≈0.5, std≈0.1)
        gripper_open   = gripper > 0.6

        dist_tcp_shape = 999.0
        if tcp_pos is not None:
            dist_tcp_shape = float(np.linalg.norm(
                np.array(tcp_pos, dtype=np.float32) - shape_pos
            ))
        self._last_dist_tcp_shape = dist_tcp_shape

        # ── REACHING ──────────────────────────────────────────────────────
        if self.current_state != self.STATE_CARRYING:
            self.current_state = self.STATE_REACHING

            if dist_tcp_shape < 999.0:
                if self.prev_dist_tcp_shape is None:
                    self.prev_dist_tcp_shape = dist_tcp_shape
                else:
                    # Reward seulement si on se rapproche (+) ou pénalité si on s'éloigne (-)
                    reward += (self.prev_dist_tcp_shape - dist_tcp_shape) * 10.0
                    self.prev_dist_tcp_shape = dist_tcp_shape

            if gripper_closed and dist_tcp_shape < 0.25:  # élargi 0.20→0.25
                self.is_holding       = True
                self.held_shape       = current_shape
                self.current_state    = self.STATE_CARRYING
                self.prev_dist_to_bin = None
                self._carry_steps     = 0   # compteur de durée de portage
                reward += self.reward_grasp_bonus
                self._last_bonus += self.reward_grasp_bonus

        # ── CARRYING ──────────────────────────────────────────────────────
        else:
            self._carry_steps += 1

            # Lâcher autorisé seulement après MIN_CARRY_STEPS
            # Empêche les lâchers accidentels dus à un bruit de politique au début du portage
            MIN_CARRY_STEPS = 5
            if gripper_open and self._carry_steps >= MIN_CARRY_STEPS:
                self.is_holding = False
                self.held_shape = None
                if is_sorted:
                    reward += self.reward_correct_bin
                    self._last_bonus += self.reward_correct_bin
                    self.sorted_shapes[current_shape] = True
                    self.current_shape_idx += 1
                    self.prev_dist_tcp_shape = None
                    if all(self.sorted_shapes.values()):
                        reward += self.reward_all_sorted
                        self._last_bonus += self.reward_all_sorted
                        self.current_state = self.STATE_DONE
                    else:
                        self.current_state = self.STATE_REACHING
                else:
                    reward += self.reward_drop
                    self.current_state       = self.STATE_REACHING
                    self.prev_dist_tcp_shape = None
            else:
                # Maintien du portage : reward pour avancer vers le bac
                if self.prev_dist_to_bin is None:
                    self.prev_dist_to_bin = dist_to_bin
                else:
                    reward += (self.prev_dist_to_bin - dist_to_bin) * 10.0
                    self.prev_dist_to_bin = dist_to_bin

        return reward

    # ─────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────

    def _get_observation(self, shapes_info=None):
        """
        Construit le vecteur d'observation envoyé au réseau de neurones.

        Le réseau de neurones reçoit un vecteur numérique qui décrit
        l'état complet de la scène. Il ne "voit" pas les images —
        il reçoit des chiffres.

        Vecteur d'observation (30 valeurs) :
        - Position de chaque forme       : 3 × 3 = 9 valeurs
        - Position de chaque bac         : 3 × 3 = 9 valeurs
        - État du tri (0 ou 1)           : 3 × 1 = 3 valeurs
        - Indice forme courante (one-hot): 3 valeurs
        - Est-ce qu'on tient une forme ? : 1 valeur
        - Indice de l'état               : 1 valeur
        - Pas restants normalisé         : 1 valeur
        Total                            : 30 valeurs
        """
        obs = []

        # Positions des formes
        if shapes_info:
            for shape_cfg in self.shapes_config:
                name = shape_cfg["name"]
                pos  = shapes_info[name]["position"] if name in shapes_info \
                    else shape_cfg["spawn_position"]
                obs.extend(pos)
        else:
            for shape_cfg in self.shapes_config:
                obs.extend(shape_cfg["spawn_position"])

        # Positions des bacs
        for bin_cfg in self.bins_config:
            obs.extend(bin_cfg["position"])

        # État du tri
        for shape_cfg in self.shapes_config:
            obs.append(float(self.sorted_shapes[shape_cfg["name"]]))

        # Forme courante (one-hot encoding)
        # Ex : forme 1 = [0, 1, 0]
        one_hot = [0.0, 0.0, 0.0]
        if self.current_shape_idx < 3:
            one_hot[self.current_shape_idx] = 1.0
        obs.extend(one_hot)

        # Tient une forme ?
        obs.append(float(self.is_holding))

        # État actuel encodé en chiffre
        state_map = {
            self.STATE_REACHING: 0.0,
            self.STATE_CARRYING: 1.0,
            self.STATE_DONE:     2.0,
        }
        obs.append(state_map.get(self.current_state, 0.0))

        # Pas restants normalisé entre 0 et 1
        obs.append(1.0 - self.step_count / self.max_steps)

        return np.array(obs, dtype=np.float32)

    # ─────────────────────────────────────────────
    # FIN D'ÉPISODE
    # ─────────────────────────────────────────────

    def _check_done(self):
        active = list(self.sorted_shapes.keys())[:self.max_shapes]
        if all(self.sorted_shapes[s] for s in active):
            self.episode_done = True
            return True
        if self.step_count >= self.max_steps:
            self.episode_done = True
            return True
        return False

    # ─────────────────────────────────────────────
    # UTILITAIRES
    # ─────────────────────────────────────────────

    def set_state(self, state):
        """Force l'état interne de la tâche."""
        self.current_state = state

    def get_current_target(self):
        """Retourne (shape_name, bin_name) de la forme en cours, ou (None, None)."""
        if self.current_shape_idx >= len(self.shapes_config):
            return None, None
        shape_name = self.shapes_config[self.current_shape_idx]["name"]
        bin_name = None
        for b in self.bins_config:
            if b.get("shape") == shape_name:
                bin_name = b["name"]
                break
        return shape_name, bin_name

    def get_episode_summary(self):
        """Résumé de fin d'épisode avec indicateur de succès."""
        active    = list(self.sorted_shapes.keys())[:self.max_shapes]
        nb_sorted = sum(self.sorted_shapes[s] for s in active)
        success   = nb_sorted == self.max_shapes
        return {
            "success":      success,
            "nb_sorted":    nb_sorted,
            "total":        self.max_shapes,
            "max_shapes":   self.max_shapes,
            "step_count":   self.step_count,
            "total_reward": round(self.total_reward, 3),
            "sorted":       self.sorted_shapes.copy(),
        }

    def get_progress(self):
        """Retourne la progression du tri."""
        nb_sorted = sum(self.sorted_shapes.values())
        total     = len(self.shapes_config)
        return {
            "sorted":     nb_sorted,
            "total":      total,
            "percentage": round(100 * nb_sorted / total) if total > 0 else 0,
            "details":    self.sorted_shapes.copy()
        }

    def get_summary(self):
        """Résumé complet de l'épisode."""
        return {
            "state":        self.current_state,
            "step_count":   self.step_count,
            "total_reward": round(self.total_reward, 3),
            "progress":     self.get_progress(),
            "history":      self.history
        }


if __name__ == "__main__":
    from src.scene.scene_builder import SceneBuilder

    print("=== Test ShapeSortingTask ===")

    task:  ShapeSortingTask = ShapeSortingTask()
    scene: SceneBuilder     = SceneBuilder()
    scene.build()

    # Reset
    obs = task.reset()
    print(f"\nObservation initiale : {len(obs)} valeurs")
    print(f"  {obs}")

    # Simule quelques pas
    print("\n--- Simulation de 5 pas ---")
    for i in range(5):
        action = np.array([0.01, 0.0, 0.0, 1.0])
        obs, reward, done, info = task.step(action, scene)
        print(
            f"  Pas {i+1} : état={info['state']:10s} "
            f"reward={reward:.3f} "
            f"triées={info['nb_sorted']}/3"
        )
        if done:
            break

    # Simule une saisie réussie
    print("\n--- Simulation saisie + dépôt ---")
    task.set_state(task.STATE_GRASPING)
    action_grasp = np.array([0.0, 0.0, 0.0, 0.0])  # ferme pince
    obs, reward, done, info = task.step(action_grasp, scene)
    print(f"  Saisie : reward={reward:.3f} tient={info['is_holding']}")

    task.set_state(task.STATE_RELEASING)
    action_release = np.array([0.0, 0.0, 0.0, 1.0])  # ouvre pince
    obs, reward, done, info = task.step(action_release, scene)
    print(f"  Dépôt  : reward={reward:.3f}")

    # Progression
    print("\n--- Progression ---")
    progress = task.get_progress()
    print(f"  {progress['sorted']}/{progress['total']} "
            f"({progress['percentage']}%)")

    # Résumé final
    print("\n--- Résumé ---")
    summary = task.get_summary()
    print(f"  Pas effectués : {summary['step_count']}")
    print(f"  Récompense    : {summary['total_reward']}")
    print(f"  Progression   : {summary['progress']['sorted']}/{summary['progress']['total']}")
    print("\n=== Test terminé ===")