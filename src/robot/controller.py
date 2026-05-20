# src/robot/controller.py
# Contrôleur du bras Franka Panda dans Isaac Sim
# Ce fichier gère l'envoi de commandes aux 9 articulations du robot

import numpy as np
import yaml
import os

# On essaie d'importer Isaac Sim
# Si on est en dehors d'Isaac Sim (ex: test unitaire), on continue quand même
try:
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.api.utils.types import ArticulationAction
    ISAAC_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.types import ArticulationAction
        ISAAC_AVAILABLE = True
    except ImportError:
        ISAAC_AVAILABLE = False
        print("[Controller] Isaac Sim non disponible — mode test seul")

from src.utils.math_utils import clamp, deg_to_rad, rad_to_deg


class PandaController:
    """
    Contrôleur principal du bras Franka Panda.

    Le Panda a 9 DOF :
    - 7 articulations du bras (joints 1 à 7)
    - 2 doigts de la pince (finger_joint1, finger_joint2)

    Deux modes de contrôle disponibles :
    - POSITION : on donne l'angle cible, le moteur y va tout seul
    - EFFORT   : on donne une force/couple directement au moteur
    """

    # Constantes de mode
    MODE_POSITION = "position"
    MODE_EFFORT   = "effort"

    def __init__(self, config_path="config/robot.yaml"):
        """
        config_path : chemin vers le fichier robot.yaml
        """
        self.config = self._load_config(config_path)
        self.robot  = None   # sera rempli quand Isaac Sim sera actif
        self.mode   = self.MODE_POSITION

        # Récupère les paramètres depuis le YAML
        robot_cfg = self.config["robot"]
        self.joint_names  = robot_cfg["joint_names"]   # liste des 9 noms
        self.num_dof      = robot_cfg["dof"]           # 9
        self.stiffness    = np.array(robot_cfg["control"]["stiffness"])
        self.damping      = np.array(robot_cfg["control"]["damping"])
        self.max_effort   = np.array(robot_cfg["control"]["max_effort"])

        # Position de départ par défaut (en radians)
        self.default_positions = np.array(
            robot_cfg["default_joint_positions"]
        )

        # Limites articulaires — shape (9, 2) : [[min, max], ...]
        limits = robot_cfg["joint_limits"]
        self.joint_limits = np.array([
            limits[name] for name in self.joint_names
        ])

        # État interne — mis à jour à chaque step
        self.current_positions  = self.default_positions.copy()
        self.current_velocities = np.zeros(self.num_dof)
        self.current_efforts    = np.zeros(self.num_dof)

        print(f"[Controller] Panda initialisé — {self.num_dof} DOF")
        print(f"[Controller] Mode : {self.mode}")

    # ─────────────────────────────────────────────
    # CHARGEMENT CONFIG
    # ─────────────────────────────────────────────

    def _load_config(self, config_path):
        """Charge le fichier YAML de configuration."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config introuvable : {config_path}\n"
                f"Lance ce script depuis la racine du projet Demo2/"
            )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # ─────────────────────────────────────────────
    # CONNEXION AU ROBOT ISAAC SIM
    # ─────────────────────────────────────────────

    def set_robot(self, robot):
        """
        Connecte ce contrôleur à un objet Robot d'Isaac Sim.
        À appeler après que la simulation a été initialisée.

        robot : instance de omni.isaac.core.robots.Robot
        """
        self.robot = robot
        print(f"[Controller] Robot connecté : {robot.name}")

    # ─────────────────────────────────────────────
    # COMMANDES DE POSITION
    # ─────────────────────────────────────────────

    def set_joint_positions(self, positions):
        """
        Envoie des positions cibles à toutes les articulations.

        positions : array de 9 valeurs en radians
                    (7 bras + 2 doigts pince)

        La commande est d'abord vérifiée contre les limites
        articulaires pour éviter d'endommager le robot.
        """
        positions = np.array(positions, dtype=np.float32)

        if len(positions) != self.num_dof:
            raise ValueError(
                f"Attendu {self.num_dof} positions, reçu {len(positions)}"
            )

        # Applique les limites articulaires
        positions_clipped = self._apply_joint_limits(positions)

        if ISAAC_AVAILABLE and self.robot is not None:
            # Envoie la commande à Isaac Sim
            action = ArticulationAction(
                joint_positions=positions_clipped
            )
            self.robot.apply_action(action)

        # Mémorise la commande envoyée
        self.current_positions = positions_clipped
        return positions_clipped

    def set_arm_positions(self, arm_positions):
        """
        Commande uniquement les 7 articulations du bras.
        La pince reste à sa position actuelle.

        arm_positions : array de 7 valeurs en radians
        """
        if len(arm_positions) != 7:
            raise ValueError(
                f"Attendu 7 positions pour le bras, reçu {len(arm_positions)}"
            )

        # Combine bras + pince actuelle
        full_positions = np.concatenate([
            arm_positions,
            self.current_positions[7:]  # conserve les 2 doigts
        ])
        return self.set_joint_positions(full_positions)

    def open_gripper(self):
        """
        Ouvre complètement la pince.
        Largeur maximale = 0.04m par doigt (8cm total).
        """
        gripper_open = np.array([0.04, 0.04])
        full_positions = np.concatenate([
            self.current_positions[:7],  # conserve le bras
            gripper_open
        ])
        print("[Controller] Pince ouverte")
        return self.set_joint_positions(full_positions)

    def close_gripper(self, width=0.0):
        """
        Ferme la pince à une largeur donnée.

        width : largeur de fermeture par doigt en mètres
                0.0 = complètement fermée
                0.04 = complètement ouverte
        """
        width = float(np.clip(width, 0.0, 0.04))
        gripper_closed = np.array([width, width])
        full_positions = np.concatenate([
            self.current_positions[:7],
            gripper_closed
        ])
        print(f"[Controller] Pince fermée à {width*100:.1f}mm")
        return self.set_joint_positions(full_positions)

    def go_to_default_position(self):
        """
        Remet le robot à sa position de départ définie dans robot.yaml.
        Utilisé au début de chaque épisode d'entraînement.
        """
        print("[Controller] Retour à la position par défaut")
        return self.set_joint_positions(self.default_positions)

    # ─────────────────────────────────────────────
    # COMMANDES D'EFFORT (FORCE/COUPLE)
    # ─────────────────────────────────────────────

    def set_joint_efforts(self, efforts):
        """
        Envoie des couples (forces de rotation) directement aux moteurs.

        efforts : array de 9 valeurs en Newton.mètre (N.m)
                  Limité par max_effort défini dans robot.yaml

        Mode avancé — utilisé pour le contrôle de force précis
        (ex: saisir un objet avec une force contrôlée).
        """
        efforts = np.array(efforts, dtype=np.float32)

        if len(efforts) != self.num_dof:
            raise ValueError(
                f"Attendu {self.num_dof} efforts, reçu {len(efforts)}"
            )

        # Limite les efforts aux valeurs maximales du moteur
        efforts_clipped = np.clip(efforts, -self.max_effort, self.max_effort)

        if ISAAC_AVAILABLE and self.robot is not None:
            action = ArticulationAction(
                joint_efforts=efforts_clipped
            )
            self.robot.apply_action(action)

        self.current_efforts = efforts_clipped
        return efforts_clipped

    # ─────────────────────────────────────────────
    # LECTURE DE L'ÉTAT DU ROBOT
    # ─────────────────────────────────────────────

    def get_joint_positions(self):
        """
        Retourne les angles actuels de toutes les articulations.
        En simulation : lit les valeurs depuis Isaac Sim.
        En test : retourne les dernières valeurs commandées.
        """
        if ISAAC_AVAILABLE and self.robot is not None:
            state = self.robot.get_joints_state()
            self.current_positions = state.positions
        return self.current_positions.copy()

    def get_joint_velocities(self):
        """
        Retourne les vitesses angulaires actuelles en rad/s.
        """
        if ISAAC_AVAILABLE and self.robot is not None:
            state = self.robot.get_joints_state()
            self.current_velocities = state.velocities
        return self.current_velocities.copy()

    def get_end_effector_pose(self):
        """
        Retourne la pose (position + orientation) du bout du bras.
        C'est le point de référence pour la cinématique inverse.

        Retourne un dict :
        {
            "position"    : array [x, y, z] en mètres,
            "orientation" : quaternion [x, y, z, w]
        }
        """
        if ISAAC_AVAILABLE and self.robot is not None:
            # Le prim_path de la main du Panda dans le USD
            hand_prim_path = self.robot.prim_path + "/panda_hand"
            try:
                from omni.isaac.core.utils.stage import get_current_stage
                import omni.usd
                stage = get_current_stage()
                prim  = stage.GetPrimAtPath(hand_prim_path)
                # Récupère la transformation mondiale
                from pxr import UsdGeom
                xform = UsdGeom.Xformable(prim)
                transform = xform.ComputeLocalToWorldTransform(0)
                position = np.array(transform.ExtractTranslation())
                # Quaternion depuis la matrice de rotation
                rotation = transform.ExtractRotationQuat()
                quat = np.array([
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal()
                ])
                return {"position": position, "orientation": quat}
            except Exception as e:
                print(f"[Controller] Erreur get_end_effector_pose : {e}")

        # Valeur par défaut si Isaac n'est pas disponible
        return {
            "position":    np.array([0.5, 0.0, 0.5]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0])
        }

    # ─────────────────────────────────────────────
    # UTILITAIRES INTERNES
    # ─────────────────────────────────────────────

    def _apply_joint_limits(self, positions):
        """
        Applique les limites articulaires à un array de positions.
        Chaque valeur est bloquée entre [min, max] pour ce joint.
        """
        clipped = positions.copy()
        for i in range(self.num_dof):
            clipped[i] = np.clip(
                positions[i],
                self.joint_limits[i, 0],
                self.joint_limits[i, 1]
            )
            if clipped[i] != positions[i]:
                print(
                    f"[Controller] Joint {self.joint_names[i]} : "
                    f"{rad_to_deg(positions[i]):.1f}° hors limites → "
                    f"bloqué à {rad_to_deg(clipped[i]):.1f}°"
                )
        return clipped

    def get_status(self):
        """
        Retourne un résumé de l'état actuel du robot.
        Utile pour le débogage.
        """
        positions_deg = rad_to_deg(self.current_positions)
        status = {
            "mode":           self.mode,
            "isaac_available": ISAAC_AVAILABLE,
            "robot_connected": self.robot is not None,
            "joint_positions_deg": {
                name: round(float(pos), 2)
                for name, pos in zip(self.joint_names, positions_deg)
            },
            "gripper_width_mm": round(
                float(self.current_positions[7] + self.current_positions[8]) * 1000, 1
            )
        }
        return status


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test PandaController ===")

    ctrl = PandaController(config_path="config/robot.yaml")

    # Test position par défaut
    print("\n--- Position par défaut ---")
    ctrl.go_to_default_position()
    positions = ctrl.get_joint_positions()
    print(f"Positions (rad) : {positions.round(3)}")
    print(f"Positions (deg) : {rad_to_deg(positions).round(1)}")

    # Test ouverture/fermeture pince
    print("\n--- Test pince ---")
    ctrl.open_gripper()
    print(f"Pince ouverte : {ctrl.current_positions[7]*1000:.1f}mm par doigt")

    ctrl.close_gripper(width=0.02)
    print(f"Pince mi-fermée : {ctrl.current_positions[7]*1000:.1f}mm par doigt")

    ctrl.close_gripper(width=0.0)
    print(f"Pince fermée : {ctrl.current_positions[7]*1000:.1f}mm par doigt")

    # Test limites articulaires
    print("\n--- Test limites articulaires ---")
    positions_hors_limites = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0])
    ctrl.set_joint_positions(positions_hors_limites)

    # Test status
    print("\n--- Status ---")
    status = ctrl.get_status()
    for key, value in status.items():
        if key != "joint_positions_deg":
            print(f"  {key} : {value}")
    print("  Positions articulations :")
    for name, deg in status["joint_positions_deg"].items():
        print(f"    {name} : {deg}°")

    print("\n=== Test terminé ✓ ===")