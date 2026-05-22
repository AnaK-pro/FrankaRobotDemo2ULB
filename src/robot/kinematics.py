# src/robot/kinematics.py
# Cinématique du bras Franka Panda - Version corrigée et stabilisée

import numpy as np
from src.utils.math_utils import distance_3d, quaternions_normalize


# Paramètres Denavit-Hartenberg du Franka Panda
PANDA_DH_PARAMS = np.array([
    [0.0,    0.333,   0.0,       0.0],      # joint 1
    [0.0,    0.0,    -np.pi/2,   0.0],      # joint 2
    [0.0,    0.316,   np.pi/2,   0.0],      # joint 3
    [0.0825, 0.0,     np.pi/2,   0.0],      # joint 4
    [-0.0825,0.384,  -np.pi/2,   0.0],      # joint 5
    [0.0,    0.0,     np.pi/2,   0.0],      # joint 6
    [0.088,  0.107,   np.pi/2,   0.0],      # joint 7
], dtype=np.float32)


class PandaKinematics:
    """
    Cinématique Forward + Inverse pour Franka Panda
    Version améliorée pour Isaac Sim
    """

    def __init__(self):
        self.dh_params = PANDA_DH_PARAMS
        self.num_joints = 7
        print("[Kinematics] Panda DH initialisé — IK améliorée ✓")

    def _dh_matrix(self, a, d, alpha, theta):
        """Matrice de transformation DH"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0, sa,       ca,      d],
            [0.0, 0.0,      0.0,     1.0]
        ], dtype=np.float32)

    def forward_kinematics(self, joint_angles):
        """Calcul de la position et orientation du TCP"""
        angles = np.array(joint_angles[:7], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)

        for i in range(self.num_joints):
            a, d, alpha, _ = self.dh_params[i]
            theta = angles[i]
            T_i = self._dh_matrix(a, d, alpha, theta)
            T = T @ T_i

        position = T[:3, 3]
        quaternion = self._rotation_matrix_to_quaternion(T[:3, :3])

        return {
            "position": position,
            "orientation": quaternion,
            "transform": T
        }

    def inverse_kinematics(
        self,
        target_position,
        initial_angles=None,
        max_iterations=80,
        tolerance=0.018,
        learning_rate=0.08   # Très réduit pour éviter les oscillations
    ):
        target_pos = np.array(target_position, dtype=np.float32)

        if initial_angles is None:
            angles = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
        else:
            angles = np.array(initial_angles[:7], dtype=np.float32).copy()

        joint_limits = np.array([
            [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
            [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]
        ])

        damping = 0.25  # Damping fort pour stabiliser

        for it in range(max_iterations):
            current = self.forward_kinematics(angles)
            error = target_pos - current["position"]
            dist = np.linalg.norm(error)

            if dist < tolerance:
                return {"success": True, "angles": angles, "error": dist, "iterations": it+1}

            J = self._compute_jacobian(angles)
            JJT = J @ J.T
            reg = (damping ** 2) * np.eye(3)
            
            try:
                delta = J.T @ np.linalg.solve(JJT + reg, error)
            except:
                delta = J.T @ error * 0.05

            # Mise à jour très douce + lissage
            step = learning_rate * np.clip(delta, -0.08, 0.08)
            angles = angles + step

            # Limites strictes
            angles = np.clip(angles, joint_limits[:,0], joint_limits[:,1])

        final_error = np.linalg.norm(target_pos - self.forward_kinematics(angles)["position"])
        return {"success": False, "angles": angles, "error": final_error, "iterations": max_iterations}
    def _compute_jacobian(self, angles, delta=1e-5):
        """Jacobienne numérique 3x7"""
        J = np.zeros((3, self.num_joints), dtype=np.float32)
        ref_pos = self.forward_kinematics(angles)["position"]

        for i in range(self.num_joints):
            angles_pert = angles.copy()
            angles_pert[i] += delta
            new_pos = self.forward_kinematics(angles_pert)["position"]
            J[:, i] = (new_pos - ref_pos) / delta

        # Correction importante pour le repère Isaac Sim (axe Y inversé)
        J[1, :] = -J[1, :]

        return J

    def _rotation_matrix_to_quaternion(self, R):
        """Conversion matrice de rotation → quaternion"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 2] + R[2, 0]) / s
            z = (R[0, 1] + R[1, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return quaternions_normalize(np.array([x, y, z, w], dtype=np.float32))


# Test rapide
if __name__ == "__main__":
    kin = PandaKinematics()
    pos = kin.forward_kinematics([0, -0.785, 0, -2.356, 0, 1.571, 0.785])["position"]
    print("Position TCP par défaut :", pos.round(4))
    print("=== Kinematics.py chargé avec succès ===")