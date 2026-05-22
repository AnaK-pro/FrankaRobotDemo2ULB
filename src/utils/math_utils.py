# src/utils/math_utils.py
# Fonctions mathematiques pour le robot Franka Panda

import numpy as np


# ─── Quaternions ─────────────────────────────────────────────────────────────

def quaternions_to_Euler(q):
    # convertit un quaternion (x, y, z, w) en angles d'Euler (roll, pitch, yaw)
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2 * (w * y - z * x), -1, 1)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def Euler_to_quaternions(roll, pitch, yaw):
    # convertit des angles d'Euler (roll, pitch, yaw) en quaternion (x, y, z, w)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])


def quaternions_multiply(q1, q2):
    # multiplie deux quaternions q1 et q2
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


def quaternions_normalize(q):
    # normalise un quaternion pour qu'il ait une norme de 1
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return q
    return q / norm


# ─── Matrices de transformation ───────────────────────────────────────────────

def make_transform_matrix(position, quaternion):
    # cree une matrice de transformation homogene 4x4 (rotation + translation)
    T = np.eye(4)
    x, y, z, w = quaternion
    T[0, 0] = 1 - 2 * (y*y + z*z)
    T[0, 1] = 2 * (x*y - w*z)
    T[0, 2] = 2 * (x*z + w*y)
    T[1, 0] = 2 * (x*y + w*z)
    T[1, 1] = 1 - 2 * (x*x + z*z)
    T[1, 2] = 2 * (y*z - w*x)
    T[2, 0] = 2 * (x*z - w*y)
    T[2, 1] = 2 * (y*z + w*x)
    T[2, 2] = 1 - 2 * (x*x + y*y)
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]
    return T


def invert_transform(T):
    # inverse une matrice de transformation homogene
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3,  3]
    T_inv[:3, :3] = R.T
    T_inv[:3,  3] = -R.T @ t
    return T_inv


# ─── Distances ────────────────────────────────────────────────────────────────

def distance_3d(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def distance_2d(pos1, pos2):
    p1 = np.array(pos1[:2])
    p2 = np.array(pos2[:2])
    return np.linalg.norm(p1 - p2)


# ─── Vecteurs ─────────────────────────────────────────────────────────────────

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def angle_between_vectors(v1, v2):
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)


# ─── Conversion angles ────────────────────────────────────────────────────────

def deg_to_rad(degree):
    return degree * np.pi / 180.0


def rad_to_deg(radian):
    return radian * 180.0 / np.pi


def clamp(value, min_value, max_value):
    return np.clip(value, min_value, max_value)


# ─── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test des fonctions de math_utils.py")

    q_original = np.array([0.0, 0.0, 0.3827, 0.9239])
    euler = quaternions_to_Euler(q_original)
    print(f"quaternion to Euler: roll={rad_to_deg(euler[0]):.1f}°, pitch={rad_to_deg(euler[1]):.1f}°, yaw={rad_to_deg(euler[2]):.1f}°")

    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 1.0, 0.0]
    print("Distance 2D entre p1 et p2:", distance_2d(p1, p2))

    pos = [0.5, 0.0, 0.3]
    T = make_transform_matrix(pos, np.array([0.0, 0.0, 0.0, 1.0]))
    print("Matrice de transformation T:\n", T)

    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    print("Angle entre v1 et v2:", rad_to_deg(angle_between_vectors(v1, v2)))

    print("=== Tout est OK dans math_utils.py ===")
