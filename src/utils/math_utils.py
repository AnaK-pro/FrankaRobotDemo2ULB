#src/utils/math_utils.py
#Fonctions mathematiques pour le robot Franka Panda : cinematique directe, inverse, jacobienne, etc.

import numpy as np

#quaternions
#un quaternions represente une rotation dans l'espace 3D. Il est compose de 4 elements : (x, y, z, w) ou (qx, qy, qz, qw)
#format : (x, y, z, w) ou (qx, qy, qz, qw)
#plus stable que les angles d'Euler pour representer les rotations, evite les singularites (gimbal lock) et permet des interpolations plus fluides entre les orientations. Utilise dans la robotique pour representer l'orientation des robots et des capteurs.

def quaternions_to_Euler(q):
    #convertit un quaternion en angles d'Euler (roll, pitch, yaw)
    x, y, z, w = q
    #roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    #pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1) #clamp pour eviter les erreurs de domaine
    pitch = np.arcsin(sinp)
    
    #yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def Euler_to_quaternions(roll, pitch, yaw):
    #convertit des angles d'Euler (roll, pitch, yaw) en quaternion (x, y, z, w)
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
    #multiplie deux quaternions q1 et q2
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

def quaternions_normalize(q):
    #normalise un quaternion pour qu'il ait une norme de 1
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return q
    return q / norm

#matrice de tranformation homogène
#une matrice de transformation homogène est une matrice 4x4 qui combine une rotation et une translation dans l'espace 3D. Elle est utilisee pour representer la position
#et l'orientation d'un objet dans l'espace. La partie superieure gauche 3x3 de la matrice represente la rotation, tandis que la partie superieure droite 3x1 represente la translation. La derniere ligne est toujours [0, 0, 0, 1].


def transformation_matrix(position, quaternion):
#cree une matrice de transformation homogène a partir d'une position (x, y, z) et d'une orientation en quaternion (qx, qy, qz, qw)
    T = np.eye(4) #matrice identité 4x4
    x, y, z, w = quaternion 
    #calcul de la matrice de rotation a partir du quaternion
    #matrice de rotation depuis le quaternion
    T[0,0] = 1 - 2 * (y * y + z * z)
    T[0,1] = 2 * (x * y - z * w)
    T[0,2] = 2 * (x * z + y * w)
    T[1,0] = 2 * (x * y + z * w)
    T[1,1] = 1 - 2 * (x * x + z * z)
    T[1,2] = 2 * (y * z - x * w)
    T[2,0] = 2 * (x * z - y * w)
    T[2,1] = 2 * (y * z + x * w)
    T[2,2] = 1 - 2 * (x * x + y * y)
    #ajout de la translation
    T[0,3] = position[0]
    T[1,3] = position[1]
    T[2,3] = position[2]
    return T


def invert_transform(T):
    #inverse une matrice de transformation homogène T
    R = T[0:3, 0:3] #rotation
    t = T[0:3, 3]   #translation
    R_inv = R.T     #inverse de la rotation (transpose)
    t_inv = -R_inv @ t #inverse de la translation
    T_inv = np.eye(4) #matrice identité 4x4
    T_inv[0:3, 0:3] = R_inv #ajout de la rotation inverse
    T_inv[0:3, 3] = t_inv #ajout de la translation inverse
    return T_inv

def distance_2d(pos1, pos2):
    #calcule la distance euclidienne entre deux positions 2D (x, y)
    p1 = np.array(pos1[:2])
    p2 = np.array(pos2[:2])
    return np.linalg.norm(p1 - p2)

def normalize_vector(v):
    #normalise un vecteur pour qu'il ait une norme de 1
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm

def angle_between_vectors(v1, v2):
    #calcule l'angle entre deux vecteurs v1 et v2 en radians
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0) #clamp pour eviter les erreurs de domaine
    return np.arccos(dot_product)

def deg_to_rad(degree):
    #convertit des degrés en radians
    return degree * np.pi / 180.0

def rad_to_deg(radian):
    #convertit des radians en degrés
    return radian * 180.0 / np.pi

def clamp(value, min_value, max_value):
    #clamp une valeur entre un minimum et un maximum
    return np.clip(value, min_value, max_value)

if __name__ == "__main__":
    print("Test des fonctions de math_utils.py")
    #testquternion to Euler et inverse
    q_original = np.array([0.0, 0.0, 0.3827, 0.9239]) #rotation de 45 degrés autour de l'axe z
    euler = quaternions_to_Euler(q_original)
    print(f'quaternion to Euler: roll={rad_to_deg(euler[0]):.1f}°, pitch={rad_to_deg(euler[1]):.1f}°, yaw={rad_to_deg(euler[2]):.1f}°')
    
    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 1.0, 0.0]
    print("Distance 2D entre p1 et p2:", distance_2d(p1, p2))
    
    #test matrice de transformation
    pos = [0.5, 0.0, 0.3]
    T = transformation_matrix(pos, np.array([0.0, 0.0, 0.0, 1.0])) #pas de rotation, juste une translation
    print("Matrice de transformation T:\n", T)
    
    
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    print("Angle entre v1 et v2:", rad_to_deg(angle_between_vectors(v1, v2)))
    
    print('===Tout est OK dans math_utils.py===')