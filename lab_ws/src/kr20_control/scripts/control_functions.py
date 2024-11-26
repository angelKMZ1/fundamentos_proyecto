#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

def dh(d, theta, a, alpha):
    """
    Matriz de transformación de Denavit-Hartenberg.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    H = np.array([[ct, -st*ca, st*sa, a*ct],
                  [st, ct*ca, -ct*sa, a*st],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return H

def kr20_fk(q):
    """
    Cinemática directa del KR20.
    Input: configuración articular
    Output: matriz de transformación homogénea
    """
    # Parámetros DH del KR20
    d1 = 0.0
    d2 = 0.520
    d3 = 0.0
    d4 = 0.0
    d5 = 0.86
    d6 = 0.0
    
    a1 = 0.0
    a2 = 0.160
    a3 = 0.780
    a4 = -0.150
    a5 = 0.0
    a6 = 0.0
    
    alp1 = 0
    alp2 = np.pi/2
    alp3 = 0
    alp4 = -np.pi/2
    alp5 = np.pi/2
    alp6 = -np.pi/2
    
    # Matrices de transformación
    T1 = dh(d1, q[0], a1, alp1)
    T2 = dh(d2, q[1], a2, alp2)
    T3 = dh(d3, q[2], a3, alp3)
    T4 = dh(d4, q[3], a4, alp4)
    T5 = dh(d5, q[4], a5, alp5)
    T6 = dh(d6, q[5], a6, alp6)
    
    # Transformación total
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6
    return T

def geometric_jacobian(q, delta=0.0001):
    """
    Calcula el Jacobiano geométrico.
    """
    J = np.zeros((6, 6))
    T = kr20_fk(q)
    p = T[0:3, 3]
    R = T[0:3, 0:3]
    
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = delta
        
        Td = kr20_fk(q + dq)
        pd = Td[0:3, 3]
        Rd = Td[0:3, 0:3]
        
        # Velocidad lineal
        J[0:3, i] = (pd - p) / delta
        
        # Velocidad angular
        dR = (Rd - R) / delta
        S = dR @ R.T
        J[3:6, i] = np.array([S[2,1], S[0,2], S[1,0]])
    
    return J

def compute_pose_error(T_current, T_desired):
    """
    Calcula el error de pose (posición y orientación).
    """
    # Error de posición
    p_error = T_desired[0:3, 3] - T_current[0:3, 3]
    
    # Error de orientación
    R_error = T_desired[0:3, 0:3] @ T_current[0:3, 0:3].T
    r = R.from_matrix(R_error)
    angle_axis = r.as_rotvec()
    
    return np.concatenate([p_error, angle_axis])

def normalize_joints(q):
    """
    Normaliza los ángulos al rango [-pi, pi].
    """
    return np.mod(q + np.pi, 2*np.pi) - np.pi
