import numpy as np
from copy import copy
from pyquaternion import Quaternion
import scipy.linalg as scl

pi = np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T

"""
def fkine_kr20(q):
    
    Calcular la cinematica directa del robot KUKA KR20 dados sus valores articulares. 
    q es un vector numpy de la forma [q0, q1, q2, q3, q4, q5, q6]
    

    # Longitudes (en metros)
    L0 = 0.0
    L1 = 0.520
    L2 = 0.780
    L3 = 0.150
    L4 = 0.86
    L5 = 0
    L6 = 0.153

    # Matrices DH (completar)
    T0 = dh(L0, q[0],     0, 0) #
    T1 = dh(L1, q[1],     0.160, pi/2) #
    T2 = dh(0 , -q[2]+pi/2,      L2,0) #
    T3 = dh(0 , q[3]+pi     ,-L3  ,-pi/2) #
    T4 = dh(L4  , q[4]     ,0    ,pi/2)
    T5 = dh(0  , q[5]+pi     ,0    ,-pi/2)
    T6 = dh(-L6-q[6]  ,  -pi/2    ,0   ,-pi/2)
    # Efector final con respecto a la base
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6  
    return T
"""
def fkine_kr20(q):
    """
    Calcular la cinematica directa del robot KUKA KR20 dados sus valores articulares. 
    q es un vector numpy de las primeras 6 articulaciones
    """
    # Longitudes (en metros)
    L0 = 0.0
    L1 = 0.520
    L2 = 0.780
    L3 = 0.150
    L4 = 0.86
    L5 = 0
    L6 = 0.153

    # Matrices DH
    T0 = dh(L0, q[0],     0, 0)
    T1 = dh(L1, q[1],     0.160, pi/2)
    T2 = dh(0 , -q[2]+pi/2,      L2,0)
    T3 = dh(0 , q[3]+pi     ,-L3  ,-pi/2)
    T4 = dh(L4  , q[4]     ,0    ,pi/2)
    T5 = dh(0  , q[5]+pi     ,0    ,-pi/2)
    T6 = dh(-L6  , 0   ,0   ,0)  # La pala en posición neutral
    
    # Efector final con respecto a la base
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6
    return T

def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analítico para la posición. Retorna una matriz de 3x8
    """
    # Alocación de memoria
    J = np.zeros((3,8))
    # Transformación homogénea inicial
    T = fkine_kr20(q[:7])
    
    # Iteración para cada articulación
    for i in range(8):
        # Copia de configuración articular
        dq = copy(q)
        # Incremento en la articulación i
        dq[i] = dq[i] + delta
        # Transformación homogénea con el incremento
        dT = fkine_kr20(dq[:7])
        # Aproximación del Jacobiano
        J[:,i] = (dT[0:3,3] - T[0:3,3])/delta
    
    return J
    
def ikine(xdes, q0):
    """
    Calcular la cinematica inversa numéricamente
    xdes: posición deseada del efector final [x, y, z]
    q0: configuración inicial/semilla [q1, q2, q3, q4, q5, q6, q7, q8]
    """
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    q = copy(q0)

    for i in range(max_iter):
        # Calcular Jacobiano actual
        J = jacobian_position(q, delta)
        # Obtener posición actual (usando los primeros 7 joints)
        T = fkine_kr20(q[:7])
        x = T[0:3,3]
        # Error
        e = xdes - x
        # Actualizar q
        q = q + np.dot(J.T, e)
        
        # Verificar convergencia
        if np.linalg.norm(e) < epsilon:
            break
            
    return q

def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analítico para la posición y orientación usando cuaterniones.
    """
    J = np.zeros((7,6))
    
    # Transformación inicial
    T = fkine_kr20(q)
    x = TF2xyzquat(T)
    
    # Calcular Jacobiano numéricamente
    for i in range(6):
        # Configuración con incremento
        dq = copy(q)
        dq[i] = dq[i] + delta
        
        # Nueva transformación
        T_inc = fkine_kr20(dq)
        x_inc = TF2xyzquat(T_inc)
        
        # Aproximación del Jacobiano
        J[:,i] = (x_inc - x)/delta
        
    return J

def TF2xyzquat(T):
    """
    Convertir matriz de transformación homogénea a vector [x y z qw qx qy qz]
    """
    quat = Quaternion(matrix=T[0:3,0:3])
    return np.array([T[0,3], T[1,3], T[2,3], quat.w, quat.x, quat.y, quat.z])

def PoseError(x, xd):
    """
    Calcular el error de pose entre la pose actual y la deseada
    """
    # Error de posición
    pos_err = x[0:3] - xd[0:3]
    
    # Error de orientación usando cuaterniones
    qact = Quaternion(x[3:7])
    qdes = Quaternion(xd[3:7])
    qdif = qdes * qact.inverse
    qua_err = np.array([qdif.w, qdif.x, qdif.y, qdif.z])
    
    # Combinar errores
    return np.hstack((pos_err, qua_err))



def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R

def kr20_fkine_debug(q):
    print("\nPosiciones intermedias:")
    
    # Base a primera articulación (rotación)
    T0 = dh(0.15,    q[0],   0,      0)    
    print("T0 (base):", np.round(T0[0:3,3], 4))
    
    # Primera a segunda articulación
    T1 = T0 @ dh(0.15,    q[1],   0.16,   -np.pi/2)
    print("T1:", np.round(T1[0:3,3], 4))
    
    # Segunda a tercera articulación
    T2 = T1 @ dh(0,       q[2],   0.78,   0)
    print("T2:", np.round(T2[0:3,3], 4))
    
    # Tercera a cuarta articulación
    T3 = T2 @ dh(0,       q[3],   0.15,   np.pi/2)
    print("T3:", np.round(T3[0:3,3], 4))
    
    # Cuarta a quinta articulación
    T4 = T3 @ dh(0.70,    q[4],   0,      np.pi/2)
    print("T4:", np.round(T4[0:3,3], 4))
    
    # Quinta a sexta articulación
    T5 = T4 @ dh(0,       q[5],   0.155,  np.pi/2)
    print("T5:", np.round(T5[0:3,3], 4))
    
    # Sexta al efector final (prismático)
    T6 = T5 @ dh(0.115,   0,      q[6],   0)
    print("T6 (final):", np.round(T6[0:3,3], 4))
    
    return T6

def rotmat2euler(R):
    """
    Convierte una matriz de rotación a ángulos de Euler ZYX
    """
    if R[2,0] > 0.998:
        phi = 0
        theta = -np.pi/2
        psi = np.arctan2(-R[0,1], R[1,1])
    elif R[2,0] < -0.998:
        phi = 0
        theta = np.pi/2
        psi = np.arctan2(R[0,1], -R[1,1])
    else:
        phi = np.arctan2(R[1,0], R[0,0])
        theta = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        psi = np.arctan2(R[2,1], R[2,2])
    return np.array([phi, theta, psi])

def euler2rotmat(angles):
    """
    Convierte ángulos de Euler ZYX a matriz de rotación
    """
    phi, theta, psi = angles
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi),  np.cos(phi), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi), np.cos(psi)]])
    
    return Rz @ Ry @ Rx

def spatial_jacobian(q):
    """
    Calcula el Jacobiano espacial del robot
    """
    J = np.zeros((6,6))
    T = fkine_kr20(q)
    p = T[0:3,3]
    
    # Calcular cada columna del Jacobiano
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = 0.0001  # pequeño incremento
        
        # Nueva transformación
        Td = fkine_kr20(q + dq)
        pd = Td[0:3,3]
        Rd = Td[0:3,0:3]
        R = T[0:3,0:3]
        
        # Velocidad lineal
        J[0:3,i] = (pd - p)/0.0001
        
        # Velocidad angular
        dR = (Rd - R)/0.0001
        J[3:6,i] = np.array([dR[2,1] - dR[1,2],
                            dR[0,2] - dR[2,0],
                            dR[1,0] - dR[0,1]])/2
    return J

def analytic_to_spatial(J_a, R):
    """
    Convierte el Jacobiano analítico a espacial
    """
    T = np.zeros((6,6))
    T[0:3,0:3] = R
    T[3:6,3:6] = R
    return T @ J_a

def pose_error(T, Td):
    """
    Calcula el error de pose entre la pose actual y la deseada
    """
    # Error de posición
    ep = Td[0:3,3] - T[0:3,3]
    
    # Error de orientación
    Re = Td[0:3,0:3] @ T[0:3,0:3].T
    eo = np.array([Re[2,1] - Re[1,2],
                   Re[0,2] - Re[2,0],
                   Re[1,0] - Re[0,1]]) / 2
    
    return np.concatenate([ep, eo])

def get_joint_limits():
    """
    Retorna los límites articulares del robot
    """
    q_min = np.array([-185, -145, -120, -350, -125, -350]) * np.pi/180
    q_max = np.array([185, 0, 155, 350, 125, 350]) * np.pi/180
    return q_min, q_max