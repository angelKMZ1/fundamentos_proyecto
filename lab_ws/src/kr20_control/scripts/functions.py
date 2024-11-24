import numpy as np
from copy import copy

pi = np.pi

def dh(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada con los parametros DH.
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

def kr20_fkine(q):
    """
    Calcular la cinematica directa del robot KUKA KR20 dados sus valores articulares. 
    """
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
    
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6
    return T

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.dt = dt
        self.ndof = ndof
        
    def gravity_compensation(self, q):
        """
        Calcula el vector de compensación de gravedad (simplificado)
        """
        g = 9.81
        gravity = np.zeros(self.ndof)
        # Aproximación simple de la compensación de gravedad
        gravity[1] = g * np.cos(q[1])  # Compensación para el hombro
        gravity[2] = g * np.cos(q[2])  # Compensación para el codo
        return gravity
        
    def send_command(self, tau):
        """
        Implementación simplificada de la dinámica del robot
        """
        # Matriz de inercia simplificada (diagonal)
        M = np.diag([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.1, 0.1])
        
        # Compensación de gravedad
        g = self.gravity_compensation(self.q)
        
        # Ecuación dinámica simplificada: M * ddq = tau - g
        ddq = np.linalg.solve(M, tau - g)
        
        # Integración numérica
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq
        
        # Límites articulares simplificados
        self.q = np.clip(self.q, -np.pi, np.pi)
        
    def read_joint_positions(self):
        return self.q
        
    def read_joint_velocities(self):
        return self.dq
