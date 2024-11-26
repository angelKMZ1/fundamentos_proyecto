import rospy
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def dh(d, theta, a, alpha):
    """Matriz de transformación DH"""
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T

def fkine_kr20(q):
    """Cinemática directa incluyendo la pala"""
    # Longitudes
    L0 = 0.0
    L1 = 0.520
    L2 = 0.780
    L3 = 0.150
    L4 = 0.86
    L5 = 0
    L6 = 0.153

    # Matrices DH
    T0 = dh(L0, q[0],     0, 0)
    T1 = dh(L1, q[1],     0.160, np.pi/2)
    T2 = dh(0 , -q[2]+np.pi/2,  L2, 0)
    T3 = dh(0 , q[3]+np.pi,    -L3,  -np.pi/2)
    T4 = dh(L4, q[4],     0,    np.pi/2)
    T5 = dh(0 , q[5]+np.pi,     0,    -np.pi/2)
    T6 = dh(-L6, q[6],    0,   0)
    
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6
    return T

def jacobian_position(q, delta=0.0001):
    """Jacobiano de posición para 7 articulaciones"""
    J = np.zeros((3,7))
    T = fkine_kr20(q)
    
    for i in range(7):
        dq = copy(q)
        dq[i] = dq[i] + delta
        dT = fkine_kr20(dq)
        J[:,i] = (dT[0:3,3] - T[0:3,3])/delta
    
    return J

def ikine(xdes, q0):
    """
    Calcular la cinematica inversa numéricamente
    xdes: posición deseada del efector final [x, y, z]
    q0: configuración inicial/semilla [q1, q2, q3, q4, q5, q6, q7]
    """
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    q = copy(q0)

    # Registrar la evolución del error
    error_history = []

    for i in range(max_iter):
        # Calcular Jacobiano actual
        J = jacobian_position(q, delta)
        # Obtener posición actual (usando los primeros 7 joints)
        T = fkine_kr20(q[:7])
        x = T[0:3,3]
        # Error
        e = xdes - x
        error_history.append(np.linalg.norm(e))
        # Actualizar q
        q = q + np.dot(J.T, e)
        
        # Verificar convergencia
        if np.linalg.norm(e) < epsilon:
            break
            
    return q, error_history

if __name__ == '__main__':
    # Posiciones deseadas
    xd_list = [
        np.array([0.84, 0.125, 0.249]),  # Posición original
        np.array([0.5, 0.5, 0.5]),       # Punto en el espacio de trabajo
        np.array([0.7, 0.0, 0.6])        # Otro punto
    ]

    # Configuración inicial
    q0 = np.array([0.0, 0.0, 0, 0, 0.0, 0, 0.0])

    # Iterar sobre las posiciones deseadas
    for xd in xd_list:
        print(f"Calculando cinemática inversa para posición deseada: {xd}")
        q, error_history = ikine(xd, q0)

        # Verificar el resultado
        T = fkine_kr20(q[:7])
        print('Posición deseada:', xd)
        print('Posición alcanzada:', T[0:3,3])
        print('Error:', np.linalg.norm(xd - T[0:3,3]))
        print('Configuración articular:', q)

        # Graficar la convergencia
        plt.figure(figsize=(8, 6))
        plt.plot(error_history)
        plt.xlabel('Iteración')
        plt.ylabel('Error de posición')
        plt.title(f'Convergencia de la cinemática inversa para posición deseada: {xd}')
        plt.grid()
        plt.show()