import numpy as np
from copy import copy

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


def fkine_kr20(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q0, q1, q2, q3, q4, q5, q6]

    """
    # Longitudes (en metros)
    L0 = 0.0
    L1 = 0.520
    L2 = 0.796
    L3 = 0.150
    L4 = 0.86
    L5 = 0
    L6 = 0.153

    # Matrices DH (completar)
    T0 = dh(L0,     q[0],               0,      0) #
    T1 = dh(L1,     q[1],               0,  pi/2) #
    T2 = dh(0 ,     q[2]-101.59*pi/180, L2,     0) #
    T3 = dh(0 ,     q[3]-168*pi/180,    L3,    pi/2) #
    T4 = dh(L4  ,   q[4]+pi,              0 ,     pi/2)
    T5 = dh(0  ,    q[5]+pi ,           0 ,     pi/2)
    T6 = dh(L6  ,  q[6] ,         0 ,     0)
    # Efector final con respecto a la base
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6  
    return T


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((3,9))
    # Transformacion homogenea inicial (usando q)
    x = fkine_kr20(q)
    # Iteracion para la derivada de cada columna

    for i in range(9):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+dq)
        dx = fkine_kr20(dq)
        columna_i = 1/delta * (dx-x)
        columna_i = columna_i[0:3,3]
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[:,i] = columna_i


    return J
    
def ikine(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la
    configuracion articular inicial de q0. Emplear el metodo de newton
    """
    fqact = open("/tmp/error_newton.txt", "w")
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    q = copy(q0)
    for i in range(max_iter):
        # Main loop
        
        J= jacobian_position(q,delta)
        T = fkine_kr20(q)
        f = T [0:3,3]
        e = xdes -f
        q = q + np.dot(J.T,e)
        if q[0]<0:
            q[0] = 0
        if q[5]>=2:
            q[5]=2
        if q[5]<=-2:
            q[5] = -2
        fqact.write(str(np.linalg.norm(e))+'\n ')
        if (np.linalg.norm(e)<epsilon):      
            break
        
    return q

def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    J = np.zeros((7,6))
    # Implementar este Jacobiano aqui

    
    return J



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
