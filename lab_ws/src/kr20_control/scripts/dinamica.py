#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages
import rbdl
import numpy as np

if __name__ == '_main_':
    # Lectura del modelo del robot a partir de URDF (parsing)
    modelo = rbdl.loadModel('../urdf/kr20.urdf')
    
    # Grados de libertad
    ndof = modelo.q_size
    
    # Configuracion articular
    q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6])
    # Velocidad articular
    dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0])
    # Aceleracion articular
    ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5])
    
    # Arrays numpy
    zeros = np.zeros(ndof)          # Vector de ceros
    tau = np.zeros(ndof)           # Para torque
    g = np.zeros(ndof)             # Para la gravedad
    c = np.zeros(ndof)             # Para el vector de Coriolis+centrifuga
    M = np.zeros([ndof, ndof])     # Para la matriz de inercia
    
    # Parte 1: Calcular los elementos de la dinámica usando InverseDynamics
    
    # 1.1 Vector de gravedad: g(q) = InverseDynamics(q, 0, 0)
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
    print("\nVector de gravedad g(q):")
    print(np.round(g, 3))
    
    # 1.2 Vector de Coriolis y centrífuga
    # c(q,qdot) = InverseDynamics(q, qdot, 0) - g(q)
    tau_coriolis = np.zeros(ndof)
    rbdl.InverseDynamics(modelo, q, dq, zeros, tau_coriolis)
    c = tau_coriolis - g
    print("\nVector de Coriolis c(q,qdot):")
    print(np.round(c, 3))
    
    # 1.3 Matriz de inercia
    # mi = InverseDynamics(q, 0, ei) − g(q)
    for i in range(ndof):
        # Crear vector de aceleración unitaria
        ddq_test = np.zeros(ndof)
        ddq_test[i] = 1.0
        
        # Calcular torque para esta aceleración
        tau_temp = np.zeros(ndof)
        rbdl.InverseDynamics(modelo, q, zeros, ddq_test, tau_temp)
        
        # La columna i de M es el resultado menos el vector de gravedad
        M[:, i] = tau_temp - g
    
    print("\nMatriz de inercia M(q) usando InverseDynamics:")
    print(np.round(M, 3))
    
    # Parte 2: Calcular M y los efectos no lineales usando funciones específicas
    b2 = np.zeros(ndof)          # Para efectos no lineales
    M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
    
    # Calcular M2 usando CompositeRigidBodyAlgorithm
    rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
    
    # Calcular efectos no lineales b2 usando NonlinearEffects
    rbdl.NonlinearEffects(modelo, q, dq, b2)
    
    print("\nMatriz de inercia M2(q) usando CompositeRigidBodyAlgorithm:")
    print(np.round(M2, 3))
    
    print("\nEfectos no lineales b(q,qdot):")
    print(np.round(b2, 3))
    
    # Parte 3: Verificación de la expresión de la dinámica
    # Calcular tau usando la ecuación completa
    tau_verificacion = np.dot(M, ddq) + c + g
    
    # Calcular tau directamente usando InverseDynamics
    tau_directo = np.zeros(ndof)
    rbdl.InverseDynamics(modelo, q, dq, ddq, tau_directo)
    
    print("\nVerificación de la ecuación dinámica:")
    print("tau (InverseDynamics directo) =", np.round(tau_directo, 3))
    print("tau (calculado) =", np.round(tau_verificacion, 3))
    
    # Verificación de consistencia entre métodos
    print("\nDiferencia entre matrices de inercia:")
    print(np.round(M - M2, 3))
    
    print("\nDiferencia entre efectos no lineales:")
    print(np.round((c + g) - b2, 3))