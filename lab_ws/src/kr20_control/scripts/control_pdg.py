#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages
import numpy as np
import rbdl

if __name__ == '_main_':
    rospy.init_node("control_pdg")
    pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
    bmarker_actual  = BallMarker(color['RED'])
    bmarker_deseado = BallMarker(color['GREEN'])
    
    # Files to store data
    fqact = open("/home/user/fundamentos_proyecto/lab_ws/src/proyecto/kr20_control/scripts/qactual.txt", "w")
    fqdes = open("/home/user/fundamentos_proyecto/lab_ws/src/proyecto/kr20_control/scripts/qdeseado.txt", "w")
    fxact = open("/home/user/fundamentos_proyecto/lab_ws/src/proyecto/kr20_control/scripts/xactual.txt", "w")
    fxdes = open("/home/user/fundamentos_proyecto/lab_ws/src/proyecto/kr20_control/scripts/xdeseado.txt", "w")
    
    # Joint names based on the KR20 URDF
    jnames = ['base__link01',
              'link12__link01',
              'link01__link02', 
              'link02__link03', 
              'link03__link04', 
              'link04__link05',
              'link05__shovel_base',
              'shovel_base__pusher']
    
    # JointState message
    jstate = JointState()
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    
    # Initial joint configuration (in radians)
    # Adjusted to match the KR20 robot's joint limits
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Initial velocities
    dq = np.zeros(8)
    
    # Desired joint configuration
    # Ensure these are within the joint limits from the URDF
    qdes = np.array([0.0, 1.0, 0.5, 0.5, 1.0, -1.0, 0.5, 1.0])
    
    # Posicion resultante de la configuracion articular deseada
    xdes = kr20_fkine(qdes)[0:3,3]
    # Copiar la configuracion articular en el mensaje a ser publicado
    jstate.position = q
    pub.publish(jstate)
    
    # Load RBDL model
    modelo = rbdl.loadModel('../urdf/kr20.urdf')
    ndof = modelo.q_size  # Degrees of freedom
    
    # Frequency of sending commands
    freq = 20
    dt = 1.0/freq
    rate = rospy.Rate(freq)
    
    # Dynamic simulator
    robot = Robot(q, dq, ndof, dt)
    
    # Control gains (adjusted for 8 DOF)
    Kp = np.diag([50, 50, 50, 30, 20, 10, 10, 5]) / 100
    Kd = np.diag([10, 10, 10, 5, 5, 3, 3, 2]) / 7
    
    # Execution loop
    t = 0.0
    while not rospy.is_shutdown():
        # Read simulator values
        q = robot.read_joint_positions()
        dq = robot.read_joint_velocities()
        
        # Posicion actual del efector final
        x = kr20_fkine(q)[0:3,3]
        
        # Time stamp
        jstate.header.stamp = rospy.Time.now()
        
	# Almacenamiento de datos
        fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
        fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
        fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+' '+str(q[7])+'\n ')
        fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+' '+str(qdes[7])+'\n ')

        
        # Compute inverse dynamics
        g = np.zeros(ndof)
        zeros = np.zeros(ndof)
        rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
        
        # Compute control input
        e = qdes - q
        u = np.dot(Kp, e) - np.dot(Kd, dq) + g
        
        # Convergence check
        tol = 0.001
        error = np.linalg.norm(e)
        print("\nerror norm: ", error)
        if error < tol:
            break
        
        # Send command to robot
        robot.send_command(u)
        
        # Publish joint states
        jstate.position = q
        pub.publish(jstate)
        
        t += dt
        rate.sleep()
    
    # Close files
    fqact.close()
    fqdes.close()
    fxact.close()
    fxdes.close()
