#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from lab5functions import *
import rbdl

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node("control_pdg")
    pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
    
    # Markers for visualization
    bmarker_actual = BallMarker(color['RED'])
    bmarker_deseado = BallMarker(color['GREEN'])
    
    # Files for data logging
    fqact = open("/tmp/qactual.txt", "w")
    fqdes = open("/tmp/qdeseado.txt", "w")
    fxact = open("/tmp/xactual.txt", "w")
    fxdes = open("/tmp/xdeseado.txt", "w")

    # Joint names
    jnames = ['base__link01',        # Primera articulación
          'link12__link01',      # Segunda articulación
          'link01__link02',      # Tercera articulación
          'link02__link03',      # Cuarta articulación
          'link03__link04',      # Quinta articulación
          'link04__link05',      # Sexta articulación
          'link05__gripper',     # Gripper (prismática)
          'gripper__gripper_left',  # Gripper izquierdo
          'gripper__gripper_right'] # Gripper derecho
    # Initial joint configuration (en radianes)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Initial velocities
    dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Modificar la configuración deseada para considerar el efecto del prisma
    qdes = np.array([0.8, 0.6, 0.2, -0.3, 0.5, 0.3, 0.05, -0.02, 0.02])

    # Measured end-effector position
    x = kr20_fkine(q)[0:3,3]
    # Desired end-effector position
    xdes = kr20_fkine(qdes)[0:3,3]

    # Initialize ROS message
    jstate = JointState()
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    jstate.position = q

    # Robot instance
    ndof = 9  # Grados de libertad
    dt = 0.01  # Tiempo de muestreo
    robot = Robot(q, dq, ndof, dt)

    # Control gains
    Kp = 0.01*np.diag([30.0, 30.0, 30.0, 25.0, 20.0, 20.0, 5.0, 2.0, 2.0])
    Kd = 5*np.sqrt(Kp)  # Amortiguamiento crítico

    # Loop rate (in Hz)
    freq = 200
    dt = 1.0/freq
    rate = rospy.Rate(freq)

    # Simulation loop
    t = 0.0
    while not rospy.is_shutdown():
        # Current time
        jstate.header.stamp = rospy.Time.now()

        # Read current joint positions and velocities
        q = robot.read_joint_positions()
        dq = robot.read_joint_velocities()
        
        # Compute current end-effector position
        x = kr20_fkine(q)[0:3,3]

        e = qdes - q  # Error en espacio articular
        J = jacobian_position(q, delta=dt)  # Jacobiano
        x = kr20_fkine(q)[0:3,3]  # Posición actual
        ex = xdes - x  # Error cartesiano

        # Combinar control articular y cartesiano
        u = Kp.dot(e) + Kd.dot(-dq) + J.T.dot(ex)

        # Send command to the robot
        robot.send_command(u)

        # Log data
        fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
        fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
        fqact.write(str(t)+' '+' '.join(map(str, q))+'\n')
        fqdes.write(str(t)+' '+' '.join(map(str, qdes))+'\n')

        # Publish joint states
        jstate.position = q
        pub.publish(jstate)

        # Update markers
        bmarker_deseado.xyz(xdes)
        bmarker_actual.xyz(x)

        # Print information
        print("Current q:", q)
        print("Desired q:", qdes)
        print("Current x:", x)
        print("Desired x:", xdes)
        print("Control signal u:", u)
        print("========================")

        # Wait for next iteration
        t = t + dt
        rate.sleep()

    # Close files
    fqact.close()
    fqdes.close()
    fxact.close()
    fxdes.close()