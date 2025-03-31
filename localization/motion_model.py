import numpy as np
import matplotlib.pyplot as plt

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO

        def get_rot_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        
        def get_transform_matrix(pose):
            mat = np.eye(3)

            xy = pose[:2]
            angle = pose[2]

            mat[:2, -1] = np.array(xy)
            mat[:2, :2] = get_rot_matrix(angle)
            return mat
        
        #noise
        odom_noise = np.random.normal(0, 0.04, odometry.shape)
        odom += odom_noise

        # 3N x 3 Matrix (every particles row converted to transform matrix)
        particle_txns = np.apply_along_axis(get_transform_matrix, axis=1, arr=particles)
        particle_txns = np.vstack(particle_txns)
        
        # apply transform from the odometry
        transform_delta = get_transform_matrix(odometry)
        particle_txns = particle_txns @ transform_delta
        
        # convert back to original form (N x 3 matrix)
        poses = np.zeros(particles.shape)

        # each row is the first column of the rotation matrices for each particle
        rotations = particle_txns[:, 0].reshape(particles.shape[0], 3) 
        poses[:, 2] = np.arctan2(rotations[:, 1], rotations[:, 0])

        # each row is the last column of the transformation matrix for each particle
        positions = particle_txns[:, 2].reshape((particles.shape[0], 3))
        poses[:, :2] = positions[:, :2]
        
        return poses

        ####################################




# import numpy as np
# import matplotlib.pyplot as plt

# class MotionModel:

#     def __init__(self, node):
#         ####################################
#         # TODO
#         # Do any precomputation for the motion
#         # model here.

#         pass

#         ####################################

#     def evaluate(self, _particles, odometry):
#         """
#         Update the particles to reflect probable
#         future states given the odometry data.

#         args:
#             particles: An Nx3 matrix of the form:

#                 [x0 y0 theta0]
#                 [x1 y0 theta1]
#                 [    ...     ]

#             odometry: A 3-vector [dx dy dtheta]

#         returns:
#             particles: An updated matrix of the
#                 same size
#         """

#         ####################################
#         # generate noise, update nisy particles according to equations
#         # Add noise to particles
#         #--------------------
#         particles = _particles[:,:]
#         #proportional to odometry?
#         std_dev = 0.05
#         particles += np.random.normal(0,std_dev,size=particles.shape)
#         #--------------------
#         angles = particles[:,2]
#         num_particles = particles.shape[0]
#         zeros = np.zeros(num_particles)
#         transforms = np.array([[np.cos(angles), -np.sin(angles), particles[:,0]], [np.sin(angles), np.cos(angles), particles[:,1]], [zeros, zeros, zeros+1]])
#         transforms = transforms.transpose(2,0,1)
        
#         delta_part_frame = np.array([[odometry[0]],[odometry[1]],[1]])
#         # Apply each particle transformation to the odometry -> automaticaly gets delta in the world frame, adds it to reference particle
#         new_particles = np.matmul(transforms, delta_part_frame).reshape(num_particles, 3)

#         # Add delta theta, independent of reference frame
#         new_particles[:,2] = angles + odometry[2]
#         return new_particles
#         ####################################

# if __name__ == "__main__":
#     # Debugging
#     particles = np.array([[0, 0, np.pi/6]])
#     odometry = np.array([(2*3**0.5+1)/20, (3**0.5-2)/20, np.pi/60])
    
#     motion_model = MotionModel(None)
#     new_particles = motion_model.evaluate(particles, odometry)

#     print("\nOld Particles:\n", particles)
#     print("Odometry:\n", odometry)
#     print("New Particles:\n", new_particles)

#     plt.scatter(particles[:, 0], particles[:, 1])
#     plt.scatter(new_particles[:, 0], new_particles[:, 1])
#     plt.show()