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

    def evaluate(self, _particles, odometry):
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
        # generate noise, update nisy particles according to equations
        # Add noise to particles
        #--------------------
        particles = _particles[:,:]
        #proportional to odometry?
        std_dev = 0.05
        particles += np.random.normal(0,std_dev,size=particles.shape)
        #--------------------
        angles = particles[:,2]
        num_particles = particles.shape[0]
        zeros = np.zeros(num_particles)
        transforms = np.array([[np.cos(angles), -np.sin(angles), particles[:,0]], [np.sin(angles), np.cos(angles), particles[:,1]], [zeros, zeros, zeros+1]])
        transforms = transforms.transpose(2,0,1)
        
        delta_part_frame = np.array([[odometry[0]],[odometry[1]],[1]])
        # Apply each particle transformation to the odometry -> automaticaly gets delta in the world frame, adds it to reference particle
        new_particles = np.matmul(transforms, delta_part_frame).reshape(num_particles, 3)

        # Add delta theta, independent of reference frame
        new_particles[:,2] = angles + odometry[2]
        return new_particles
        ####################################

if __name__ == "__main__":
    # Debugging
    particles = np.array([[0, 0, np.pi/6]])
    odometry = np.array([(2*3**0.5+1)/20, (3**0.5-2)/20, np.pi/60])
    
    motion_model = MotionModel(None)
    new_particles = motion_model.evaluate(particles, odometry)

    print("\nOld Particles:\n", particles)
    print("Odometry:\n", odometry)
    print("New Particles:\n", new_particles)

    plt.scatter(particles[:, 0], particles[:, 1])
    plt.scatter(new_particles[:, 0], new_particles[:, 1])
    plt.show()