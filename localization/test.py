import numpy as np

def get_rot_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        
def get_transform_matrix(pose): #actual transformation
    mat = np.eye(3)

    xy = pose[:2]
    angle = pose[2]

    mat[:2, -1] = np.array(xy)
    mat[:2, :2] = get_rot_matrix(angle)
    return mat

particles = [[1,2,np.pi],
             [2,2,np.pi],
             [3,4,np.pi],
             [2,4,np.pi]]

odometry = [1,0,0]

# particle_txns = np.apply_along_axis(get_transform_matrix, axis=1, arr=particles)
# print(particle_txns)
# particle_txns = np.vstack(particle_txns)
# transform_delta = get_transform_matrix(odometry)
# # print(f'\n{transform_delta}')
# particle_txns = particle_txns @ transform_delta
# print(f'\n{particle_txns}')

# rotations = particle_txns[:, 0].reshape(4, 3)
# print(f'\n{rotations}')
# print(f'\n{np.arctan2(rotations[:, 1], rotations[:, 0])}')

# positions = particle_txns[:, 2].reshape((4, 3))
# print(f'\n{positions}')

poses = np.array([[0,0,0],[0,0,0],[0,0,0]])
poses[:,0] += np.array([1,1,1])
print(poses)

