import torch
import numpy as np


def _to_radians(self, deg):
    return deg * (np.pi / 180)


def _angles_to_dict(self, angles):
    angles = {
        'min_angle_x': self._to_radians(angles[0]),
        'max_angle_x': self._to_radians(angles[1]),
        'min_angle_y': self._to_radians(angles[2]),
        'max_angle_y': self._to_radians(angles[3]),
        'min_angle_z': self._to_radians(angles[4]),
        'max_angle_z': self._to_radians(angles[5])
    }
    return angles


def rot_matrix_x(self, theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3, 3)).astype(np.float32)
    mat[0, 0] = 1.
    mat[1, 1] = np.cos(theta)
    mat[1, 2] = -np.sin(theta)
    mat[2, 1] = np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat


def rot_matrix_y(self, theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3, 3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 2] = np.sin(theta)
    mat[1, 1] = 1.
    mat[2, 0] = -np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat


def rot_matrix_z(self, theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3, 3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 1] = -np.sin(theta)
    mat[1, 0] = np.sin(theta)
    mat[1, 1] = np.cos(theta)
    mat[2, 2] = 1.
    return mat


def pad_rotmat(self, theta):
    """theta = (3x3) rotation matrix"""
    return np.hstack((theta, np.zeros((3, 1))))


def sample_angles(self,
                  bs,
                  min_angle_x,
                  max_angle_x,
                  min_angle_y,
                  max_angle_y,
                  min_angle_z,
                  max_angle_z):
    """Sample random yaw, pitch, and roll angles"""
    angles = []
    for i in range(bs):
        rnd_angles = [
            np.random.uniform(min_angle_x, max_angle_x),
            np.random.uniform(min_angle_y, max_angle_y),
            np.random.uniform(min_angle_z, max_angle_z),
        ]
        angles.append(rnd_angles)
    return np.asarray(angles)


def get_theta(self, angles):
    '''Construct a rotation matrix from angles.
    This uses the Euler angle representation. But
    it should also work if you use an axis-angle
    representation.
    '''
    bs = len(angles)
    theta = np.zeros((bs, 3, 4))

    angles_x = angles[:, 0]
    angles_y = angles[:, 1]
    angles_z = angles[:, 2]
    for i in range(bs):
        theta[i] = self.pad_rotmat(
            np.dot(np.dot(self.rot_matrix_z(angles_z[i]), self.rot_matrix_y(angles_y[i])),
                   self.rot_matrix_x(angles_x[i]))
        )

    return torch.from_numpy(theta).float()
