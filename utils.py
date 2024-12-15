import numpy as np
import pinocchio as pin
from scipy.linalg import logm

def skew_to_vector(skew_matrix):
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

def so3_error(x: pin.SE3, xd: pin.SE3) -> np.ndarray:
    e_o = logm(xd.rotation @ x.rotation.T)
    e_p = xd.translation - x.translation
    return np.concat([e_p, skew_to_vector(e_o)])

def jacobian(model: pin.Model, data: pin.Data, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    J_l = pin.getFrameJacobian(model, data, site_id, pin.LOCAL)
    dJ_l = pin.getFrameJacobianTimeVariation(model, data, site_id, pin.LOCAL)
    J_w = pin.getFrameJacobian(model, data, site_id, pin.LOCAL_WORLD_ALIGNED)
    dJ_w = pin.getFrameJacobianTimeVariation(model, data, site_id, pin.LOCAL_WORLD_ALIGNED)
    
    J = np.zeros_like(J_l)
    dJ = np.zeros_like(dJ_l)
    
    J[:3, :] = J_w[:3, :]
    J[3:, :] = J_l[3:, :]
    dJ[:3, :] = dJ_w[:3, :]
    dJ[3:, :] = dJ_l[3:, :]
    return J, dJ
