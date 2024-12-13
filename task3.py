from pathlib import Path
import numpy as np
from scipy.linalg import logm
import os 
import pinocchio as pin
import matplotlib.pyplot as plt
from simulator import Simulator

times = []
position = []
velocity = []

def circular_trajectory(r, t):
    x = r * np.cos(t) 
    y = r * np.sin(t)
    z = r * np.sin(t) - r * np.cos(t)
    return np.array([0.5, 0, 0.5]) + np.array([x, y, z])

def skew_to_vector(skew_matrix):
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

def so3_error(R, Rd):
    error_matrix = R.T @ Rd
    error_log = logm(error_matrix)
    error_vector = skew_to_vector(error_log)
    return error_vector

def pose_error(r, R, r_d,R_d):
    # get error in position
    p_e = np.zeros(6)
    p_e[:3] = r_d - r

    # get error in orientation
    S_e = so3_error(R=R, Rd=R_d)
    p_e[3:] = S_e

    return p_e

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

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    times.append(t)
    position.append(q)
    velocity.append(dq)
    
    # Compute all dynamics quantities
    ee_frame_id = model.getFrameId("end_effector")
    pin.computeAllTerms(model, data, q, dq)
    current = pin.updateFramePlacement(model, data, ee_frame_id)
    
    # Take matricies    
    M = data.M
    g = data.g
    C = data.C
    J, dJ = jacobian(model, data, ee_frame_id)
    Minv = np.linalg.inv(M)
    
    Mx = np.linalg.pinv(np.linalg.multi_dot([J, Minv, J.T]))
    Cx = np.linalg.multi_dot([Mx, J, Minv, C]) - Mx @ dJ
    gx = np.linalg.multi_dot([Mx, J, Minv, g])
    
    dx_d = np.array([0,0,0,0,0,0])
    ddx_d = np.array([0,0,0,0,0,0])
    
    Rd = np.array([
        [np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
        [np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
        [0, 0, 1]
    ])
    xd = circular_trajectory(0.1, t)
    R = current.rotation
    x = current.translation
    
    p = pose_error(x, R, xd, Rd)
    dp = dx_d - J @ dq
    
    kp = 100
    kd = 20
    
    ddx_d = kp * p + kd * dp - dJ @ dq
    f = Mx @ ddx_d + Cx @ dq + gx
    u = J.T @ f
    
    return u

def plotter(filename: str) -> None:
    joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
    ]
    
    plt.figure(figsize=(18, 6))
    plt.title("Joint positions")
    for i, name in enumerate(joint_names):
        plt.plot(times, position[:, i], label=name)
    plt.grid(True, linewidth=1, linestyle="--", alpha=0.7, color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig(f"logs/plots/{filename}_pos.png")
    
    plt.figure(figsize=(18, 6))
    plt.title("Joint velocities")
    for i, name in enumerate(joint_names):
        plt.plot(times, velocity[:, i], label=name)
    plt.grid(True, linewidth=1, linestyle="--", alpha=0.7, color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.savefig(f"logs/plots/{filename}_vel.png")

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/trajectory.mp4",
        fps=30,
        width=1920,
        height=1080 
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=10.0)
    global times, position, velocity
    times = np.array(times)
    position = np.array(position)
    velocity = np.array(velocity)
    plotter("trajectory")

    
if __name__ == "__main__":
    # Import the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    main()