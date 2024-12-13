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

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: dict) -> np.ndarray:
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
    
    # Convert desired pose to SE3
    desired_position = desired['pos']
    desired_quaternion = desired['quat'] # [w, x, y, z] in MuJoCo format
    desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) # Convert to [x,y,z,w] for Pinocchio
    desired_se3 = pin.XYZQUATToSE3(np.concat([desired_position, desired_quaternion_pin]))
    
    dx_d = np.array([0,0,0,0,0,0])
    ddx_d = np.array([0,0,0,0,0,0])
    
    p = so3_error(current, desired_se3)
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
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/target_moving.mp4",
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
    plotter("target_moving")

    
if __name__ == "__main__":
    # Import the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    main()