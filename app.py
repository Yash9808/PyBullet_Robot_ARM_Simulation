import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr

# Helper to get Panda arm and gripper joints
def get_panda_joints(robot):
    revolute = []
    finger = []
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[1].decode()
        joint_type = p.getJointInfo(robot, i)[2]
        if joint_type == p.JOINT_REVOLUTE:
            if "finger" in name:
                finger.append(i)
            else:
                revolute.append(i)
    return revolute, finger

# Simulation and rendering function
def simulate_robot(j1, j2, j3, j4, gripper):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0,0,0])
    arm_joints, finger_joints = get_panda_joints(robot)

    # Set arm joint positions
    targets = [j1, j2, j3, j4]
    for idx, tgt in zip(arm_joints[:4], targets):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)

    # Gripper control
    for fj in finger_joints:
        p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper)

    for _ in range(100):
        p.stepSimulation()

    # Render image
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(256, 256, view_matrix, proj_matrix)
    rgb = np.reshape(img, (256, 256, 4))[:, :, :3]

    # Save image
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.imsave(tmp.name, rgb.astype(np.uint8))

    p.disconnect()

    # Create joint info text
    joint_text = f"Joint Angles:\nJ1 = {j1:.2f} rad\nJ2 = {j2:.2f} rad\nJ3 = {j3:.2f} rad\nJ4 = {j4:.2f} rad\nGripper = {gripper:.3f} m"
    return tmp.name, joint_text

# Gradio interface
demo = gr.Interface(
    fn=simulate_robot,
    inputs=[
        gr.Slider(-3.14, 3.14, value=0, label="Joint 1"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 2"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 3"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 4"),
        gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper Opening")
    ],
    outputs=[
        gr.Image(type="filepath", label="Simulation View"),
        gr.Textbox(label="Joint States")
    ],
    title="Panda Arm Simulation with Joint Display",
    description="Control 4 Panda arm joints + gripper. The updated simulation view and joint states are shown below."
)

if __name__ == "__main__":
    demo.launch()
