import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr

# Setup
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0])

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

arm_joints, finger_joints = get_panda_joints(robot)

# Common function to simulate and render
def render_sim(joint_values, gripper_val):
    # Set joint positions
    for idx, tgt in zip(arm_joints[:4], joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)

    for fj in finger_joints:
        p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)

    for _ in range(10):
        p.stepSimulation()

    # Camera
    width, height = 512, 512
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb = np.reshape(img, (height, width, 4))[:, :, :3]

    # Annotate with labels
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    # Approx joint label positions (you may tweak)
    labels = ["J1", "J2", "J3", "J4"]
    positions = [(180, 400), (220, 320), (260, 240), (300, 160)]
    for label, pos in zip(labels, positions):
        ax.text(*pos, label, color="red", fontsize=12, fontweight="bold")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight')
    plt.close()

    # Joint info
    joint_text = (
        f"Joint Angles:\n"
        f"J1 = {joint_values[0]:.2f} rad\n"
        f"J2 = {joint_values[1]:.2f} rad\n"
        f"J3 = {joint_values[2]:.2f} rad\n"
        f"J4 = {joint_values[3]:.2f} rad\n"
        f"Gripper = {gripper_val:.3f} m"
    )
    return tmp.name, joint_text

# Main control
def update_robot(j1, j2, j3, j4, gripper):
    return render_sim([j1, j2, j3, j4], gripper)

# Reset button
def reset_robot():
    return render_sim([0, 0, 0, 0], 0.02)

# Interface
demo = gr.Interface(
    fn=update_robot,
    inputs=[
        gr.Slider(-3.14, 3.14, value=0, label="Joint 1"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 2"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 3"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 4"),
        gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper Opening")
    ],
    outputs=[
        gr.Image(type="filepath", label="Live Simulation View (with Joint Labels)"),
        gr.Textbox(label="Live Joint States")
    ],
    live=True,
    title="Live Robot Control with Joint Labeling",
    description="Control 4 joints and gripper. Click Reset to restore default position.",
)

# Add Reset Button
demo.add_component("button", "Reset", variant="secondary", elem_id="reset_btn")
demo.load(reset_robot, None, [demo.outputs[0], demo.outputs[1]], trigger="reset_btn")

if __name__ == "__main__":
    demo.launch()
