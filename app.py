import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr

# Setup PyBullet simulation
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# 🔒 Fix the base of the robot to the ground
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

def get_panda_joints(robot):
    revolute, finger = [], []
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[1].decode()
        joint_type = p.getJointInfo(robot, i)[2]
        if joint_type == p.JOINT_REVOLUTE:
            (finger if "finger" in name else revolute).append(i)
    return revolute, finger

arm_joints, finger_joints = get_panda_joints(robot)

def render_sim(joint_values, gripper_val):
    # Set joint angles
    for idx, tgt in zip(arm_joints[:4], joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)
    for fj in finger_joints:
        p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)
    for _ in range(10): p.stepSimulation()

    # Render camera
    width, height = 512, 512
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb = np.reshape(img, (height, width, 4))[:, :, :3]

    # Plot image with joint labels
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    labels = ["J1", "J2", "J3", "J4"]
    positions = [(180, 400), (220, 320), (260, 240), (300, 160)]
    for label, pos in zip(labels, positions):
        ax.text(*pos, label, color="red", fontsize=12, fontweight="bold")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight')
    plt.close()

    # Joint state text
    joint_text = (
        f"Joint Angles:\n"
        f"J1 = {joint_values[0]:.2f} rad\n"
        f"J2 = {joint_values[1]:.2f} rad\n"
        f"J3 = {joint_values[2]:.2f} rad\n"
        f"J4 = {joint_values[3]:.2f} rad\n"
        f"Gripper = {gripper_val:.3f} m"
    )
    return tmp.name, joint_text

# Launch Gradio with Blocks
with gr.Blocks(title="Live Robot Control with Reset") as demo:
    gr.Markdown("## 🤖 Live Robot Control\nUse sliders below or click Reset to restore pose.")

    with gr.Row():
        j1 = gr.Slider(-3.14, 3.14, value=0, label="Joint 1", interactive=True)
        j2 = gr.Slider(-3.14, 3.14, value=0, label="Joint 2", interactive=True)
        j3 = gr.Slider(-3.14, 3.14, value=0, label="Joint 3", interactive=True)
        j4 = gr.Slider(-3.14, 3.14, value=0, label="Joint 4", interactive=True)
        gripper = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper Opening", interactive=True)

    with gr.Row():
        img_output = gr.Image(type="filepath", label="Simulation View")
        text_output = gr.Textbox(label="Joint States")

    # Live update function
    def live_update(j1_val, j2_val, j3_val, j4_val, grip_val):
        return render_sim([j1_val, j2_val, j3_val, j4_val], grip_val)

    # Connect sliders to live update
    for slider in [j1, j2, j3, j4, gripper]:
        slider.change(live_update, inputs=[j1, j2, j3, j4, gripper], outputs=[img_output, text_output])

    # Reset button
    def reset():
        return render_sim([0, 0, 0, 0], 0.02)

    gr.Button("🔄 Reset Robot").click(fn=reset, inputs=[], outputs=[img_output, text_output])

demo.launch()
