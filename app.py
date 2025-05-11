import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr

# Setup PyBullet
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# Get joint indices
def get_panda_joints(robot):
    arm, fingers = [], []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode()
        joint_type = info[2]
        if "finger" in name and joint_type == p.JOINT_PRISMATIC:
            fingers.append(i)
        elif joint_type == p.JOINT_REVOLUTE:
            arm.append(i)
    return arm, fingers

arm_joints, finger_joints = get_panda_joints(robot)

# Add 3D debug labels to the robot's joints
debug_labels = []
def add_joint_labels():
    global debug_labels
    for i in debug_labels:
        p.removeUserDebugItem(i)
    debug_labels.clear()
    for idx in arm_joints:
        link_state = p.getLinkState(robot, idx)
        pos = link_state[0]
        lbl = f"J{arm_joints.index(idx)+1}"
        text_id = p.addUserDebugText(lbl, pos, textColorRGB=[1, 0, 0], textSize=1.2)
        debug_labels.append(text_id)

# Render image and info
def render_sim(joint_values, gripper_val):
    # Apply joint controls
    for idx, tgt in zip(arm_joints, joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)

    # Open/close gripper symmetrically
    if len(finger_joints) == 2:
        p.setJointMotorControl2(robot, finger_joints[0], p.POSITION_CONTROL, targetPosition=gripper_val)
        p.setJointMotorControl2(robot, finger_joints[1], p.POSITION_CONTROL, targetPosition=gripper_val)

    for _ in range(10): p.stepSimulation()

    # Refresh joint labels
    add_joint_labels()

    # Camera
    width, height = 512, 512
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb = np.reshape(img, (height, width, 4))[:, :, :3]

    # Image to file
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight')
    plt.close()

    # Text output
    joint_text = "Joint Angles:\n" + "\n".join([f"J{i+1} = {v:.2f} rad" for i, v in enumerate(joint_values)])
    joint_text += f"\nGripper = {gripper_val:.3f} m"
    return tmp.name, joint_text

# Load gripper type (placeholder logic)
def switch_gripper(gripper_type):
    print(f"Switching to gripper: {gripper_type}")
    return f"Switched to: {gripper_type}"

# Gradio UI
with gr.Blocks(title="Franka Arm Control with 7 DoF and Gripper Options") as demo:
    gr.Markdown("## 🤖 Franka 7-DOF Control\nUse the sliders to manipulate the robot arm.")

    # Gripper selection
    gripper_selector = gr.Dropdown(["Two-Finger", "Suction"], value="Two-Finger", label="Select Gripper")
    gripper_feedback = gr.Textbox(label="Gripper Status", interactive=False)
    gripper_selector.change(fn=switch_gripper, inputs=gripper_selector, outputs=gripper_feedback)

    # 7 DoF sliders arranged in two rows
    joint_sliders = []
    with gr.Row():
        for i in range(4):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
    with gr.Row():
        for i in range(4, 7):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
        gripper = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper Opening")

    # Outputs
    with gr.Row():
        img_output = gr.Image(type="filepath", label="Simulation View")
        text_output = gr.Textbox(label="Joint States")

    # Live update
    def live_update(*vals):
        joints = list(vals[:-1])
        grip = vals[-1]
        return render_sim(joints, grip)

    for s in joint_sliders + [gripper]:
        s.change(fn=live_update, inputs=joint_sliders + [gripper], outputs=[img_output, text_output])

    # Reset button
    def reset():
        return render_sim([0]*7, 0.02)

    gr.Button("🔄 Reset Robot").click(fn=reset, inputs=[], outputs=[img_output, text_output])

demo.launch(debug=True)
