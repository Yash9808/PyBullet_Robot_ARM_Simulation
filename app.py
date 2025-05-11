import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr
import time

# Setup PyBullet
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# Add a cube to pick (black color)
cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.6, 0, 0.02])
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])  # black

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

# Add debug labels
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

# Render simulation image
def render_sim(joint_values, gripper_val):
    for idx, tgt in zip(arm_joints, joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)
    if len(finger_joints) == 2:
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)
    for _ in range(10): p.stepSimulation()
    add_joint_labels()
    width, height = 1280, 1280
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb = np.reshape(img, (height, width, 4))[:, :, :3]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight', dpi=200)
    plt.close()
    joint_text = "Joint Angles:\n" + "\n".join([f"J{i+1} = {v:.2f} rad" for i, v in enumerate(joint_values)])
    joint_text += f"\nGripper = {gripper_val:.3f} m"
    return tmp.name, joint_text

# Move robot to given angles smoothly
def move_to_input_angles(joint_str):
    try:
        target_angles = [float(x.strip()) for x in joint_str.split(",")]
        if len(target_angles) != 7:
            return None, "❌ Please enter exactly 7 joint angles."
        current = [p.getJointState(robot, idx)[0] for idx in arm_joints]
        steps = 100
        for i in range(steps):
            blend = [(1 - i/steps) * c + (i/steps) * t for c, t in zip(current, target_angles)]
            for idx, val in zip(arm_joints, blend):
                p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=val)
            p.stepSimulation()
            time.sleep(0.01)
        current_grip = p.getJointState(robot, finger_joints[0])[0]
        return render_sim(target_angles, current_grip)
    except Exception as e:
        return None, f"Error: {str(e)}"

# Auto Pick and Place using IK
def pick_and_place_auto(px, py, pz):
    try:
        obj_pos, _ = p.getBasePositionAndOrientation(cube_id)
        above_obj = list(obj_pos)
        above_obj[2] += 0.1

        # Move above object
        ik_approach = p.calculateInverseKinematics(robot, 11, above_obj)
        move_to_input_angles(",".join([f"{a:.4f}" for a in ik_approach[:7]]))

        # Move to grasp position
        ik_grasp = p.calculateInverseKinematics(robot, 11, obj_pos)
        move_to_input_angles(",".join([f"{a:.4f}" for a in ik_grasp[:7]]))

        # Close gripper
        for _ in range(30):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.0)
            p.stepSimulation()
            time.sleep(0.01)

        # Lift
        lift_pos = list(obj_pos)
        lift_pos[2] += 0.15
        ik_lift = p.calculateInverseKinematics(robot, 11, lift_pos)
        move_to_input_angles(",".join([f"{a:.4f}" for a in ik_lift[:7]]))

        # Move to place
        place_pos = [px, py, pz]
        ik_place = p.calculateInverseKinematics(robot, 11, place_pos)
        move_to_input_angles(",".join([f"{a:.4f}" for a in ik_place[:7]]))

        # Open gripper
        for _ in range(30):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.04)
            p.stepSimulation()
            time.sleep(0.01)

        return render_sim(ik_place[:7], 0.04)
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gripper selection stub
def switch_gripper(gripper_type):
    return f"Switched to: {gripper_type}"

# Gradio Interface
with gr.Blocks(title="Franka Arm Auto Pick and Place") as demo:
    gr.Markdown("## 🤖 Franka 7-DOF Control")

    gripper_selector = gr.Dropdown(["Two-Finger", "Suction"], value="Two-Finger", label="Select Gripper")
    gripper_feedback = gr.Textbox(label="Gripper Status", interactive=False)
    gripper_selector.change(fn=switch_gripper, inputs=gripper_selector, outputs=gripper_feedback)

    joint_sliders = []
    with gr.Row():
        for i in range(4):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
    with gr.Row():
        for i in range(4, 7):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
        gripper = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper")

    with gr.Row():
        img_output = gr.Image(type="filepath", label="Simulation View")
        text_output = gr.Textbox(label="Joint States")

    def live_update(*vals):
        joints = list(vals[:-1])
        grip = vals[-1]
        return render_sim(joints, grip)

    for s in joint_sliders + [gripper]:
        s.change(fn=live_update, inputs=joint_sliders + [gripper], outputs=[img_output, text_output])

    gr.Button("🔄 Reset Robot").click(
        fn=lambda: render_sim([0]*7, 0.02),
        inputs=[], outputs=[img_output, text_output]
    )

    gr.Markdown("### 🎯 Auto Pick and Place (from object to custom location)")
    px = gr.Slider(0.3, 0.8, value=0.4, label="Place X")
    py = gr.Slider(-0.3, 0.3, value=0.0, label="Place Y")
    pz = gr.Slider(0.02, 0.4, value=0.05, label="Place Z")

    gr.Button("🤖 Auto Pick and Place").click(
        fn=pick_and_place_auto,
        inputs=[px, py, pz],
        outputs=[img_output, text_output]
    )

demo.launch(debug=True)
