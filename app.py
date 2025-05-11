import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr
import time

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.6, 0, 0.02])
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])

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

def render_sim(joint_values, gripper_val, cam_xyz, target_xyz):
    for idx, tgt in zip(arm_joints, joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)
    if len(finger_joints) == 2:
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)
    for _ in range(10): p.stepSimulation()
    add_joint_labels()
    width, height = 1280, 1280
    view_matrix = p.computeViewMatrix(cam_xyz, target_xyz, [0, 0, 1])
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

def move_to_input_angles(joint_str, cam_xyz, target_xyz):
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
        return render_sim(target_angles, current_grip, cam_xyz, target_xyz)
    except Exception as e:
        return None, f"Error: {str(e)}"

def pick_and_place(position_str, approach_str, place_str, cam_xyz, target_xyz):
    try:
        position_angles = [float(x.strip()) for x in position_str.split(",")]
        approach_angles = [float(x.strip()) for x in approach_str.split(",")]
        place_angles = [float(x.strip()) for x in place_str.split(",")]

        if len(position_angles) != 7 or len(approach_angles) != 7 or len(place_angles) != 7:
            return None, "❌ All inputs must have 7 joint angles."

        # Move to pre-grasp approach pose
        move_to_input_angles(approach_str, cam_xyz, target_xyz)

        # Grasping: move to actual grasp pose (over the object)
        steps = 50
        for i in range(steps):
            interpolated = [(1 - i/steps) * approach_angles[j] + (i/steps) * position_angles[j] for j in range(7)]
            for idx, val in zip(arm_joints, interpolated):
                p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=val)
            p.stepSimulation()
            time.sleep(0.01)

        # Close the gripper to grasp
        for _ in range(30):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.0)  # Closed
            p.stepSimulation()
            time.sleep(0.01)

        # Lift object slightly from grasp position
        lifted_angles = [a + 0.1 for a in position_angles]  # Add a small lift to each joint
        for i in range(steps):
            lifted_pose = [(1 - i/steps) * position_angles[j] + (i/steps) * lifted_angles[j] for j in range(7)]
            for idx, val in zip(arm_joints, lifted_pose):
                p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=val)
            p.stepSimulation()
            time.sleep(0.01)

        # Move to place location
        move_to_input_angles(place_str, cam_xyz, target_xyz)

        # Release the gripper
        for _ in range(30):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.04)  # Open
            p.stepSimulation()
            time.sleep(0.01)

        return render_sim(place_angles, 0.04, cam_xyz, target_xyz)
    except Exception as e:
        return None, f"Error: {str(e)}"


# Copy current joint angles
def copy_current_joint_angles(*vals):
    return ", ".join([f"{v:.4f}" for v in vals])

with gr.Blocks(title="Franka Arm with 3D Camera Control") as demo:
    gr.Markdown("## 🤖 Franka Robot with Camera + Joint Control")

    # Joint and gripper sliders
    joint_sliders = []
    with gr.Row():
        for i in range(4):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
    with gr.Row():
        for i in range(4, 7):
            joint_sliders.append(gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}"))
        gripper = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper")

    # Compact camera + target controls
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Camera Position**", elem_id="camera-position-label")
            cam_x = gr.Slider(-3, 3, value=1.5, label="X")
            cam_y = gr.Slider(-3, 3, value=0.0, label="Y")
            cam_z = gr.Slider(-1, 3, value=1.0, label="Z")
        with gr.Column(scale=1):
            gr.Markdown("**Target Point**", elem_id="target-point-label")
            tgt_x = gr.Slider(-1, 1, value=0.0, label="X")
            tgt_y = gr.Slider(-1, 1, value=0.0, label="Y")
            tgt_z = gr.Slider(0, 2, value=0.5, label="Z")


    # Live image + output
    with gr.Row():
        img_output = gr.Image(type="filepath", label="Simulation View")
        text_output = gr.Textbox(label="Joint States", lines=10)

    # Live simulation update
    def live_update(*vals):
        joints = list(vals[:7])
        grip = vals[7]
        cam = [vals[8], vals[9], vals[10]]
        tgt = [vals[11], vals[12], vals[13]]
        return render_sim(joints, grip, cam, tgt)

    sliders = joint_sliders + [gripper, cam_x, cam_y, cam_z, tgt_x, tgt_y, tgt_z]
    for s in sliders:
        s.change(fn=live_update, inputs=sliders, outputs=[img_output, text_output])

    gr.Button("🔄 Reset Robot").click(
        fn=lambda: render_sim([0]*7, 0.02, [1.5, 0, 1.0], [0, 0, 0.5]),
        inputs=[], outputs=[img_output, text_output]
    )

    # Move to angles box
    gr.Markdown("### ✍️ Move Robot to Custom Joint Angles")
    joint_input_box = gr.Textbox(label="Enter 7 Joint Angles (comma-separated)")
    gr.Button("▶️ Move to Angles").click(
        fn=lambda s, x, y, z, tx, ty, tz: move_to_input_angles(s, [x, y, z], [tx, ty, tz]),
        inputs=[joint_input_box, cam_x, cam_y, cam_z, tgt_x, tgt_y, tgt_z],
        outputs=[img_output, text_output]
    )

    # Pick and Place section
    gr.Markdown("### 🧾 Pick and Place Input (3 sets of joint angles)")

    with gr.Row():
        position_input = gr.Textbox(label="Object Position Angles")
        gr.Button("📋 Copy").click(fn=copy_current_joint_angles, inputs=joint_sliders, outputs=position_input)

    with gr.Row():
        approach_input = gr.Textbox(label="Approach Angles")
        gr.Button("📋 Copy").click(fn=copy_current_joint_angles, inputs=joint_sliders, outputs=approach_input)

    with gr.Row():
        place_input = gr.Textbox(label="Place Angles")
        gr.Button("📋 Copy").click(fn=copy_current_joint_angles, inputs=joint_sliders, outputs=place_input)

    gr.Button("🤖 Perform Pick and Place").click(
        fn=lambda p, a, pl, x, y, z, tx, ty, tz: pick_and_place(p, a, pl, [x, y, z], [tx, ty, tz]),
        inputs=[position_input, approach_input, place_input, cam_x, cam_y, cam_z, tgt_x, tgt_y, tgt_z],
        outputs=[img_output, text_output]
    )

demo.launch(debug=True)


