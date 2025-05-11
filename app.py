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
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])

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

# Add 3D debug labels to joints
def add_joint_labels():
    for i in range(len(arm_joints)):
        pos = p.getLinkState(robot, arm_joints[i])[0]
        p.addUserDebugText(f"J{i+1}", pos, textColorRGB=[1, 0, 0], textSize=1.2)

# Render simulation view
def render_sim(joint_values, gripper_val):
    for idx, tgt in zip(arm_joints, joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)
    if len(finger_joints) == 2:
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)

    for _ in range(10): p.stepSimulation()
    
    width, height = 640, 640
    view_matrix = p.computeViewMatrix(cameraEyePosition=[1.5, 0, 1],
                                      cameraTargetPosition=[0, 0, 0.5],
                                      cameraUpVector=[0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=3.1)
    _, _, img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb = np.reshape(img, (height, width, 4))[:, :, :3]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight')
    plt.close()

    joint_text = "Joint Angles:\n" + "\n".join([f"J{i+1} = {v:.2f} rad" for i, v in enumerate(joint_values)])
    joint_text += f"\nGripper = {gripper_val:.3f} m"
    return tmp.name, joint_text

# Move smoothly to given joint angles
def move_to_input_angles(joint_str):
    try:
        target_angles = [float(x.strip()) for x in joint_str.split(",")]
        if len(target_angles) != 7:
            return None, "❌ Enter exactly 7 joint angles."
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
        return None, f"❌ Error: {str(e)}"

# Pick and Place with validation
def pick_and_place(position_str, approach_str, place_str):
    try:
        if not position_str or not approach_str or not place_str:
            return None, "❌ Fill in all three joint angle sets."

        position = [float(x) for x in position_str.split(",")]
        approach = [float(x) for x in approach_str.split(",")]
        place = [float(x) for x in place_str.split(",")]

        if not all(len(lst) == 7 for lst in [position, approach, place]):
            return None, "❌ Each input must have 7 values."

        move_to_input_angles(",".join(map(str, approach)))

        for _ in range(50):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.0)
            p.stepSimulation()
            time.sleep(0.01)

        lifted = [x + 0.1 for x in approach]
        move_to_input_angles(",".join(map(str, lifted)))

        move_to_input_angles(",".join(map(str, place)))

        for _ in range(50):
            for fj in finger_joints:
                p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.04)
            p.stepSimulation()
            time.sleep(0.01)

        return render_sim(place, 0.04)

    except ValueError:
        return None, "❌ Invalid format. Use comma-separated numbers."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Gradio UI
def build_ui():
    with gr.Blocks(title="Franka Pick & Place") as demo:
        gr.Markdown("## 🤖 Franka 7-DOF Pick & Place Arm")

        joint_sliders = [gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}") for i in range(7)]
        gripper_slider = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper Opening")

        img_output = gr.Image(type="filepath", label="Simulation View")
        text_output = gr.Textbox(label="Joint Info")

        def live_update(*vals):
            return render_sim(list(vals[:-1]), vals[-1])

        for s in joint_sliders + [gripper_slider]:
            s.change(fn=live_update, inputs=joint_sliders + [gripper_slider], outputs=[img_output, text_output])

        gr.Button("🔄 Reset").click(lambda: render_sim([0]*7, 0.02), outputs=[img_output, text_output])

        gr.Markdown("### 🎯 Move to Joint Angles")
        joint_input = gr.Textbox(label="Joint Angles (7 values)", placeholder="e.g. 0.1, -0.5, 0.3, -1.0, 0.0, 1.5, 0.8")
        gr.Button("▶️ Move").click(fn=move_to_input_angles, inputs=joint_input, outputs=[img_output, text_output])

        gr.Markdown("### 📦 Pick and Place")
        position_input = gr.Textbox(label="Position Angles (7 values)")
        approach_input = gr.Textbox(label="Approach Angles (7 values)")
        place_input = gr.Textbox(label="Place Angles (7 values)")
        gr.Button("🤖 Execute Pick & Place").click(
            fn=pick_and_place,
            inputs=[position_input, approach_input, place_input],
            outputs=[img_output, text_output]
        )

    return demo

if __name__ == '__main__':
    demo = build_ui()
    demo.launch(debug=True)
