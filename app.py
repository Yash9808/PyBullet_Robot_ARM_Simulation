import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gradio as gr
import time

# Initialize PyBullet
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# Add object (default: black cube)
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

def add_joint_labels():
    for i in range(len(arm_joints)):
        pos = p.getLinkState(robot, arm_joints[i])[0]
        p.addUserDebugText(f"J{i+1}", pos, textColorRGB=[1, 0, 0], textSize=1.2)

# Camera config (yaw, pitch, dist)
camera_angles = [45, -30, 1.5]  # Default values
def render_sim(joint_values, gripper_val, cube_color):
    for idx, tgt in zip(arm_joints, joint_values):
        p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=tgt)

    for _ in range(5):
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=gripper_val)
        p.stepSimulation()

    # Cube color update
    p.changeVisualShape(cube_id, -1, rgbaColor=cube_color + [1])

    # View camera
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.5],
        distance=camera_angles[2],
        yaw=camera_angles[0],
        pitch=camera_angles[1],
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(640, 640, view_matrix, proj_matrix)
    rgb = np.reshape(img, (640, 640, 4))[:, :, :3]

    fig, ax = plt.subplots()
    ax.imshow(rgb.astype(np.uint8))
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
    plt.close()

    return tmp.name, f"Joint Angles: {[round(j,2) for j in joint_values]}, Gripper: {round(gripper_val, 3)}"

def move_to_input_angles(joint_str):
    try:
        target_angles = [float(x.strip()) for x in joint_str.split(",")]
        if len(target_angles) != 7:
            return None, "❌ Please enter exactly 7 joint angles."

        current = [p.getJointState(robot, idx)[0] for idx in arm_joints]
        for i in range(50):
            blend = [(1 - i/50) * c + (i/50) * t for c, t in zip(current, target_angles)]
            for idx, val in zip(arm_joints, blend):
                p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, targetPosition=val)
            p.stepSimulation()
            time.sleep(0.005)

        grip = p.getJointState(robot, finger_joints[0])[0]
        return render_sim(target_angles, grip, current_cube_color)
    except Exception as e:
        return None, str(e)

# Pick and place
current_cube_color = [0, 0, 0]  # Default black

def pick_and_place(position_str, approach_str, place_str):
    position = [float(x) for x in position_str.split(",")]
    approach = [float(x) for x in approach_str.split(",")]
    place = [float(x) for x in place_str.split(",")]

    move_to_input_angles(','.join(map(str, approach)))

    for _ in range(30):
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.0)
        p.stepSimulation()
        time.sleep(0.01)

    move_to_input_angles(','.join([str(x + 0.1) for x in approach]))
    move_to_input_angles(','.join(map(str, place)))

    for _ in range(30):
        for fj in finger_joints:
            p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL, targetPosition=0.04)
        p.stepSimulation()
        time.sleep(0.01)

    return render_sim(place, 0.04, current_cube_color)

# UI
with gr.Blocks(title="Franka Arm Control with Pick & Place") as demo:
    gr.Markdown("# 🤖 Franka 7-DOF + Pick & Place")

    # Camera controls
    yaw = gr.Slider(-180, 180, value=camera_angles[0], label="Yaw")
    pitch = gr.Slider(-90, 90, value=camera_angles[1], label="Pitch")
    dist = gr.Slider(0.5, 3.0, value=camera_angles[2], label="Camera Distance")

    def update_camera(y, p_, d):
        camera_angles[0], camera_angles[1], camera_angles[2] = y, p_, d
        return "Camera updated"

    gr.Button("🎥 Update Camera").click(update_camera, [yaw, pitch, dist], outputs=gr.Textbox(label="Status"))

    # Cube color
    cube_color_picker = gr.ColorPicker(label="Pick Cube Color", value="#000000")

    def update_color(color_hex):
        global current_cube_color
        rgb = tuple(int(color_hex[i:i+2], 16)/255. for i in (1, 3, 5))
        current_cube_color = list(rgb)
        return f"Updated color to {rgb}"

    cube_color_picker.change(fn=update_color, inputs=cube_color_picker, outputs=gr.Textbox(label="Color Info"))

    # Manual joints + gripper
    joint_sliders = [gr.Slider(-3.14, 3.14, value=0, label=f"Joint {i+1}") for i in range(7)]
    gripper = gr.Slider(0.0, 0.04, value=0.02, step=0.001, label="Gripper")

    def live_update(*vals):
        return render_sim(list(vals[:-1]), vals[-1], current_cube_color)

    for s in joint_sliders + [gripper]:
        s.change(fn=live_update, inputs=joint_sliders + [gripper], outputs=[gr.Image(type="filepath"), gr.Textbox()])

    # Pick and Place Inputs
    gr.Markdown("## 🧠 Pick & Place Inputs")
    pos_box = gr.Textbox(label="Position Angles")
    approach_box = gr.Textbox(label="Approach Angles")
    place_box = gr.Textbox(label="Place Angles")
    gr.Button("🚚 Run Pick & Place").click(pick_and_place, [pos_box, approach_box, place_box], outputs=[gr.Image(type="filepath"), gr.Textbox()])

demo.launch(debug=True)
