import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import gradio as gr

def simulate_robot(joint1, joint2):
    # Start PyBullet (headless)
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load ground and robot
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0])

    # Apply joint controls (just controlling 2 for demo)
    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, targetPosition=joint1)
    p.setJointMotorControl2(robot, 3, p.POSITION_CONTROL, targetPosition=joint2)

    # Step simulation
    for _ in range(100):
        p.stepSimulation()

    # Render image
    view_matrix = p.computeViewMatrix([1.5, 0, 1], [0, 0, 0.5], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 3.1)
    _, _, img, _, _ = p.getCameraImage(256, 256, view_matrix, proj_matrix)
    rgb_array = np.reshape(img, (256, 256, 4))[:, :, :3]

    # Save as temp image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.imsave(temp_file.name, rgb_array.astype(np.uint8))
    p.disconnect()
    return temp_file.name

# Gradio Interface
demo = gr.Interface(
    fn=simulate_robot,
    inputs=[
        gr.Slider(-3.14, 3.14, value=0, label="Joint 1"),
        gr.Slider(-3.14, 3.14, value=0, label="Joint 2"),
    ],
    outputs=gr.Image(type="filepath", label="Simulated Robot View"),
    title="PyBullet Panda Simulation",
    description="Control joint angles and simulate the Panda robot using PyBullet headless mode.",
)

if __name__ == "__main__":
    demo.launch()
