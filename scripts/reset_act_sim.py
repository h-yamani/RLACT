
import mujoco
import mujoco.viewer

MODEL_PATH = "external/act/assets/bimanual_viperx_transfer_cube.xml"

START_ARM_POSE = [
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239
]

BOX_POSE = [0.2, 0.5, 0.05, 1, 0, 0, 0]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Reset simulation
mujoco.mj_resetData(model, data)

# Initialize both robot arms and grippers
data.qpos[:16] = START_ARM_POSE
data.ctrl[:] = START_ARM_POSE

# Initialize cube position and orientation
data.qpos[-7:] = BOX_POSE

# Update robot and cube positions
mujoco.mj_forward(model, data)

print("Robot initialized!")
print("Cube initialized!")
print("Camera: top")

# Open MuJoCo viewer
mujoco.viewer.launch(model, data)
