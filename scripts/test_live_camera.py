
from pathlib import Path

import mujoco
import matplotlib.pyplot as plt

MODEL_PATH = "external/act/assets/bimanual_viperx_transfer_cube.xml"

START_ARM_POSE = [
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239
]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[:16] = START_ARM_POSE
data.ctrl[:] = START_ARM_POSE
data.qpos[-7:] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

mujoco.mj_forward(model, data)

# Render original ACT top camera
renderer = mujoco.Renderer(
    model,
    height=480,
    width=640
)

renderer.update_scene(data, camera="top")
image = renderer.render()

Path("outputs").mkdir(exist_ok=True)
plt.imsave("outputs/act_live_camera.png", image)

print("Camera image shape:", image.shape)
print("Camera image saved!")

renderer.close()
