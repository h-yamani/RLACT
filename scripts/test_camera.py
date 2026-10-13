
from pathlib import Path

import mujoco
import matplotlib.pyplot as plt

MODEL_PATH = "external/mujoco_menagerie/aloha/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Reset to initial robot configuration
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Render the overhead camera
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="overhead_cam")
image = renderer.render()

Path("outputs").mkdir(exist_ok=True)

plt.imsave("outputs/mujoco_overhead.png", image)

renderer.close()

print("Camera image saved to outputs/mujoco_overhead.png")
