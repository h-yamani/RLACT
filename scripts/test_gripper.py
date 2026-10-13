
import time
import mujoco
import mujoco.viewer

MODEL_PATH = "external/mujoco_menagerie/aloha/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Reset robot to its initial pose
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Gripper actuator indices
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13

CLOSED = 0.002
OPEN = 0.037

print("Starting ALOHA gripper test...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Open both grippers
        print("Opening grippers...")
        data.ctrl[LEFT_GRIPPER] = OPEN
        data.ctrl[RIGHT_GRIPPER] = OPEN

        for _ in range(500):
            if not viewer.is_running():
                break
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Close both grippers
        print("Closing grippers...")
        data.ctrl[LEFT_GRIPPER] = CLOSED
        data.ctrl[RIGHT_GRIPPER] = CLOSED

        for _ in range(500):
            if not viewer.is_running():
                break
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

print("Gripper test finished.")
