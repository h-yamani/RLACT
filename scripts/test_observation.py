
import mujoco
import numpy as np

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

# Original ACT gripper calibration
GRIPPER_CLOSE = 0.01844
GRIPPER_OPEN = 0.05800

def normalize_gripper(value):
    return (value - GRIPPER_CLOSE) / (
        GRIPPER_OPEN - GRIPPER_CLOSE
    )

def get_robot_state(data):
    qpos = data.qpos

    left_arm = qpos[0:6]
    left_gripper = normalize_gripper(qpos[6])

    right_arm = qpos[8:14]
    right_gripper = normalize_gripper(qpos[14])

    return np.concatenate([
        left_arm,
        [left_gripper],
        right_arm,
        [right_gripper],
    ])

state = get_robot_state(data)

print("Robot state:", state)
print("State shape:", state.shape)
print("Left gripper normalized:", state[6])
print("Right gripper normalized:", state[13])
