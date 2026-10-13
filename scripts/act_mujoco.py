
import numpy as np
import mujoco

MODEL_PATH = "external/act/assets/bimanual_viperx_transfer_cube.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)


def act_to_mujoco(action):
    """Convert 14 ACT actions to 16 MuJoCo controls."""

    action = np.asarray(action, dtype=np.float64)

    if action.shape != (14,):
        raise ValueError("Expected 14 ACT actions")

    controls = np.zeros(16)

    # Left arm: 6 joints
    controls[0:6] = action[0:6]

    # Left gripper: normalized [0, 1] to physical range
    left = np.clip(action[6], 0, 1)
    left_position = 0.01844 + left * (0.05800 - 0.01844)

    controls[6] = left_position
    controls[7] = -left_position

    # Right arm: 6 joints
    controls[8:14] = action[7:13]

    # Right gripper
    right = np.clip(action[13], 0, 1)
    right_position = 0.01844 + right * (0.05800 - 0.01844)

    controls[14] = right_position
    controls[15] = -right_position

    return np.clip(
        controls,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )


if __name__ == "__main__":
    import torch

    results = torch.load(
        "outputs/act_first_prediction.pt",
        map_location="cpu",
        weights_only=True,
    )

    action = results["predicted_actions"][0, 0].numpy()
    controls = act_to_mujoco(action)

    print("ACT action shape:", action.shape)
    print("MuJoCo control shape:", controls.shape)
    print("Left gripper controls:", controls[6:8])
    print("Right gripper controls:", controls[14:16])
    print("Conversion successful!")
