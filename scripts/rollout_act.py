
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

# Import ACT-to-MuJoCo controller
sys.path.insert(0, str(Path(__file__).resolve().parent))
from act_mujoco import act_to_mujoco


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "external/act/assets/bimanual_viperx_transfer_cube.xml"

POLICY_ID = "lerobot/act_aloha_sim_transfer_cube_human"
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

VIDEO_PATH = "outputs/act_rollout.mp4"

START_ARM_POSE = np.array([
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239
], dtype=np.float64)

BOX_POSE = [0.2, 0.5, 0.05, 1, 0, 0, 0]

CONTROL_HZ = 50
CHUNK_SIZE = 100
TOTAL_STEPS = 400

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

DEVICE = torch.device("cpu")

GRIPPER_CLOSE = 0.01844
GRIPPER_OPEN = 0.05800


# ============================================================
# 2. LOAD DATASET STATISTICS
# ============================================================

print("Loading dataset statistics...")

dataset = LeRobotDataset(DATASET_ID)
stats = dataset.meta.stats


# ============================================================
# 3. LOAD PRETRAINED ACT POLICY
# ============================================================

print("Loading pretrained ACT policy...")

policy = ACTPolicy.from_pretrained(POLICY_ID)
policy = policy.to(DEVICE)
policy.eval()


# ============================================================
# 4. INITIALIZE MUJOCO
# ============================================================

print("Loading MuJoCo environment...")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Initialize robot arms and grippers
data.qpos[:16] = START_ARM_POSE
data.ctrl[:] = START_ARM_POSE

# Initialize red cube
data.qpos[-7:] = BOX_POSE

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(
    model,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH
)


# ============================================================
# 5. VIDEO RECORDING
# ============================================================

Path("outputs").mkdir(parents=True, exist_ok=True)

video = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    CONTROL_HZ,
    (IMAGE_WIDTH, IMAGE_HEIGHT)
)

if not video.isOpened():
    renderer.close()
    raise RuntimeError("Could not create MP4 video")

print("Recording video to:", VIDEO_PATH)


# ============================================================
# 6. ROBOT OBSERVATIONS
# ============================================================

def normalize_gripper(value):
    return (
        (value - GRIPPER_CLOSE)
        / (GRIPPER_OPEN - GRIPPER_CLOSE)
    )


def get_state():
    qpos = data.qpos

    state = np.concatenate([
        qpos[0:6],
        [normalize_gripper(qpos[6])],
        qpos[8:14],
        [normalize_gripper(qpos[14])]
    ])

    return torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


def get_image():
    renderer.update_scene(data, camera="top")
    image = renderer.render().copy()

    image = torch.from_numpy(image).float() / 255.0

    return image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


# ============================================================
# 7. NORMALIZATION
# ============================================================


def normalize(value, key):

    if key == "observation.images.top":
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=DEVICE
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=DEVICE
        ).view(1, 3, 1, 1)

    else:
        mean = torch.as_tensor(
            stats[key]["mean"],
            dtype=torch.float32,
            device=DEVICE
        )

        std = torch.as_tensor(
            stats[key]["std"],
            dtype=torch.float32,
            device=DEVICE
        ).clamp(min=1e-6)

    return (value - mean) / std



def unnormalize_action(action):
    mean = torch.as_tensor(
        stats["action"]["mean"],
        dtype=torch.float32,
        device=DEVICE
    )

    std = torch.as_tensor(
        stats["action"]["std"],
        dtype=torch.float32,
        device=DEVICE
    )

    return action * std + mean


# ============================================================
# 8. ACT INFERENCE
# ============================================================

def predict_chunk():
    image = get_image()
    state = get_state()

    batch = {
        "observation.images.top": normalize(
            image, "observation.images.top"
        ),
        "observation.state": normalize(
            state, "observation.state"
        )
    }

    with torch.inference_mode():
        predicted_chunk = policy.predict_action_chunk(batch)

    actions = unnormalize_action(predicted_chunk)

    return actions[0].cpu().numpy()


# ============================================================
# 9. SIMULATION SETTINGS
# ============================================================

physics_steps = round(
    1.0 / (CONTROL_HZ * model.opt.timestep)
)

print("\nStarting ACT rollout...")
print("Device:", DEVICE)
print("Control frequency:", CONTROL_HZ, "Hz")
print("Physics timestep:", model.opt.timestep)
print("Physics steps per action:", physics_steps)
print("Total control steps:", TOTAL_STEPS)


# ============================================================
# 10. CLOSED-LOOP ROLLOUT + VIDEO
# ============================================================

try:
    actions = None

    for step in range(TOTAL_STEPS):

        # Predict a new action chunk
        if step % CHUNK_SIZE == 0:
            print(f"\nPredicting ACT chunk at step {step}...")

            actions = predict_chunk()

            print("Predicted action shape:", actions.shape)

        # Select the current action
        action_index = step % CHUNK_SIZE
        action = actions[action_index]

        # Convert 14 ACT actions into 16 MuJoCo controls
        controls = act_to_mujoco(action)

        # Apply action
        data.ctrl[:] = controls

        # Advance physics
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)

        # Capture the current camera frame
        renderer.update_scene(data, camera="top")
        frame = renderer.render().copy()

        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        video.write(frame_bgr)

        # Print progress
        if step % 20 == 0:
            cube_position = data.qpos[-7:-4].copy()

            print(
                f"Step {step}/{TOTAL_STEPS} | "
                f"Cube position: "
                f"{np.round(cube_position, 3)}"
            )

    print("\nACT rollout finished!")

    print(
        "Final cube position:",
        np.round(data.qpos[-7:-4], 3)
    )

finally:
    video.release()
    renderer.close()

print("\nVideo saved:", VIDEO_PATH)
print("Simulation closed.")
