
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
EPISODE_INDEX = 0
FPS = 50

print("Loading dataset...")
dataset = LeRobotDataset(DATASET_ID)

episode = dataset.meta.episodes[EPISODE_INDEX]
start = int(episode["dataset_from_index"])
end = int(episode["dataset_to_index"])

print(f"Episode {EPISODE_INDEX}: {end - start} frames")

Path("outputs").mkdir(exist_ok=True)
output_path = "outputs/aloha_episode_0.mp4"

video = None

for i in range(start, end):
    sample = dataset[i]

    # Convert PyTorch image [C, H, W] to NumPy [H, W, C]
    image = sample["observation.images.top"]
    frame = image.permute(1, 2, 0).cpu().numpy()

    # Convert image from [0, 1] to [0, 255]
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    # OpenCV uses BGR instead of RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if video is None:
        height, width = frame.shape[:2]
        video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (width, height),
        )

        if not video.isOpened():
            raise RuntimeError("Could not create MP4 video")

    video.write(frame)

    if (i - start + 1) % 100 == 0:
        print(f"Processed {i - start + 1} frames")

if video is not None:
    video.release()

print(f"\nVideo saved: {output_path}")
print("Duration: 8 seconds")
