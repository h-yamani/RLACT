
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Public robot demonstration dataset on Hugging Face
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

print("Loading dataset from Hugging Face...")

dataset = LeRobotDataset(DATASET_ID)

print("\nDataset loaded successfully!")
print("Number of frames:", len(dataset))
print("Number of episodes:", dataset.num_episodes)

# Inspect the first demonstration frame
sample = dataset[0]

print("\nAvailable observation and action fields:")
for key, value in sample.items():
    print(f"{key}: {getattr(value, 'shape', type(value))}")

import matplotlib.pyplot as plt

# Get the camera image from the first frame
image = sample["observation.images.top"]

# PyTorch format: [Channels, Height, Width]
# Matplotlib format: [Height, Width, Channels]
image = image.permute(1, 2, 0)

plt.imshow(image)
plt.title("ALOHA Robot - First Camera Frame")
plt.axis("off")
plt.show()

