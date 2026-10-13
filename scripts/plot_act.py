
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load saved ACT predictions
results = torch.load(
    "outputs/act_first_prediction.pt",
    map_location="cpu",
    weights_only=True,
)

predicted = results["predicted_actions"][0].numpy()

# Load corresponding expert actions
dataset = LeRobotDataset(
    "lerobot/aloha_sim_transfer_cube_human"
)

episode_id = int(dataset[0]["episode_index"])
expert_actions = []

for i in range(len(predicted)):
    sample = dataset[i]

    if int(sample["episode_index"]) != episode_id:
        break

    expert_actions.append(sample["action"])

expert = torch.stack(expert_actions).numpy()
predicted = predicted[:len(expert)]

# Create plots for all 14 action dimensions
fig, axes = plt.subplots(
    7, 2, figsize=(14, 18), sharex=True
)

for j, ax in enumerate(axes.flat):
    ax.plot(
        expert[:, j],
        label="Expert",
        linewidth=2,
    )
    ax.plot(
        predicted[:, j],
        label="ACT prediction",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_title(f"Action dimension {j + 1}")
    ax.set_ylabel("Action value")
    ax.grid(True, alpha=0.3)

    if j == 0:
        ax.legend()

for ax in axes[-1]:
    ax.set_xlabel("Timestep")

fig.suptitle(
    "ACT Predictions vs Expert Demonstration",
    fontsize=16,
)

plt.tight_layout()

# Save figure
Path("outputs").mkdir(exist_ok=True)
plt.savefig(
    "outputs/act_vs_expert.png",
    dpi=150,
    bbox_inches="tight",
)

print("Plot saved to outputs/act_vs_expert.png")

plt.show()
