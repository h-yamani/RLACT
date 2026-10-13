
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

# 1. Load our saved ACT predictions
print("Loading ACT predictions...")
results = torch.load(
    "outputs/act_first_prediction.pt",
    map_location="cpu",
    weights_only=True,
)

predicted = results["predicted_actions"][0]

# 2. Load the demonstration dataset
print("Loading expert demonstrations...")
dataset = LeRobotDataset(DATASET_ID)

# 3. Get the corresponding expert actions
# Our prediction started from dataset[0].
# ACT predicted the next 100 actions.
num_steps = predicted.shape[0]

episode_id = int(dataset[0]["episode_index"])
expert_actions = []

for i in range(num_steps):
    sample = dataset[i]

    # Never compare actions from different episodes
    if int(sample["episode_index"]) != episode_id:
        break

    expert_actions.append(sample["action"])

expert = torch.stack(expert_actions)

# Match the number of predicted actions
predicted = predicted[:len(expert)]

# 4. Calculate Mean Absolute Error
mae = torch.mean(torch.abs(predicted - expert))

print("\nEvaluation results")
print("------------------")
print("Predicted actions:", predicted.shape)
print("Expert actions:", expert.shape)
print("Mean Absolute Error:", mae.item())

# 5. Calculate error for each of the 14 action dimensions
per_joint_mae = torch.mean(
    torch.abs(predicted - expert), dim=0
)

print("\nMAE for each action dimension:")
for i, error in enumerate(per_joint_mae):
    print(f"Action {i + 1}: {error.item():.6f}")

print("\nOffline evaluation completed!")
