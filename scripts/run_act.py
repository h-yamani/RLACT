
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"

print("Loading dataset...")
dataset = LeRobotDataset(DATASET_ID)

print("Loading pretrained ACT...")
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy.eval()

# Statistics used during ACT training
stats = dataset.meta.stats

def normalize(x, key):
    mean = torch.as_tensor(stats[key]["mean"], dtype=x.dtype)
    std = torch.as_tensor(stats[key]["std"], dtype=x.dtype)
    return (x - mean) / std.clamp(min=1e-6)

def unnormalize(x, key):
    mean = torch.as_tensor(stats[key]["mean"], dtype=x.dtype)
    std = torch.as_tensor(stats[key]["std"], dtype=x.dtype)
    return x * std + mean

# First recorded observation
sample = dataset[0]

# Prepare image and robot state with batch dimension
image = sample["observation.images.top"].unsqueeze(0)
state = sample["observation.state"].unsqueeze(0)

batch = {
    "observation.images.top": normalize(
        image, "observation.images.top"
    ),
    "observation.state": normalize(
        state, "observation.state"
    ),
}

# Predict an entire action chunk
print("Running ACT inference on CPU...")

with torch.inference_mode():
    predicted = policy.predict_action_chunk(batch)

# Convert predicted actions to original robot units
predicted = unnormalize(predicted, "action")

print("\nPredicted action chunk shape:", predicted.shape)
print("\nFirst predicted action:")
print(predicted[0, 0])

print("\nExpert action from dataset:")
print(sample["action"])

print("\nACT inference completed!")
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

torch.save(
    {
        "predicted_actions": predicted.cpu(),
        "expert_action": sample["action"].cpu(),
    },
    "outputs/act_first_prediction.pt",
)

print("Predictions saved successfully!")
