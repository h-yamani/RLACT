
from lerobot.policies.act.modeling_act import ACTPolicy

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"

print("Loading pretrained ACT policy...")

# Download pretrained weights from Hugging Face
policy = ACTPolicy.from_pretrained(MODEL_ID)

# Use evaluation mode
policy.eval()

print("\nACT policy loaded successfully!")

# Inspect the model configuration
print("\nPolicy configuration:")
print(policy.config)

# Inspect the neural network architecture
print("\nACT architecture:")
print(policy)
