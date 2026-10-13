
import torch
import mujoco
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


class ACTAdapter:
    """Adapter for pretrained LeRobot ACT policies."""

    def __init__(
        self,
        policy_id="lerobot/act_aloha_sim_transfer_cube_human",
        dataset_id="lerobot/aloha_sim_transfer_cube_human",
        device="cpu",
    ):
        self.device = torch.device(device)

        print("Loading ACT dataset statistics...")
        dataset = LeRobotDataset(dataset_id)
        self.stats = dataset.meta.stats

        print("Loading pretrained ACT policy...")
        self.policy = ACTPolicy.from_pretrained(policy_id)
        self.policy = self.policy.to(self.device)
        self.policy.eval()

        # Action chunk state
        self.actions = None
        self.action_index = 0

        # MuJoCo renderer, created by attach()
        self.renderer = None

    # ========================================================
    # 1. POLICY INFORMATION
    # ========================================================

    @property
    def name(self):
        return "ACT"

    # ========================================================
    # 2. RESET
    # ========================================================

    def reset(self):
        """Reset ACT state and clear the current action chunk."""

        self.policy.reset()
        self.actions = None
        self.action_index = 0

    # ========================================================
    # 3. NORMALIZATION
    # ========================================================

    def _normalize(self, value, key):
        """Apply the normalization used in our working ACT rollout."""

        if key == "observation.images.top":

            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1)

            std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1)

        else:

            mean = torch.as_tensor(
                self.stats[key]["mean"],
                dtype=torch.float32,
                device=self.device,
            )

            std = torch.as_tensor(
                self.stats[key]["std"],
                dtype=torch.float32,
                device=self.device,
            ).clamp(min=1e-6)

        return (value - mean) / std

    # ========================================================
    # 4. ACT INFERENCE
    # ========================================================

    def predict_chunk(self, image, state):
        """
        Predict a chunk of robot actions.

        Parameters
        ----------
        image:
            RGB numpy array, shape (480, 640, 3).

        state:
            Robot joint positions with normalized grippers,
            shape (14,).

        Returns
        -------
        numpy.ndarray:
            Predicted action chunk, typically (100, 14).
        """

        # Convert RGB image to PyTorch tensor
        image_tensor = (
            torch.from_numpy(image.copy())
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        # Convert robot state to tensor
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # Prepare normalized observations
        batch = {
            "observation.images.top": self._normalize(
                image_tensor,
                "observation.images.top",
            ),
            "observation.state": self._normalize(
                state_tensor,
                "observation.state",
            ),
        }

        # Predict action chunk
        with torch.inference_mode():
            predicted = self.policy.predict_action_chunk(batch)

        # Retrieve action normalization statistics
        mean = torch.as_tensor(
            self.stats["action"]["mean"],
            dtype=torch.float32,
            device=self.device,
        )

        std = torch.as_tensor(
            self.stats["action"]["std"],
            dtype=torch.float32,
            device=self.device,
        )

        # Convert predicted normalized actions
        # back to the original action space
        actions = predicted * std + mean

        return actions[0].cpu().numpy()

    # ========================================================
    # 5. ATTACH TO MUJOCO ENVIRONMENT
    # ========================================================

    def attach(self, task):
        """Create a camera renderer for the environment."""

        if self.renderer is not None:
            self.renderer.close()

        self.renderer = mujoco.Renderer(
            task.model,
            height=480,
            width=640,
        )

        self.actions = None
        self.action_index = 0

    # ========================================================
    # 6. SINGLE-ACTION INTERFACE
    # ========================================================

    def act(self, task):
        """
        Return one robot action.

        A new action chunk is predicted whenever the
        previous chunk has been fully executed.
        """

        if self.renderer is None:
            raise RuntimeError(
                "Call adapter.attach(task) before act()."
            )

        # Predict a new chunk when necessary
        if (
            self.actions is None
            or self.action_index >= len(self.actions)
        ):

            # Capture top-camera image
            self.renderer.update_scene(
                task.data,
                camera="top",
            )

            image = self.renderer.render().copy()

            # Read MuJoCo joint positions
            qpos = task.data.qpos

            gripper_close = 0.01844
            gripper_open = 0.05800

            # Convert MuJoCo state to ACT's 14D state
            state = np.concatenate([
                qpos[0:6],
                [
                    (qpos[6] - gripper_close)
                    / (gripper_open - gripper_close)
                ],
                qpos[8:14],
                [
                    (qpos[14] - gripper_close)
                    / (gripper_open - gripper_close)
                ],
            ])

            # Predict next action chunk
            self.actions = self.predict_chunk(
                image,
                state,
            )

            if (
                self.actions.ndim != 2
                or self.actions.shape[1] != 14
                or len(self.actions) == 0
            ):
                raise ValueError(
                    "Unexpected ACT action chunk shape: "
                    f"{self.actions.shape}"
                )

            self.action_index = 0

        # Return the next action from the chunk
        action = self.actions[self.action_index].copy()

        self.action_index += 1

        return action

    # ========================================================
    # 7. CLEANUP
    # ========================================================

    def close(self):
        """Release MuJoCo renderer resources."""

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        self.actions = None
        self.action_index = 0
