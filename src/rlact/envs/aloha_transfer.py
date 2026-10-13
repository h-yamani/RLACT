
from pathlib import Path

import mujoco
import numpy as np

from rlact.tasks.base import Task


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "external"
    / "act"
    / "assets"
    / "bimanual_viperx_transfer_cube.xml"
)

START_ARM_POSE = np.array([
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,
])

CUBE_POSE = np.array([
    0.2, 0.5, 0.05, 1, 0, 0, 0
])


class AlohaTransferTask(Task):
    """ALOHA bimanual cube-transfer simulation."""

    def __init__(self, max_steps=400):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.max_steps = max_steps
        self.current_step = 0

    @property
    def name(self):
        return "aloha.transfer_cube"

    def reset(self, seed=None):
        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:16] = START_ARM_POSE
        self.data.qpos[-7:] = CUBE_POSE

        self.data.ctrl[:] = START_ARM_POSE

        mujoco.mj_forward(self.model, self.data)

        return self._observation()

    def _observation(self):
        return {
            "joint_positions": self.data.qpos[:16].copy(),
            "cube_position": self.data.qpos[-7:-4].copy(),
        }

    def _convert_action(self, action):
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (14,):
            raise ValueError("Expected 14-dimensional action")

        controls = np.zeros(self.model.nu)

        controls[0:6] = action[0:6]

        left = np.clip(action[6], 0, 1)
        left_position = 0.01844 + left * (0.05800 - 0.01844)

        controls[6:8] = [left_position, -left_position]

        controls[8:14] = action[7:13]

        right = np.clip(action[13], 0, 1)
        right_position = 0.01844 + right * (0.05800 - 0.01844)

        controls[14:16] = [right_position, -right_position]

        return np.clip(
            controls,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
    
    def _contact_pairs(self):
        """Return MuJoCo contact pairs using original ACT ordering."""
        pairs = set()

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1
            )
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2
            )

            pairs.add((name1, name2))

        return pairs

    def _task_reward(self):
        """Original ACT cube-transfer reward definition."""
        contacts = self._contact_pairs()

        touch_left = (
            "red_box",
            "vx300s_left/10_left_gripper_finger"
        ) in contacts

        touch_right = (
            "red_box",
            "vx300s_right/10_right_gripper_finger"
        ) in contacts

        touch_table = ("red_box", "table") in contacts

        reward = 0

        if touch_right:
            reward = 1

        if touch_right and not touch_table:
            reward = 2

        if touch_left:
            reward = 3

        if touch_left and not touch_table:
            reward = 4

        return reward

    def step(self, action):
        self.data.ctrl[:] = self._convert_action(action)

        # 10 physics steps at 0.002 s = 0.02 s (50 Hz)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        observation = self._observation()

        reward = float(self._task_reward())
        terminated = reward == 4.0
        truncated = (
               self.current_step >= self.max_steps
               and not terminated
        )

        info = {
            "step": self.current_step,
            "success": terminated,
            "task_reward": reward,
        }

        return observation, reward, terminated, truncated, info

    def is_success(self):
        return self._task_reward() == 4
        

    def close(self):
        pass
