
from pathlib import Path

import mujoco
import numpy as np

from rlact.envs.aloha_transfer import AlohaTransferTask
from rlact.policies.act.adapter import ACTAdapter
from rlact.evaluation.metrics import EpisodeMetrics
from rlact.evaluation.results import EpisodeResult


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_STEPS = 400
CHUNK_SIZE = 100
SEED = 42

POLICY_NAME = "ACT"
POLICY_ID = "lerobot/act_aloha_sim_transfer_cube_human"

GRIPPER_CLOSE = 0.01844
GRIPPER_OPEN = 0.05800

OUTPUT_DIR = Path("outputs/benchmarks")


# ============================================================
# 2. ROBOT OBSERVATIONS
# ============================================================

def get_robot_state(task):
    """Convert MuJoCo joint positions to ACT's 14D state."""

    qpos = task.data.qpos

    state = np.concatenate([
        qpos[0:6],
        [
            (qpos[6] - GRIPPER_CLOSE)
            / (GRIPPER_OPEN - GRIPPER_CLOSE)
        ],
        qpos[8:14],
        [
            (qpos[14] - GRIPPER_CLOSE)
            / (GRIPPER_OPEN - GRIPPER_CLOSE)
        ],
    ])

    return state


def get_camera_image(task, renderer):
    """Capture an RGB image from the top camera."""

    renderer.update_scene(task.data, camera="top")

    return renderer.render().copy()


# ============================================================
# 3. BENCHMARK
# ============================================================

def main():

    print("Initializing RLACT benchmark...")

    task = AlohaTransferTask(max_steps=MAX_STEPS)
    renderer = None

    try:
        # ----------------------------------------------------
        # Initialize environment
        # ----------------------------------------------------

        task.reset(seed=SEED)

        renderer = mujoco.Renderer(
            task.model,
            height=480,
            width=640,
        )

        # ----------------------------------------------------
        # Initialize pretrained ACT policy
        # ----------------------------------------------------

        adapter = ACTAdapter(
            policy_id=POLICY_ID,
            device="cpu",
        )

        adapter.reset()

        # ----------------------------------------------------
        # Initialize metrics
        # ----------------------------------------------------

        metrics = EpisodeMetrics(success_reward=4.0)

        actions = None

        print("\nStarting ACT benchmark...")
        print("Task:", task.name)
        print("Policy:", POLICY_NAME)
        print("Seed:", SEED)
        print("Maximum steps:", MAX_STEPS)

        # ====================================================
        # 4. ACT ROLLOUT
        # ====================================================

        for step in range(MAX_STEPS):

            # Predict a new chunk every 100 control steps
            if step % CHUNK_SIZE == 0:

                image = get_camera_image(
                    task,
                    renderer,
                )

                state = get_robot_state(task)

                actions = adapter.predict_chunk(
                    image,
                    state,
                )

                if actions.shape != (CHUNK_SIZE, 14):
                    raise ValueError(
                        "Unexpected ACT action shape: "
                        f"{actions.shape}"
                    )

                print(
                    f"\nPredicted ACT chunk at step {step}: "
                    f"{actions.shape}"
                )

            # Select one action from the current chunk
            action = actions[step % CHUNK_SIZE]

            # Execute action in MuJoCo
            observation, reward, terminated, truncated, info = (
                task.step(action)
            )

            # ------------------------------------------------
            # Track metrics
            # ------------------------------------------------

            metrics.update(reward)

            # Print every 20 steps or on first success
            if (
                step % 20 == 0
                or (
                    metrics.first_success_step
                    == metrics.executed_steps
                )
            ):

                print(
                    f"Step {metrics.executed_steps:3d}/{MAX_STEPS} | "
                    f"Reward: {reward:.0f} | "
                    f"Max: {metrics.max_reward:.0f} | "
                    f"Cube: "
                    f"{np.round(observation['cube_position'], 3)}"
                )

            # Continue after success to measure final reward
            if truncated:
                break

        # ====================================================
        # 5. CREATE STANDARD EPISODE RESULT
        # ====================================================

        result = EpisodeResult(
            task=task.name,
            policy=POLICY_NAME,
            seed=SEED,
            max_steps=MAX_STEPS,
            executed_steps=metrics.executed_steps,
            max_reward=float(metrics.max_reward),
            success=bool(metrics.success),
            first_success_step=metrics.first_success_step,
            final_reward=float(metrics.final_reward),
        )

        # ====================================================
        # 6. DISPLAY BENCHMARK RESULTS
        # ====================================================

        print("\n========== RLACT BENCHMARK ==========")
        print("Task:", result.task)
        print("Policy:", result.policy)
        print("Seed:", result.seed)
        print("Executed steps:", result.executed_steps)
        print("Maximum reward:", result.max_reward)
        print("Success:", result.success)
        print("First success step:", result.first_success_step)
        print("Final reward:", result.final_reward)
        print("=====================================")

        # ====================================================
        # 7. SAVE RESULTS
        # ====================================================

        output_path = (
            OUTPUT_DIR / f"act_transfer_seed{SEED}.json"
        )

        result.save_json(output_path)

        print("\nResults saved to:", output_path)

    finally:
        # ====================================================
        # 8. CLEANUP
        # ====================================================

        if renderer is not None:
            renderer.close()

        task.close()

        print("Simulation closed.")


if __name__ == "__main__":
    main()
