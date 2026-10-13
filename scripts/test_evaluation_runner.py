
from pathlib import Path

from rlact.envs.aloha_transfer import AlohaTransferTask
from rlact.policies.act.adapter import ACTAdapter
from rlact.evaluation.runner import EvaluationRunner


def main():

    print("Initializing RLACT EvaluationRunner...")

    task = AlohaTransferTask(max_steps=400)
    policy = None

    try:
        # 1. Load pretrained ACT policy
        policy = ACTAdapter(
            policy_id="lerobot/act_aloha_sim_transfer_cube_human",
            device="cpu",
        )

        # 2. Attach policy to MuJoCo environment
        policy.attach(task)

        # 3. Create reusable evaluation runner
        runner = EvaluationRunner(
            max_steps=400,
            success_reward=4.0,
            continue_after_success=True,
        )

        # 4. Evaluate ACT
        print("\nStarting evaluation...")

        result = runner.evaluate(
            task=task,
            policy=policy,
            seed=42,
        )

        # 5. Display results
        print("\n========== RLACT EVALUATION ==========")
        print("Task:", result.task)
        print("Policy:", result.policy)
        print("Seed:", result.seed)
        print("Executed steps:", result.executed_steps)
        print("Maximum reward:", result.max_reward)
        print("Success:", result.success)
        print("First success step:", result.first_success_step)
        print("Final reward:", result.final_reward)
        print("======================================")

        # 6. Save results
        output_path = Path(
            "outputs/benchmarks/act_runner_seed42.json"
        )

        result.save_json(output_path)

        print("\nResults saved to:", output_path)

    finally:
        if policy is not None:
            policy.close()

        task.close()

        print("Evaluation resources closed.")


if __name__ == "__main__":
    main()
