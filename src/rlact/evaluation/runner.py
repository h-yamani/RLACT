
from rlact.evaluation.metrics import EpisodeMetrics
from rlact.evaluation.results import EpisodeResult


class EvaluationRunner:
    """Evaluate robot policies using a shared task interface."""

    def __init__(
        self,
        max_steps=400,
        success_reward=4.0,
        continue_after_success=True,
    ):
        self.max_steps = max_steps
        self.success_reward = success_reward
        self.continue_after_success = continue_after_success

    def evaluate(self, task, policy, seed=42):
        """Run one evaluation episode and return its metrics."""

        task.reset(seed=seed)
        policy.reset()

        metrics = EpisodeMetrics(
            success_reward=self.success_reward
        )

        for _ in range(self.max_steps):

            # Request one action from the policy
            action = policy.act(task)

            # Execute the action in the environment
            observation, reward, terminated, truncated, info = (
                task.step(action)
            )

            # Update shared evaluation metrics
            metrics.update(reward)

            # Stop at the time limit
            if truncated:
                break

            # Optionally stop immediately after success
            if terminated and not self.continue_after_success:
                break

        return EpisodeResult(
            task=task.name,
            policy=policy.name,
            seed=seed,
            max_steps=self.max_steps,
            executed_steps=metrics.executed_steps,
            max_reward=float(metrics.max_reward),
            success=metrics.success,
            first_success_step=metrics.first_success_step,
            final_reward=float(metrics.final_reward),
        )
