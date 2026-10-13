
class EpisodeMetrics:
    """Track reward and success during a robot episode."""

    def __init__(self, success_reward=4.0):
        self.success_reward = success_reward
        self.reset()

    def reset(self):
        self.executed_steps = 0
        self.max_reward = 0.0
        self.final_reward = 0.0
        self.first_success_step = None

    def update(self, reward):
        self.executed_steps += 1
        self.max_reward = max(self.max_reward, reward)
        self.final_reward = reward

        if (
            reward >= self.success_reward
            and self.first_success_step is None
        ):
            self.first_success_step = self.executed_steps

    @property
    def success(self):
        return self.first_success_step is not None
