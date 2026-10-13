
from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class EpisodeResult:
    """Standard evaluation result for one robot episode."""

    task: str
    policy: str
    seed: int
    max_steps: int
    executed_steps: int
    max_reward: float
    success: bool
    first_success_step: int | None
    final_reward: float

    def save_json(self, path: str | Path) -> None:
        """Save the episode result as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)
