
from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Common interface for robot manipulation tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier."""
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the task and return its initial observation."""
        pass

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one action.

        Returns:
            observation, reward, terminated, truncated, info
        """
        pass

    @abstractmethod
    def is_success(self) -> bool:
        """Check whether the task objective has been achieved."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""
        pass
