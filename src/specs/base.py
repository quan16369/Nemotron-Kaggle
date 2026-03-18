from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PuzzleSpec(ABC):
    name: str = "base"

    @abstractmethod
    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def render(self, config: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def solve(self, config: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not implement reproduce()")