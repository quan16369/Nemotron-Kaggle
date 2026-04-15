import dataclasses


@dataclasses.dataclass
class LRSchedule:
    learning_rate: float = 2e-5
    class_name: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.class_name = type(self).__name__

    def get_lr(
        self, step: int, total_steps: int, epoch: int, total_epochs: int
    ) -> float:
        return self.learning_rate


@dataclasses.dataclass
class LinearDecayLRSchedule(LRSchedule):
    final_learning_rate: float = 1e-5

    def get_lr(
        self, step: int, total_steps: int, epoch: int, total_epochs: int
    ) -> float:
        mult = min(1.0, max(0.0, 1.0 - epoch / (1 + total_epochs)))
        return (
            self.final_learning_rate
            + (self.learning_rate - self.final_learning_rate) * mult
        )


@dataclasses.dataclass
class StepLinearDecayLRSchedule(LRSchedule):
    def get_lr(
        self, step: int, total_steps: int, epoch: int, total_epochs: int
    ) -> float:
        return self.learning_rate * (1 - step / total_steps)
