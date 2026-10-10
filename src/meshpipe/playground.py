from dataclasses import dataclass


@dataclass(frozen=True)
class Gesture:
    name: str
    timestamp: float


class GestureController:
    def __init__(self, threshold: float = 0.6, release: float = 0.3, cooldown: float = 0.65):
        if not 0 <= release < threshold <= 1 or cooldown < 0:
            raise ValueError("Invalid gesture thresholds")
        self.threshold = threshold
        self.release = release
        self.cooldown = cooldown
        self.active: set[str] = set()
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.active.clear()

    def update(self, scores: dict[str, float], tracked: bool, clock: float) -> list[Gesture]:
        if not tracked:
            self.reset()
            return []
        left, right = scores.get("eyeBlinkLeft", 0), scores.get("eyeBlinkRight", 0)
        values = {
            "Left wink": left if right < self.release else 0,
            "Right wink": right if left < self.release else 0,
            "Open mouth": scores.get("jawOpen", 0),
            "Smile": (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2,
        }
        events = []
        for name, value in values.items():
            if value < self.release:
                self.active.discard(name)
            elif value >= self.threshold and name not in self.active:
                self.active.add(name)
                if clock - self.last.get(name, float("-inf")) >= self.cooldown:
                    self.last[name] = clock
                    events.append(Gesture(name, clock))
        return events


class ExpressionGame:
    targets = ("Smile", "Open mouth", "Left wink", "Right wink")

    def __init__(self):
        self.score = 0
        self.started: float | None = None
        self.last_hit: float | None = None
        self.best_seconds: float | None = None

    @property
    def target(self) -> str:
        return self.targets[self.score % len(self.targets)]

    def hit(self, gesture: str, clock: float) -> bool:
        if gesture != self.target:
            return False
        if self.started is None:
            self.started = clock
        if self.last_hit is not None:
            response = clock - self.last_hit
            self.best_seconds = (
                response if self.best_seconds is None else min(self.best_seconds, response)
            )
        self.last_hit = clock
        self.score += 1
        return True

    def reset(self) -> None:
        self.__init__()
