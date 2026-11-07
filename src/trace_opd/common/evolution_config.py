"""Configuration for evidence-gated, within-session Skill evolution."""
import math
from dataclasses import dataclass, field


@dataclass
class EvolutionCfg:
    enabled: bool = False
    initial_skill: str = ""
    # These augment the fixed executor scaffold, never the Skill itself.
    discovery_prompts: list[str] = field(default_factory=lambda: [""])
    validation_prompt: str = ""
    samples: int = 4
    max_rounds: int = 3
    seed: int = 1729
    min_error_reduction: float = 0.01
    min_success_gain: float = 0.01
    max_holdout_regression: float = 0.0
    # JSONL of dataset extra_info objects; must exclude the current task.
    holdout_tasks: str = ""
    require_holdout: bool = True

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.require_holdout) is not bool:
            raise ValueError("evolution switches must be booleans")
        for name in ("samples", "max_rounds", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"evolution.{name} must be a positive integer (seed >= 0)")
        if self.samples > 64 or self.max_rounds > 100 or self.seed >= 2**31 - 100000:
            raise ValueError("evolution sampling/round/seed limit exceeded")
        if (not isinstance(self.discovery_prompts, list)
                or not 1 <= len(self.discovery_prompts) <= 2
                or any(not isinstance(p, str) for p in self.discovery_prompts)):
            raise ValueError("evolution.discovery_prompts requires one or two strings")
        for name in ("initial_skill", "validation_prompt", "holdout_tasks"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"evolution.{name} must be a string")
        for name in ("min_error_reduction", "min_success_gain", "max_holdout_regression"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"evolution.{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"evolution.{name} must be in [0, 1]")
        if self.min_error_reduction <= 0 or self.min_success_gain <= 0:
            raise ValueError("evolution requires positive error reduction and success gain")
