import abc
import dataclasses
import math
from bisect import bisect_left, bisect_right

from tinker.types import LossFnType


@dataclasses.dataclass
class LossConfig(abc.ABC):
    name: LossFnType = dataclasses.field(init=False)
    class_name: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.class_name = type(self).__name__

    def config(self, epoch: int) -> dict[str, float]:
        return {}

    def chart_layout(self) -> list[list[str]]:
        """Define chart grouping: each inner list is keys sharing one chart."""
        return [
            ["logprob_decreased", "logprob_increased"],
        ]

    def compute_global_metrics(
        self,
        all_token_diffs: list[float],
        all_epoch_logprobs: list[float],
    ) -> list[list[dict[str, float]]]:
        """Compute global metrics from all token-level diffs across examples."""
        if not all_token_diffs:
            return []
        sorted_diffs = sorted(all_token_diffs)
        n = len(sorted_diffs)
        percentile_chart: list[dict[str, float]] = []
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            idx = min(int(n * p / 100), n - 1)
            percentile_chart.append({f"diff_p{p:02d}": round(sorted_diffs[idx], 6)})
        DIFF_TARGETS = [
            -0.2,
            -0.1,
            -0.05,
            -0.01,
            -0.005,
            0.0,
            0.005,
            0.01,
            0.05,
            0.1,
            0.2,
        ]
        diff2p_chart: list[dict[str, float]] = []
        for d in DIFF_TARGETS:
            lo = bisect_left(sorted_diffs, d)
            hi = bisect_right(sorted_diffs, d)
            pct = round((lo + hi) / (2 * n), 6)
            diff2p_chart.append({f"diff2p_{d}": pct})
        return [percentile_chart, diff2p_chart]

    def compute_metrics(
        self,
        final_logprobs: list[float],
        ref_logprobs: list[float],
        mask: list[int],
        epoch: int,
    ) -> dict[str, float]:
        """Compute per-example post-training metrics.

        mask aligns with logprobs (target token positions): 1=unmasked, 0=masked.
        """
        n = len(final_logprobs)
        assert n == len(ref_logprobs) == len(mask)
        if n == 0:
            return {}
        metrics: dict[str, float] = {}
        idxs = [i for i in range(n) if mask[i] == 1]
        if idxs:
            decreased = sum(1 for i in idxs if final_logprobs[i] < ref_logprobs[i])
            increased = sum(1 for i in idxs if final_logprobs[i] > ref_logprobs[i])
            unchanged = len(idxs) - decreased - increased
            metrics["logprob_decreased"] = round(
                (decreased + unchanged * 0.5) / len(idxs), 6
            )
            metrics["logprob_increased"] = round(
                (increased + unchanged * 0.5) / len(idxs), 6
            )
        return metrics


@dataclasses.dataclass
class CrossEntropyLossConfig(LossConfig):
    name: LossFnType = dataclasses.field(default="cross_entropy", init=False)


@dataclasses.dataclass
class CrossEntropyWithWeightingLossConfig(CrossEntropyLossConfig, abc.ABC):
    name: LossFnType = dataclasses.field(default="cross_entropy", init=False)
    branch_logprob: float  # e.g. 0.01
    first_cutoff_weight: float  # e.g. 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.branch_logprob > 0

    def _branch_weight(self, logprob: float) -> float:
        """Branch scaling weight: min(1, |logprob| / branch_logprob)."""
        return min(1.0, abs(logprob) / self.branch_logprob)

    def _cutoff_weight(
        self, prev_logprob: float, ref_logprob: float, epoch: int
    ) -> float:
        if epoch == 0:
            return self.first_cutoff_weight
        return 1.0

    def apply_weights(
        self,
        float_advantages: list[float],
        prev_logprobs: list[float] | None,
        ref_logprobs: list[float] | None,
        epoch: int,
    ) -> list[float]:
        """Apply branch and cutoff weights to advantages."""
        if prev_logprobs is not None:
            float_advantages = [
                float_advantage * self._branch_weight(prev_logprob)
                for float_advantage, prev_logprob in zip(
                    float_advantages, prev_logprobs
                )
            ]
        if ref_logprobs is not None and prev_logprobs is not None:
            float_advantages = [
                float_advantage * self._cutoff_weight(prev_logprob, ref_logprob, epoch)
                for float_advantage, prev_logprob, ref_logprob in zip(
                    float_advantages, prev_logprobs, ref_logprobs
                )
            ]
        return float_advantages

    def compute_global_metrics(
        self,
        all_token_diffs: list[float],
        all_epoch_logprobs: list[float],
    ) -> list[list[dict[str, float]]]:
        result = super().compute_global_metrics(all_token_diffs, all_epoch_logprobs)
        if not all_token_diffs:
            return result
        sorted_vw = sorted(
            (
                (d, self._branch_weight(lp))
                for d, lp in zip(all_token_diffs, all_epoch_logprobs)
            ),
            key=lambda x: x[0],
        )
        total_w = sum(w for _, w in sorted_vw)
        if total_w > 0:
            weighted_pct_chart: list[dict[str, float]] = []
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                target = total_w * p / 100
                cumulative = 0.0
                val = sorted_vw[-1][0]
                for v, w in sorted_vw:
                    cumulative += w
                    if cumulative >= target:
                        val = v
                        break
                weighted_pct_chart.append({f"weighted_diff_p{p:02d}": round(val, 6)})
            result.insert(1, weighted_pct_chart)
            sorted_diffs_only = [d for d, _ in sorted_vw]
            cum_weights = []
            cumulative = 0.0
            for _, w in sorted_vw:
                cumulative += w
                cum_weights.append(cumulative)
            DIFF_TARGETS = [
                -0.2,
                -0.1,
                -0.05,
                -0.01,
                -0.005,
                0.0,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
            ]
            weighted_diff2p_chart: list[dict[str, float]] = []
            for d in DIFF_TARGETS:
                lo = bisect_left(sorted_diffs_only, d)
                hi = bisect_right(sorted_diffs_only, d)
                cum_lo = cum_weights[lo - 1] if lo > 0 else 0.0
                cum_hi = cum_weights[hi - 1] if hi > 0 else 0.0
                pct = round((cum_lo + cum_hi) / (2 * total_w), 6)
                weighted_diff2p_chart.append({f"weighted_diff2p_{d}": pct})
            result.append(weighted_diff2p_chart)
        return result

    def chart_layout(self) -> list[list[str]]:
        return super().chart_layout() + [
            ["weighted_logprob_decreased", "weighted_logprob_increased"],
            ["avg_branch_weight", "cutoff_avg_weight", "avg_weight"],
            ["branch_mask_fraction"],
        ]

    def compute_metrics(
        self,
        final_logprobs: list[float],
        ref_logprobs: list[float],
        mask: list[int],
        epoch: int,
    ) -> dict[str, float]:
        metrics = super().compute_metrics(final_logprobs, ref_logprobs, mask, epoch)
        n = len(final_logprobs)
        if n == 0:
            return metrics
        idxs = [i for i in range(n) if mask[i] == 1]
        if not idxs:
            return metrics
        branch_scaled = sum(
            1 for i in idxs if abs(final_logprobs[i]) <= self.branch_logprob
        )
        metrics["branch_mask_fraction"] = round(branch_scaled / len(idxs), 6)
        branch_weights = [self._branch_weight(final_logprobs[i]) for i in idxs]
        metrics["avg_branch_weight"] = round(sum(branch_weights) / len(idxs), 6)
        total_w = sum(branch_weights)
        if total_w > 0:
            w_dec = sum(
                self._branch_weight(final_logprobs[i])
                for i in idxs
                if final_logprobs[i] < ref_logprobs[i]
            )
            w_inc = sum(
                self._branch_weight(final_logprobs[i])
                for i in idxs
                if final_logprobs[i] > ref_logprobs[i]
            )
            w_unch = total_w - w_dec - w_inc
            metrics["weighted_logprob_decreased"] = round(
                (w_dec + w_unch * 0.5) / total_w, 6
            )
            metrics["weighted_logprob_increased"] = round(
                (w_inc + w_unch * 0.5) / total_w, 6
            )
        weights = [
            self._cutoff_weight(final_logprobs[i], ref_logprobs[i], epoch) for i in idxs
        ]
        metrics["cutoff_avg_weight"] = round(sum(weights) / len(idxs), 6)
        metrics["avg_weight"] = round(
            sum(bw * cw for bw, cw in zip(branch_weights, weights)) / len(idxs), 6
        )
        return metrics


@dataclasses.dataclass
class ImportanceSamplingLossConfig(LossConfig):
    name: LossFnType = dataclasses.field(default="importance_sampling", init=False)

    def chart_layout(self) -> list[list[str]]:
        return super().chart_layout() + [
            ["kl_per_token"],
            ["mean_importance_ratio"],
        ]

    def compute_metrics(
        self,
        final_logprobs: list[float],
        ref_logprobs: list[float],
        mask: list[int],
        epoch: int,
    ) -> dict[str, float]:
        metrics = super().compute_metrics(final_logprobs, ref_logprobs, mask, epoch)
        n = len(final_logprobs)
        assert n == len(ref_logprobs) == len(mask)
        idxs = [i for i in range(n) if mask[i] == 1]
        if idxs:
            log_ratios = [final_logprobs[i] - ref_logprobs[i] for i in idxs]
            metrics["kl_per_token"] = round(sum(log_ratios) / len(idxs), 6)
            ratios = [math.exp(min(lr, 20)) for lr in log_ratios]
            metrics["mean_importance_ratio"] = round(sum(ratios) / len(idxs), 6)
        return metrics


@dataclasses.dataclass
class ClipLossConfig(LossConfig, abc.ABC):
    clip_low: float
    clip_high: float

    def chart_layout(self) -> list[list[str]]:
        return super().chart_layout() + [
            ["kl_per_token"],
            ["mean_importance_ratio"],
            ["clip_fraction_low", "clip_fraction_high"],
        ]

    def config(self, epoch: int) -> dict[str, float]:
        if epoch == 0:
            return {"clip_low_threshold": 1.0, "clip_high_threshold": 1.0}
        return {
            "clip_low_threshold": self.clip_low,
            "clip_high_threshold": self.clip_high,
        }

    def compute_metrics(
        self,
        final_logprobs: list[float],
        ref_logprobs: list[float],
        mask: list[int],
        epoch: int,
    ) -> dict[str, float]:
        metrics = super().compute_metrics(final_logprobs, ref_logprobs, mask, epoch)
        n = len(final_logprobs)
        assert n == len(ref_logprobs) == len(mask)
        idxs = [i for i in range(n) if mask[i] == 1]
        if idxs:
            log_ratios = [final_logprobs[i] - ref_logprobs[i] for i in idxs]
            metrics["kl_per_token"] = round(sum(log_ratios) / len(idxs), 6)
            ratios = [math.exp(min(lr, 20)) for lr in log_ratios]
            metrics["mean_importance_ratio"] = round(sum(ratios) / len(idxs), 6)
            low = 1 - self.clip_low
            high = 1 + self.clip_high
            clipped_low = sum(1 for r in ratios if r < low)
            clipped_high = sum(1 for r in ratios if r > high)
            metrics["clip_fraction_low"] = round(clipped_low / len(idxs), 6)
            metrics["clip_fraction_high"] = round(clipped_high / len(idxs), 6)
        return metrics


@dataclasses.dataclass
class PPOLossConfig(ClipLossConfig):
    clip_low: float  # e.g. 0.2
    clip_high: float  # e.g. 0.2
    name: LossFnType = dataclasses.field(default="ppo", init=False)


@dataclasses.dataclass
class CISPOLossConfig(ClipLossConfig):
    clip_low: float  # e.g. 0.8
    clip_high: float  # e.g. 1.2
    name: LossFnType = dataclasses.field(default="cispo", init=False)


@dataclasses.dataclass
class DROLossConfig(LossConfig):
    beta: float  # e.g. 0.05
    name: LossFnType = dataclasses.field(default="dro", init=False)

    def chart_layout(self) -> list[list[str]]:
        return super().chart_layout() + [
            ["kl_per_token"],
            ["dro_penalty_per_token"],
        ]

    def config(self, epoch: int) -> dict[str, float]:
        if epoch == 0:
            return {"beta": 0.0}
        return {"beta": self.beta}

    def compute_metrics(
        self,
        final_logprobs: list[float],
        ref_logprobs: list[float],
        mask: list[int],
        epoch: int,
    ) -> dict[str, float]:
        metrics = super().compute_metrics(final_logprobs, ref_logprobs, mask, epoch)
        n = len(final_logprobs)
        assert n == len(ref_logprobs) == len(mask)
        idxs = [i for i in range(n) if mask[i] == 1]
        if idxs:
            log_ratios = [final_logprobs[i] - ref_logprobs[i] for i in idxs]
            metrics["kl_per_token"] = round(sum(log_ratios) / len(idxs), 6)
        sq_diffs = [(final_logprobs[i] - ref_logprobs[i]) ** 2 for i in range(n)]
        metrics["dro_penalty_per_token"] = round(self.beta * 0.5 * sum(sq_diffs) / n, 6)
        return metrics
