"""Compute resource tracking: wall-clock time, frozen module stats."""

import time


class ComputeTracker:
    """Track training compute metrics per epoch.

    Records:
        - Wall-clock time per epoch (seconds).
        - Number of frozen modules per epoch.
        - Cumulative training time.
        - Estimated backward FLOPs saved from freezing.
    """

    def __init__(self, total_branch_params=None):
        """
        Args:
            total_branch_params: list of param counts per branch (for FLOP estimation).
        """
        self.epoch_times = []
        self.frozen_counts = []
        self.total_branch_params = total_branch_params or []
        self._epoch_start = None

    def start_epoch(self):
        """Call at the beginning of each training epoch."""
        self._epoch_start = time.time()

    def end_epoch(self, frozen_flags=None):
        """Call at the end of each training epoch.

        Args:
            frozen_flags: list of bool, True if branch k is frozen.
        """
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)

        if frozen_flags is not None:
            self.frozen_counts.append(sum(frozen_flags))
        else:
            self.frozen_counts.append(0)

    def get_summary(self) -> dict:
        """Return summary of compute metrics."""
        total_time = sum(self.epoch_times)
        avg_time = total_time / max(len(self.epoch_times), 1)

        # Estimate backward FLOP savings from freezing
        # (frozen modules skip backward pass, roughly 2x forward FLOPs)
        total_branch_flops = sum(self.total_branch_params) if self.total_branch_params else 0
        frozen_param_epochs = 0
        for i, count in enumerate(self.frozen_counts):
            if self.total_branch_params and count > 0:
                # Sum params of frozen branches at this epoch
                # (Approximation: assume first `count` branches are frozen)
                frozen_param_epochs += sum(
                    self.total_branch_params[:count])

        return {
            'total_training_time_sec': round(total_time, 2),
            'avg_epoch_time_sec': round(avg_time, 2),
            'epoch_times': [round(t, 2) for t in self.epoch_times],
            'frozen_counts_per_epoch': self.frozen_counts,
            'num_epochs': len(self.epoch_times),
            'frozen_param_epochs': frozen_param_epochs,
        }
