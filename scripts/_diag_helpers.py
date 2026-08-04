"""Shared safety helpers for diagnose / analyze / eval scripts.

Two recurring failure modes that previously produced silently-wrong results:

1. **Step-mode warmup never advanced.** Setting `algorithm.current_epoch` is
   a no-op when `warmup_unit='step'` (LoRA + NLP-500M convention). The
   `_warmup_progress()` reads `current_step / total_warmup_steps` which
   stays at 0 → mask returned is uniform-start-of-warmup, NOT the trained
   terminal mask. Per-branch ablation analysis becomes meaningless because
   the baseline mask is itself uniform.

   Fix: `advance_softspecdrop_to_terminal()` advances BOTH current_epoch and
   current_step, then asserts `_warmup_progress() == 1.0` post-advance.
   Refuses to proceed if the assertion fails.

2. **Silent cfg-key fallback.** `.get(key, default)` masks config corruption:
   if the trained model used `num_experts=10` but the cfg key got renamed,
   the diagnose script would silently use 20 (the default) and produce a
   misshapen ablation matrix.

   Fix: `require_keys()` raises KeyError listing missing keys, with a
   context string so the failure points at the section that needed the key.

Both fail loudly by design — the cost of a wrong analysis figure surviving
into the paper is much higher than the cost of an early crash.
"""
from __future__ import annotations

from typing import Iterable


def advance_softspecdrop_to_terminal(algorithm, total_epochs: int) -> None:
    """Advance a SoftSpecDrop algorithm to its end-of-training mask state.

    Handles both `warmup_unit='epoch'` (CIFAR/ViT/legacy NLP convention) and
    `'step'` (LoRA, NLP 500M convention). Setting only `current_epoch` is a
    silent no-op for step-mode → mask stays at uniform-start (cf. 2026-05-02
    LoRA spec bug, where this produced a 0/15 diagonal-hits artifact instead
    of the real trained-mask ablation matrix).

    Args:
        algorithm: a `SoftSpecDrop` instance, OR `None` (no-op).
        total_epochs: trained-config epoch count (used for current_epoch
            advance under epoch mode).

    Raises:
        RuntimeError: if `_warmup_progress()` is not 1.0 after advance —
            indicates a future warmup mode this helper hasn't been updated to
            handle. Refusing to proceed prevents the script from silently
            running on a non-terminal mask.
    """
    if algorithm is None:
        return
    if hasattr(algorithm, 'current_epoch'):
        algorithm.current_epoch = max(int(total_epochs), 1)
    # Step-mode also requires current_step / total_warmup_steps; values are
    # clamped to ≤ 1.0 by min(...) inside SoftSpecDrop._warmup_progress, so
    # setting (1, 1) is the cleanest "definitely past warmup" signal.
    if hasattr(algorithm, 'set_total_steps'):
        algorithm.set_total_steps(1)
    if hasattr(algorithm, 'current_step'):
        algorithm.current_step = 1
    if hasattr(algorithm, '_warmup_progress'):
        progress = algorithm._warmup_progress()
        if progress < 1.0 - 1e-9:
            raise RuntimeError(
                f'algorithm._warmup_progress()={progress:.4f} after advance '
                f'— expected 1.0. warmup_unit='
                f'{getattr(algorithm, "warmup_unit", "?")!r}, '
                f'current_epoch={getattr(algorithm, "current_epoch", "?")}, '
                f'current_step={getattr(algorithm, "current_step", "?")}, '
                f'total_warmup_steps='
                f'{getattr(algorithm, "total_warmup_steps", "?")}. '
                f'Refusing to proceed: any analysis would measure a '
                f'non-terminal mask state and silently mislead.')


def require_keys(d: dict, required: Iterable[str], context: str) -> None:
    """Raise KeyError if any of `required` keys are missing in dict `d`.

    Use this on cfg sections where silent `.get(default)` would mask config
    corruption AND produce a wrong-but-runnable analysis output. Reserve for
    keys whose absence indicates a real configuration problem (algorithm
    type, model arch params, data root) — not for optional knobs with
    sensible fallbacks.

    Args:
        d: dict to check.
        required: keys that must be present.
        context: human-readable context (e.g., 'cfg["algorithm"] in
            outputs/foo/_tmp.yaml') prepended to the error message.

    Raises:
        KeyError: with explicit list of missing keys.
    """
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(
            f'{context}: missing required keys {missing}. Refusing silent '
            f'defaults — they would produce wrong analysis without erroring.')
