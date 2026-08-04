"""Unit tests for split_lora_rank_budget_for_se (the LoRA-rank SE helper).

Mirrors test_slimpajama_proportional.py's SE-budget tests but for LoRA rank
budgets. Verifies:
  - SE=0 returns (r_full, 0) with r_full = total_rank/K
  - Ratio semantics: r_SE / r_expert ≈ SE_ratio
  - Allocation close to total_rank within ±3% after integer rounding
  - API-compat: no impact on existing split_ffn_budget_for_se behaviour
"""
import pytest

from data.slimpajama import split_ffn_budget_for_se, split_lora_rank_budget_for_se


# ── SE=0 degenerate case ────────────────────────────────────────────────────

def test_se0_returns_r_full():
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=320, se_ratio=0.0,
                                               num_experts=20)
    assert r_e == 16
    assert r_se == 0


def test_se0_k7_total_140():
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=140, se_ratio=0.0,
                                               num_experts=7)
    assert r_e == 20
    assert r_se == 0


# ── SE ratio semantics (r_SE / r_expert ≈ SE_ratio) ───────────────────────

def test_se1_roughly_equal_rank():
    """SE_ratio=1.0 → SE rank should equal per-expert rank (within 1)."""
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=320, se_ratio=1.0,
                                               num_experts=20)
    assert abs(r_e - r_se) <= 1


def test_se_half():
    """SE_ratio=0.5 → r_SE ≈ r_expert/2."""
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=320, se_ratio=0.5,
                                               num_experts=20)
    assert abs(r_se - r_e * 0.5) < 1.5


def test_se2_double():
    """SE_ratio=2.0 → r_SE ≈ 2 * r_expert."""
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=320, se_ratio=2.0,
                                               num_experts=20)
    assert abs(r_se - r_e * 2.0) < 2.0


# ── Total rank budget within ±3% ────────────────────────────────────────────

@pytest.mark.parametrize("se_ratio", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_total_budget_within_3pct(se_ratio):
    total_rank = 320
    K = 20
    r_e, r_se = split_lora_rank_budget_for_se(total_rank, se_ratio, K)
    actual_total = K * r_e + r_se
    dev = abs(actual_total - total_rank) / total_rank
    assert dev < 0.03, (f"se_ratio={se_ratio}: K={K} r_e={r_e} r_se={r_se} "
                        f"total={actual_total} (target {total_rank}, "
                        f"deviation {dev:.2%})")


# ── API compat: ffn helper unchanged ────────────────────────────────────────

def test_ffn_helper_behaviour_unchanged():
    """Guardrail: the existing NLP ffn helper must not be perturbed by the
    new LoRA helper being added alongside it (backward compatibility with
    SlimPajama NLP ablation + rtx5090_4 main table)."""
    # Canonical SlimPajama NLP values must be bit-identical to pre-LoRA code.
    assert split_ffn_budget_for_se(1540, 0.0, 7) == (1540, 0)
    assert split_ffn_budget_for_se(1540, 0.5, 7) == (1437, 103)
    assert split_ffn_budget_for_se(1540, 1.0, 7) == (1348, 192)
    # ViT K=46 values for sanity (rtx5090_5 chain)
    total_branch, se = split_ffn_budget_for_se(1536, 1.0, 46)
    assert total_branch + se == pytest.approx(1536, abs=3)


# ── Input validation ─────────────────────────────────────────────────────────

def test_invalid_num_experts():
    with pytest.raises(ValueError):
        split_lora_rank_budget_for_se(total_rank=320, se_ratio=1.0, num_experts=0)


def test_invalid_se_ratio():
    with pytest.raises(ValueError):
        split_lora_rank_budget_for_se(total_rank=320, se_ratio=-0.5, num_experts=20)


# ── Integer type guarantees ─────────────────────────────────────────────────

@pytest.mark.parametrize("se_ratio", [0.0, 0.5, 1.0, 2.0])
def test_returns_ints(se_ratio):
    r_e, r_se = split_lora_rank_budget_for_se(total_rank=320, se_ratio=se_ratio,
                                               num_experts=20)
    assert isinstance(r_e, int)
    assert isinstance(r_se, int)
