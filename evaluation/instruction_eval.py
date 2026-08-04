"""SuperNI held-out evaluation for the LoRA post-train track.

SuperNIEvaluator: greedy-decode each held-out task's instances, score
with ROUGE-L + exact match (Wang 2022 Sec 5 evaluation protocol).

The evaluator wraps a BaseLoRAModel (or any HF CausalLM); it doesn't
depend on the trainer. Imports of transformers / rouge-score / datasets
are deferred so this module loads in bare torch envs.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch


def _lazy_rouge():
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            "rouge-score required for SuperNI eval; "
            "install with `pip install rouge-score`.") from e
    return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# ══════════════════════════════════════════════════════════════════════════
# SuperNI held-out evaluator
# ══════════════════════════════════════════════════════════════════════════


class SuperNIEvaluator:
    """Greedy decode each held-out instance, score with ROUGE-L + EM.

    Args:
        model:      a BaseLoRAModel (its .base is the HF CausalLM).
        tokenizer:  HF AutoTokenizer.
        eval_loader: DataLoader over SuperNI test split.
        max_new_tokens: generation budget per instance.
        device:     'cuda' / 'cpu'.
        routing_fn: callable(cluster_id) → routing dict, or None for
                    methods that ignore routing (single_lora, hydra_lora,
                    lora_moe).
        mask_scale_fn: callable() → float, or None. Used by MultiBranch-LoRA
                    to set the fixed-denominator mask_scale before each forward.
    """

    def __init__(self, model, tokenizer, eval_loader,
                 max_new_tokens: int = 256, device: str = 'cuda',
                 routing_fn=None, mask_scale_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_loader = eval_loader
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.routing_fn = routing_fn
        self.mask_scale_fn = mask_scale_fn

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run eval, return dict with rougeL_mean, exact_match_mean, per-task breakdown."""
        from models.lora_base import set_routing_all
        from models.lora_models import MultiBranchLoRAModel
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            _tqdm = None
        scorer = _lazy_rouge()
        self.model.eval()
        base = self.model.base  # HF CausalLM
        by_task: Dict[str, List[Dict]] = {}

        # Progress bar over per-example generations. 119 tasks × 10 inst = 1190
        # decodes per cell, each ~3-5s on 5090 for 128 tokens at bs=1. Without
        # a bar the user sees ~60 minutes of silence and assumes the process
        # hung.
        total_examples = sum(b['input_ids'].size(0) for b in self.eval_loader)
        pbar = _tqdm(total=total_examples, desc='ROUGE-L eval',
                      unit='ex', smoothing=0.1) if _tqdm is not None else None

        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            attn_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            cluster_id = batch['cluster_id'].to(self.device)
            task_ids = batch.get('task_id', ['?'] * input_ids.size(0))

            # Install mask_scale once per batch (scalar, batch-independent).
            if self.mask_scale_fn is not None and isinstance(self.model, MultiBranchLoRAModel):
                self.model.set_mask_scale(self.mask_scale_fn())

            # Build the "prompt only" portion: input_ids up to where labels != -100 starts.
            # For each sample i, the first j with labels[i,j] != -100 marks target start.
            for i in range(input_ids.size(0)):
                lbl = labels[i]
                nonneg = (lbl != -100).nonzero(as_tuple=True)[0]
                if len(nonneg) == 0:
                    continue
                prompt_end = int(nonneg[0])
                prompt_ids = input_ids[i:i+1, :prompt_end]

                # Install routing PER EXAMPLE: generate() processes prompts one at
                # a time (size-1 batch), so the routing mask must also be size-1.
                # Previously routing was installed once per eval-batch with size-B
                # mask; per-example generate then hit "branch_mask shape (B, K) !=
                # (1, K)". Re-installing per i keeps mask shape in sync with
                # generate's (1, T) input.
                if self.routing_fn is not None:
                    sample_cluster = cluster_id[i:i+1]
                    routing = self.routing_fn(sample_cluster)
                    set_routing_all(base, routing)
                # target ground truth (decode from labels[prompt_end:])
                gt_ids = lbl[prompt_end:]
                gt_ids = gt_ids[gt_ids != -100]
                gt_text = self.tokenizer.decode(gt_ids, skip_special_tokens=True)

                # Wrap generate() in bf16 autocast — the base Llama is bf16
                # but LoRA adapter nn.Linear layers default to fp32. Training
                # uses autocast(bf16) in trainer_lora._forward_step so this
                # works silently; generate() has no such wrapper and crashes
                # with "expected BFloat16 but found Float" at the base+adapter
                # add site. Match training-path precision exactly here.
                # Explicit attention_mask (all ones since prompt_ids is already
                # sliced to [:prompt_end], no padding inside). Silences the
                # "pad token == eos token" inference warning.
                prompt_attn = torch.ones_like(prompt_ids)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(self.device == 'cuda')):
                    gen = base.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attn,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # greedy per Wang 2022
                        pad_token_id=self.tokenizer.pad_token_id,
                        # Silence HF's "temperature/top_p not valid for greedy"
                        # warning by explicitly unsetting them (they come from
                        # Llama-3.2's chat-oriented generation_config defaults
                        # and are ignored under do_sample=False anyway).
                        temperature=None,
                        top_p=None,
                    )
                pred_ids = gen[0, prompt_end:]
                pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

                task_id = (task_ids[i] if isinstance(task_ids, (list, tuple))
                           else task_ids)
                rec = {
                    'pred': pred_text.strip(),
                    'gt': gt_text.strip(),
                    'rougeL': scorer.score(gt_text, pred_text)['rougeL'].fmeasure,
                    'exact_match': 1.0 if pred_text.strip() == gt_text.strip() else 0.0,
                }
                by_task.setdefault(task_id, []).append(rec)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Aggregate per task then mean-of-tasks (Wang 2022 protocol).
        per_task_stats = {}
        for tid, recs in by_task.items():
            if not recs:
                continue
            rouge_mean = sum(r['rougeL'] for r in recs) / len(recs)
            em_mean = sum(r['exact_match'] for r in recs) / len(recs)
            per_task_stats[tid] = {'rougeL': rouge_mean, 'exact_match': em_mean,
                                    'n': len(recs)}
        if per_task_stats:
            rouge_overall = sum(s['rougeL'] for s in per_task_stats.values()) / len(per_task_stats)
            em_overall = sum(s['exact_match'] for s in per_task_stats.values()) / len(per_task_stats)
        else:
            rouge_overall = em_overall = 0.0

        return {
            'rougeL_mean': rouge_overall,
            'exact_match_mean': em_overall,
            'per_task': per_task_stats,
            'num_tasks': len(per_task_stats),
        }
