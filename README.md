# SpecDrop: Parameter-Free Category-Conditioned Routing for Modular Specialization

Official PyTorch implementation for the "SpecDrop: Parameter-Free Category-Conditioned Routing for Modular Specialization" paper, **NeurIPS 2026 (Poster)**.

> *Granularity alignment, not algorithm choice, localizes when routing helps.*

<p align="center">
    📃 <a href="https://arxiv.org/abs/2608.04084" target="_blank">Paper (arXiv)</a> <br>
</p>

![overview](assets/specdrop_overview.png)

## News
- **[2026/09]** SpecDrop is accepted to NeurIPS 2026 (Poster).

The current release supports:

- Soft SpecDrop training and evaluation across four settings: CIFAR-100 (ResNet-110), ImageNet-1K under the BREEDS-46 partition (ViT-S/16), SlimPajama-6B language modeling (30M / 125M Transformer LM), and SuperNI instruction tuning (Llama-3.2-1B + LoRA).
- Every baseline in the paper's main tables: dense references, architecture-matched No-Routing(+SE) controls, HardCategory, Stochastic Depth, Example-Tied Dropout, Contextual Dropout, Soft MoE (deployed / tuned second-half / compute-matched), Mod-Squad, the auxiliary-loss-free top-k router, COMET, Switch, Hash Layers, SMoE-Dropout, DEMix, single LoRA, LoRAMoE, MoCLE, and HydraLoRA.
- The paper's evaluation controls and analysis tooling: the information-matched logit-masking control, label-quality sweeps, branch–category alignment via pruning sensitivity, and per-method MACs / wall-clock accounting.
- One-command reproduction of every main-table result (Tables 1–4 and their Align columns) via `reproduce.sh`, with per-cell auto-skip on resume and 470 unit tests.

## Contents
- [SpecDrop: Parameter-Free Category-Conditioned Routing for Modular Specialization](#specdrop-parameter-free-category-conditioned-routing-for-modular-specialization)
	- [News](#news)
	- [Contents](#contents)
	- [Install](#install)
	- [Usage](#usage)
	- [Datasets](#datasets)
	- [Results](#results)
	- [Reference](#reference)
	- [License](#license)

## Install
1. Clone the repository and navigate to the SpecDrop working directory
```bash
git clone https://github.com/Beryex/SpecDrop.git --depth 1
cd SpecDrop
```
2. Set up the environment
```bash
conda create -n SpecDrop python=3.12 -y
conda activate SpecDrop
pip install -r requirements.txt
```
3. Verify the installation. Without a GPU, run the unit tests (`python -m pytest tests/ -q`). With a CUDA GPU, run the smoke covering all four settings (~10 min warm; ~30–50 min on a first run). First-run caveat: the smoke itself triggers the one-time dataset downloads — small for CIFAR/LoRA (gated Llama-3.2-1B base + a ~0.8 GB shallow SuperNI clone, ~4 GB on disk), large for NLP/ViT (SlimPajama ~14 GB download, ~24 GB decompressed; ImageNet-1K ~150 GB, gated) — so populate the big caches in advance or start with the cifar/lora smokes. The NLP smoke also builds the tokenized SlimPajama caches that `reproduce.sh nlp` and `ablation_nlp` expect, so run it before those targets.
```bash
bash reproduce.sh smoke
```

Training metrics are logged to Weights & Biases by default: run `wandb login` once, or `export WANDB_MODE=offline` to skip it.

## Usage

`reproduce.sh` is the single entry point; each target reproduces one paper table end-to-end (its ablation chains plus the main-table runs, 3 seeds), skipping any cell whose `results.json` already exists (the LoRA chain additionally checks that the metric is populated). The wall-clock figures below are for the main-table runs alone; the sweep chains that run first add about 86 h (CIFAR-100), 225 h (ImageNet), 75 h (SlimPajama) and 115 h (SuperNI) on one GPU (`bash reproduce.sh --help`). For table-only reproduction, run `bash scripts/experiments/<setting>/main_table.sh` directly with the paper's operating point exported when the ablation markers are absent:

```bash
# vit/lora read the operating point from the sweep markers; when skipping the
# ablation chains, export the paper's values instead:
BEST_PA=0.6 BEST_BETA=1 BEST_SE=2.0  bash scripts/experiments/vit/main_table.sh
BEST_PA=0.8 BEST_BETA=1 BEST_SE=1.0  bash scripts/experiments/lora/main_table.sh
# vit/main_table.sh gives 6 of Tab. 2's 9 rows (plus the deployed all-blocks Soft MoE
# of the appendix); the tuned and compute-matched Soft MoE and the ALF router rows:
FOREGROUND=1 bash scripts/experiments/softmoe_study_finals.sh softmoe_study_r7   # tuned
FOREGROUND=1 bash scripts/experiments/softmoe_study_finals.sh softmoe_study_r6   # compute-matched
FOREGROUND=1 bash scripts/experiments/softmoe_study_finals.sh alf_moe_vit        # ALF top-k router
# cifar/nlp main tables hardcode the paper operating point and need no env vars.
```

Exporting the operating point also fixes it for `bash reproduce.sh vit` and `bash reproduce.sh lora`: the sweep chains still run every cell, but their later stages and the main table then use the exported values instead of the argmax of your own sweep.

```bash
bash reproduce.sh cifar          # Table 1: CIFAR-100
bash reproduce.sh vit            # Table 2: ImageNet-1K BREEDS ViT  ← longest; multi-GPU recommended
bash reproduce.sh nlp            # Table 3: SlimPajama-6B LM
bash reproduce.sh lora           # Table 4: SuperNI Llama-3.2-1B + LoRA
bash reproduce.sh alignment      # Align columns of Tables 1–4 (after the four above complete);
                                 # summary table: python scripts/aggregate_alignment.py
```

`bash reproduce.sh --help` lists every target with its expected wall-clock. Seeds are independent, so the standard multi-GPU pattern is one seed per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 SEEDS_OVERRIDE=42  bash scripts/experiments/cifar/main_table.sh &
CUDA_VISIBLE_DEVICES=1 SEEDS_OVERRIDE=123 bash scripts/experiments/cifar/main_table.sh &
CUDA_VISIBLE_DEVICES=2 SEEDS_OVERRIDE=456 bash scripts/experiments/cifar/main_table.sh &
wait
```

All reported training runs used NVIDIA RTX 5090 (32 GB) with mixed precision (bf16; fp16 with loss scaling on CIFAR-100). Other GPUs with enough memory should reproduce them within seed noise; the ImageNet and LoRA runs were sized for 32 GB.

| Setting | Single-GPU wall-clock (3 seeds) | Multi-GPU shortcut |
|---|---|---|
| CIFAR-100 (Tab 1) | ~22 h | 3 GPUs × 1 seed each → ~8 h |
| ImageNet ViT-S/16 (Tab 2) | ~850 h | 3 GPUs → ~285 h (recommended) |
| SlimPajama 30M LM (Tab 3) | ~195 h | 3 GPUs → ~65 h |
| SuperNI Llama-1B + LoRA (Tab 4) | ~235 h | 3 GPUs → ~80 h |
| Branch–category alignment | ~24 h | 6 GPUs (per-method shards) → ~4 h |

<details>
<summary><b>Repository layout</b></summary>

```
.
├── reproduce.sh                # single entry: bash reproduce.sh <target>
├── algorithms/                 # routing rules (the methodological surface)
│   ├── soft_specdrop.py        # ours (per-cat soft mask + cosine warmup, optional shared expert)
│   ├── no_dropout.py           # all-branches-equal baseline
│   ├── hard_category.py        # one-hot category routing baseline
│   └── ...                     # Stochastic SpecDrop, random dropout (App. A); baselines live in models/
├── models/                     # per-setting backbones + baseline model classes
│   ├── multi_branch.py         # MultiBranchResNet110 (K parallel branches, shared stem/head)
│   ├── multi_branch_vit.py     # MultiBranchViT (K parallel MLPs per block, shared attn)
│   ├── transformer_lm.py       # Dense + MultiBranch Transformer LM (einsum ParallelFFN)
│   ├── soft_moe_vit.py         # SoftMoEViT (all-blocks / second-half placement)
│   ├── alf_moe_vit.py          # auxiliary-loss-free top-k router (Wang et al. 2024)
│   ├── lora_models.py          # Single/MultiBranch/Hydra/LoRAMoE/MoCLE LoRA models
│   └── ...
├── data/                       # CIFAR-100 superclasses, BREEDS-46, SlimPajama domains, SuperNI clusters
├── training/                   # per-setting trainers (CV / NLP / LoRA)
├── evaluation/                 # accuracy, alignment, pruning sensitivity, ROUGE-L, MACs
├── configs/                    # per-method YAMLs (which are exact paper configs: Reproducibility notes)
├── scripts/                    # analysis tools + paper-reproduction experiment chains
│   ├── eval_logit_mask.py      # information-matched logit-masking control
│   ├── wall_clock_table.py     # per-method wall-clock table
│   ├── compute_flops_tables.py # per-method MACs (fvcore; closed form for LoRA adapters)
│   └── experiments/            # per-setting reproduction chains ({cifar,vit,nlp,lora,alignment,smoke})
├── tests/                      # 470 unit tests (python -m pytest tests/ -q)
└── run.py / run_nlp.py / run_lora.py   # per-setting entries
```
</details>

<details>
<summary><b>Reproducibility notes</b></summary>

- **Seeds**: 3 fixed seeds (42, 123, 456) for every paper-table cell. Seed scope = training; routing-structure parameters (hash_seed, mask_seed, router_seed) are fixed at 42 across all seeds, decoupling training-noise from routing-structure variance in the 3-seed standard deviation.
- **Determinism**: `torch.use_deterministic_algorithms(warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Cross-machine top-1 / PPL / ROUGE-L reproduces within seed noise on any RTX 5090; bit-identical reproduction is not claimed.
- **Param budget**: `utils/sanity_check.py` runs before every training call and crashes if the trainable parameter count is more than 2% (LoRA: 3%) off the per-setting reference (CIFAR ResNet-110 1.737 M, ViT-S/16 22.051 M, NLP 30.143 M, Llama-1B + LoRA 225 M).
- **Auto-skip on resume**: every reproduction script skips a cell whose `outputs/<run_dir>/results.json` exists (the LoRA chain additionally verifies the relevant metric is populated); re-running a chain after a partial completion only fills in the missing cells.
- **Tests**: 470 unit tests; `python -m pytest tests/ -q` should be all-green before claiming reproduction.
- Run-level provenance (per-run `results.json` with per-epoch histories) is available on request.
- **Configs**: the paper's runs are generated by the launchers in `scripts/experiments/` (operating points, batch sizes, seeds), which save each run's exact configuration in its output directory. The standalone YAMLs that equal a reported configuration (up to experiment name, output directory and seed) are: every `configs/cv/` file except `stoch_depth_pl{0.3,0.7,0.9}.yaml` (Tabs. 1 and 7, App. A.4 and E.2); `configs/vit/softmoe_study_r{0..7}.yaml`, `alf_moe_vit_screen.yaml` and `mod_squad_vit_screen.yaml` (App. E.7 screen; `softmoe_study_finals.sh` turns r6/r7 into the Tab. 2 runs) and `alf_moe_vit.yaml` (Tab. 2); `configs/nlp/{dense,switch,hash_layers,smoe_dropout,demix,mb_no_routing}.yaml` (Tab. 3; their `betas` entry is not read by the NLP trainer, whose defaults are the same values) and `configs/nlp/*_125m.yaml` (App. E.10); `configs/lora/{single_lora_r320,loramoe,hydra_lora,mocle,mb_lora_no_routing}.yaml` (Tab. 4). Every other YAML starts with a `# NOT a reported configuration` line.
- **Batch sizes**: CIFAR 128, ImageNet 256 (ALF and the Mod-Squad screen: 128 × 2 accumulation), SlimPajama 32 (the 1-epoch control 64, the 125M runs 16), SuperNI 8 × 16 accumulation.
- **Baseline evaluation as reported (paper App. B.2)**: DEMix is evaluated with the document's domain label selecting its expert (`demix_eval_mode: oracle`). SMoE-Dropout's gradual-k schedule is not applied when the model runs under `torch.compile`, as in the reported runs: training uses k=1 and evaluation all 16 experts (`configs/nlp/smoe_dropout.yaml` sets the same `_compile_mode`).
- **Environment**: developed and unit-tested with Python 3.10, PyTorch 2.10 and transformers 5.5; the unit tests also pass on Python 3.12 with current releases of `requirements.txt`. Where the run logs record it, the main-table and sweep runs used Python 3.10 and the App. E.7 Soft MoE study (run on a second node) Python 3.12; the only PyTorch version the logs record is from the pre-run smoke check on the RTX 5090: 2.11.0+cu130.
</details>

### Appendix results

Beyond the main tables, the appendix results come from these entry points (run after the corresponding main-table chain; most analysis scripts take `--help`):

| Paper item | Command |
|---|---|
| Fig. 4, Tab. 6, App. E.6–E.8 (operating-point sweeps) | `bash reproduce.sh ablation_<cifar\|vit\|nlp\|lora>` (`ablation_nlp` includes the E.6 X=0.5 matched-SE cell, `ANCHOR_SE=0.5 PA_VALUES=0.5 bash scripts/experiments/nlp/ablation_pa.sh`); epoch-warmup Phase 3a runs for the E.6 epoch-vs-step drift check: `bash archive/scripts/rtx5090_3a_phaseP_pa_sweep_epoch.sh` (writes `outputs/rtx5090_nlp_mini_ablation_epoch/`); plot: `python scripts/plot_ablation_curves.py` |
| Tab. 7 (mask × denominator) | `bash scripts/experiments/extras/denom_ablation.sh`, then `python scripts/summarize_e3_denom.py` |
| App. A.4 (random assignment) | `python run.py --config configs/cv/soft_specdrop_random_a.yaml --output_dir outputs/rtx5090_random_a/s42` (set `seed:` in the YAML to 123 / 456 for the other seeds) |
| Fig. 1, Fig. 5, Tab. 8 | `bash reproduce.sh alignment`, then `python scripts/plot_intro_specialization.py`, `python scripts/analyze_e2_heatmap.py`, `python scripts/analyze_e1_mi_table.py` |
| App. E.1 (ties in the CIFAR-100 Align) | after `bash reproduce.sh alignment`: `python scripts/cifar_align_ties.py` |
| §5.6 (uniform-mask inference) | `bash scripts/experiments/extras/uniform_mask_eval.sh` |
| App. E.2 (fine-label oracle) | `bash scripts/experiments/extras/fine_label_oracle.sh` |
| App. E.3, Tab. 9 (logit masking) | `python scripts/eval_logit_mask.py --setting <cifar\|vit> --method <method> --seed <seed>` (the out-of-superclass prediction count is `n_pred_outside_category` in the same JSON) |
| App. E.4, Tab. 10 (label quality) | `python scripts/analyze_e5_noise_sweep.py --ps 0 0.05 0.1 0.2 0.25 0.5 1.0`, then `python scripts/inference_robustness_curve.py`; predicted labels: `python scripts/finetune_coarse_classifier.py --setting <cifar\|vit> --seed <seed>` and `python scripts/eval_predicted_cluster.py --setting <cifar\|vit> --seed <seed>` |
| App. E.5 (NLP per-domain reliance) | `SEEDS_OVERRIDE=42 bash archive/scripts/rtx5090_3a_nlp_pa_sweep.sh` (its p_a=0.7 cell), then `python scripts/diagnose_nlp_specialization.py --run_dir outputs/rtx5090_nlp_mini_ablation/phaseA_pa0.7_pi0.3_s42` |
| App. E.7 (Soft MoE tuning study), Fig. 6 | screen arms r0–r6 (seed 42): `bash scripts/experiments/softmoe_study_screen.sh`; every other screen cell (r7, the ALF and Mod-Squad screens, and all arms at seeds 123 / 456): `python run.py --config configs/vit/<name>.yaml --seed <S> --output_dir outputs/softmoe_study_screen/<name>_s<S>`; full-protocol finals: `FOREGROUND=1 bash scripts/experiments/softmoe_study_finals.sh <softmoe_study_r7\|softmoe_study_r6\|alf_moe_vit>`; Fig. 6 (after `vit/main_table.sh` and the r7 finals): `python scripts/plot_training_curves.py` |
| App. E.10 (125M scale) | `bash scripts/launch_125m_seeds.sh` |
| App. E.11, App. B.2 (1-epoch and batch-64 checks) | `bash scripts/experiments/extras/nlp_1ep.sh`, `bash scripts/experiments/extras/nlp_bs64.sh` |
| App. E.12–E.14 (LoRA diagnostics) | Pruning sensitivity, per seed (Tab. 4 Align, App. E.12): `python scripts/diagnose_lora_specialization.py --run_dir outputs/rtx5090_lora_faithful/ours_s<seed>` for seeds 42/123/456 (and the other LoRA runs for their Align cells). Per-task ROUGE-L at seed 42 (App. E.13): `python scripts/eval_lora_per_task.py --run_dir outputs/rtx5090_lora_faithful/<run>_s42 --method <run>` for `<run>` = `ours`, `hydra_lora_n8`, `mb_lora_no_routing`; then `python scripts/lora_per_task_fisher.py` (App. E.14). Soft SpecDrop is evaluated at the routing state with which the run's reported checkpoint was scored: the per-step $p_a$ warmup at the end of `results.json`'s `best_epoch` (recorded as `config_summary.routing_progress`; 0.667 / 1.0 / 0.667 for the paper's seeds 42 / 123 / 456) and the per-cluster training fractions. Check: the `n_tasks`-weighted mean of `baseline_per_cat[*].rouge_l` in the diagnostic JSON, and `rougeL_mean` of the per-task JSON, equal `results.json` `eval_rouge_l` exactly on the same GPU type. |
| App. E.15 (embedding diagnostics) | BGE embeddings and k=7 clusters of the 100M-token cache: `python -m data.cluster_chunks --train-cache <tokenized_train_seq512_tok100000000_*.pt> --train-output data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt --val-cache <tokenized_val_seq512_tok5000000_*.pt> --val-output data_cache/slimpajama/clusters_val_seq512_tok5000000_bge-large_k7.pt --embedder BAAI/bge-large-en-v1.5 --n_clusters 7`; then, with `<emb>` = `data_cache/slimpajama/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt` (written by `cluster_chunks`): `python scripts/select_optimal_k.py --emb-cache <emb> --k-min 2 --k-max 50`; CIFAR-100 DINOv2: `python scripts/embed_cifar100.py --output <cifar_emb.pt>`, then `python scripts/select_optimal_k.py --emb-cache <cifar_emb.pt> --k-min 2 --k-max 50`; chunk purity vs. PPL over the 9,766 validation chunks: `python scripts/analyze_purity_ppl_correlation.py --val-token-cache <tokenized_val_seq512_tok5000000_*.pt> --train-cluster-cache <clusters_train_…_bge-large_k7.pt> --train-embedding-cache <emb>`; sub-window check on 300 training chunks: `python scripts/intra_chunk_heterogeneity.py --token-cache <tokenized_train_seq512_tok100000000_*.pt> --cluster-cache <clusters_train_…_bge-large_k7.pt> --embedding-cache <emb>` |
| App. F.3 (MACs, wall-clock) | `python scripts/compute_flops_tables.py`, `python -m scripts.profile_vit_flops`, `python scripts/wall_clock_table.py`; Tab. 16's bottom three rows are the 3-seed means of `compute.total_training_time_sec` in `outputs/softmoe_study_finals/<softmoe_study_r7\|softmoe_study_r6\|alf_moe_vit>_s<seed>/results.json` (÷100 epochs for s/epoch) |

One appendix item has no script: the Tab. 14 timings are development-time benchmarks on an A100.

## Datasets

All datasets are fetched automatically on first run (CIFAR-100 via torchvision, SuperNI via a git clone, everything else via Hugging Face Datasets / Hub); caches go to `data_cache/` (gitignored).

| Dataset | Source | Size | First-run time |
|---|---|---|---|
| CIFAR-100 | torchvision | ~170 MB | ~1 min |
| ImageNet-1K | HF `ILSVRC/imagenet-1k` (gated: accept the terms on its Hub page, then run `hf auth login` or set `HF_TOKEN`) | ~150 GB | ~30 min on fast network |
| SlimPajama-6B | HF `DKYoon/SlimPajama-6B` | ~14 GB download (~24 GB decompressed); ~12 GB tokenized at seq=512 | ~30 min tokenize at 500M tokens |
| SuperNI v2 | git clone of `allenai/natural-instructions` (automatic; task JSONs with the Domains fields we need) | ~0.8 GB download (shallow), ~4 GB on disk | ~2 min |
| Llama-3.2-1B | HF `meta-llama/Llama-3.2-1B` | ~2.5 GB | ~3 min (requires HF gated-model access) |
| BREEDS hierarchy | bundled at `data/breeds_hierarchy/`: unmodified hierarchy files from [MadryLab/BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks) (Santurkar et al., ICLR 2021), which states no license; redistributed with attribution and not covered by this repository's MIT license (see `data/breeds_hierarchy/NOTICE`) | <1 MB | n/a |

## Results

Headline comparison against the architecture-matched No-Routing(+SE) controls and the dense references (mean ± std over 3 seeds; Align = branch–category alignment via pruning sensitivity):

| Setting | Partition | Metric | Ours | Matched control | Dense | Align (ours) |
|---|---|---|---|---|---|---|
| CIFAR-100 (ResNet-110) | aligned | Top-1 ↑ | **79.23 ± 0.17** | 63.08 | 74.48 | 68.3% |
| ImageNet-1K BREEDS (ViT-S/16) | aligned | Top-1 ↑ | **79.89 ± 0.18** | 73.36 | 76.38 | 100.0% |
| SlimPajama-6B (30M LM) | fuzzy (predicted null) | PPL ↓ | 45.38 ± 0.02 | 45.28 | 44.80 | 94.4% |
| SuperNI (Llama-3.2-1B + LoRA) | fuzzy (predicted null) | ROUGE-L ↑ | 0.5106 ± 0.003 | 0.5094 | — | 22.2% |

On the aligned vision partitions SpecDrop exceeds the parameter-matched baselines that do not use the category label; these gains quantify what category supervision buys when deployed through routing, and the paper's information-matched masking control separates the label's share from routing's (given the same label, masking a dense model's outputs is stronger for accuracy alone — SpecDrop's contribution is converting the label into trained-in modular structure). On the fuzzy language partitions the routing mechanism ties the matched controls, the null predicted by the paper's granularity-alignment thesis. See the paper for the full tables, baselines, and scope statements.

## Reference

```bibtex
@inproceedings{wang2026specdrop,
  title     = {{SpecDrop}: Parameter-Free Category-Conditioned Routing for Modular Specialization},
  author    = {Wang, Boyao and Lei, Zhihan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`), except the third-party BREEDS hierarchy files in `data/breeds_hierarchy/` (see the NOTICE there).
