# RiboNN — GPU / CUDA 12 fork (fast ensemble prediction)

A fork of [**Sanofi-Public/RiboNN**](https://github.com/Sanofi-Public/RiboNN)
(translation-efficiency prediction from mRNA sequence, *Nat. Biotechnol.*
`s41587-025-02712-x`) that (a) runs on **CUDA 12 / Hopper (sm_90) GPUs** and
(b) removes a large redundant-computation bottleneck in the prediction path.

The change is **~6 lines in `src/predict.py`**; `src/data.py`, the model, and the
weights are untouched. Predictions are **output-identical to upstream** — see
[Verification](#verification).

> Upstream base: commit `af65ce5`. This fork adds one commit on top. Tracked diff
> vs upstream is exactly `src/predict.py` + `environment.yml`.

## Tested on

| | |
|---|---|
| GPU | NVIDIA **H100 80 GB** (SXM, HBM3), compute capability **sm_90** |
| Cluster | hpc-cluster |
| Toolchain | `StdEnv/2023`, `cuda/12.2`, `gcc/12.3` |
| DL stack | Python 3.10.13, **torch 2.1.0+cu121**, pytorch-lightning 2.1.4 |
| Workload | 20,384 testseq mRNA constructs (5′UTR + CDS + 3′UTR) |

## The problem

**1. The stock environment never uses the GPU.** RiboNN pins `pytorch=1.13.1`,
and the installed build is CPU-only (`torch.version.cuda is None`). torch 1.13.1
also cannot target sm_90 (it tops out at sm_86 / CUDA 11.7), so on an H100 node
`torch.cuda.is_available()` is `False` and the entire prediction silently runs on
CPU.

**2. The prediction loop re-encodes the dataset for every model (the real cost).**
Confirmed in the source:

- `DataFrameDataset.__getitem__` (`src/data.py`) one-hot-encodes each sequence
  with a **pure-Python per-nucleotide loop** over a fixed padded length of
  `max_utr5_len + max_cds_utr3_len = 1381 + 11937 = 13,318` positions — for every
  one of the 20,384 sequences. This Python loop is the dominant cost (~5 min/pass).
- `predict_dataloader()` builds a **brand-new** `DataFrameDataset` + `DataLoader`
  on every call (`src/data.py`), and the original
  `predict_using_models_trained_in_one_fold` called it **once per ensemble member**
  (`src/predict.py`) — i.e. `n_folds × top_k ≈ 25` times for the published
  multi-task models.

The encoding depends only on the input sequences and fixed config flags — never on
model weights — so 24 of those ~25 encodings are pure waste. On the 20,384-sequence
set this redundant encoding accounted for **≈2 h of the 2 h 40 m CPU prediction
job**.

## The fix

In `src/predict.py` only (so `data.py`, and therefore the encoding itself, is
provably unchanged): **encode once, reuse across the ensemble.**

- In `predict_using_nested_cross_validation_models`, immediately after the
  `RiboNNDataModule` is built, encode the dataset a single time and move the
  batches to the target device up front:
  ```python
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  cached_batches = [b.to(device) for b in dm.predict_dataloader()]
  ```
- `predict_using_models_trained_in_one_fold` drops its own
  `predict_dataloader()` call and forwards each model over the cached,
  device-resident batches instead.

**Why this is safe:** `predict_dataloader()` uses `shuffle=False` (`src/data.py`),
so batch order is deterministic and still lines up with `dm.df` for the final
`pd.concat` — the reuse cannot misalign rows. The forward math is untouched.

Also pins `setuptools<80` in `environment.yml` (newer setuptools breaks the
pinned pytorch / pytorch-lightning build).

## Verification

The change is a pure refactor of the prediction loop, gated by an **exact-match
correctness test**:

- Patched and unpatched runs on **identical inputs, same device (CPU)** — same
  device isolates the caching change from CPU↔GPU float drift.
- Compared across **all 79 `predicted_TE_*` columns** on 500 transcripts:
  **max absolute difference = 0.000e+00** (hpc-cluster job `14676171`, ~11.5 min).

Because the math is untouched, the match is exact, not merely within tolerance.
The CUDA-12 path was separately shakedown-tested on a real H100 (job `14668661`):
`torch.cuda.is_available() == True`, valid TE output.

## Performance

The fix collapses ~25 dataset encodings to **one**. Expected full-run wall time
for the 20,384-sequence set:

| | encoding | inference | total |
|---|---|---|---|
| Upstream (CPU) | ~5 min × ~25 (rebuilt per model) | — | **2 h 40 m** |
| This fork (H100) | ~2–5 min × **1** | ~1 min batched | **< 10 min** (expected) |

Allocate **> 1 h** walltime for the full run (the previous GPU attempt was killed
by a 1 h cap, not by a bug). The speedup also applies on CPU.

**Memory note:** caching every encoded batch resident on-device is ≈15 GB
(`20,384 × ~14 channels × 13,318 nt × 4 B`) — comfortable on an 80 GB H100. To
cache in host RAM and transfer per batch instead, request ~48 GB.

## Setup (GPU / CUDA 12)

```bash
git clone https://github.com/dksenthil/RiboNN_GPU_CUDA12.git
cd RiboNN_GPU_CUDA12

make install            # base conda env (deps + python 3.10)
mamba activate RiboNN

# upgrade torch to a CUDA-12 / sm_90 build (PyPI, not conda-forge)
pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.4

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA H100 80GB HBM3
```

CPU-only use needs no torch upgrade.

## Usage

Unchanged from upstream — place input as `data/prediction_input.txt`
(columns `tx_id, utr5_sequence, cds_sequence, utr3_sequence`), then:

```bash
make predict_human      # or: make predict_mouse  ->  results/<species>/prediction_output.txt
```

Pretrained weights auto-download from Zenodo (<https://zenodo.org/records/17258709>)
on first run; not redistributed here.

## Relationship to upstream

Intended to stay rebasable on upstream:

```bash
git remote add upstream https://github.com/Sanofi-Public/RiboNN.git
git fetch upstream && git rebase upstream/main
```

## License

Code under the upstream LICENSE; weights under `MODEL WEIGHTS LICENSE.txt`.

## Possible follow-ons (not in this fork)

- Vectorize `DataFrameDataset.__getitem__` (replace the per-nt Python loop with a
  NumPy index assignment) to cut the single encode from minutes to seconds.
- `torch.save` the encoded tensor so re-ranking / model-subset reruns skip
  encoding entirely.
- Raise the single encode's `num_workers`.
