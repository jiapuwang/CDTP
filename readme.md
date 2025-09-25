# Reproducibility README

This document explains how to reproduce the main experiments and evaluations reported in
**“Bridging Structured Knowledge and Text: The CDTP Dataset for Evaluating Chinese Large Language Models”**.
It provides environment, data-prep, training, checkpoint-conversion, inference and evaluation steps, together with example configs and commands.

> **Repository / supplementary files (anonymized for review):** `<ANON_REPO_URL_OR_ARCHIVE>`
> **Dataset (CDTP) snapshot / indexing files:** included in `data/` (or will be released on publication subject to licensing).
> **Hardware used in experiments:** 8 × A800 GPUs (80 GB each) unless specified otherwise.

---

## Contents

* `environment.yml` — reproducible conda env (example)
* `configs/` — sample DeepSpeed JSON and training configs
* `scripts/` — helper scripts (train, resume, zero->hf conversion, inference, evaluate)
* `data/` — dataset splits and preprocessing scripts
* `checkpoints/` — example checkpoint layout (epoch-x-xxxx-zero / epoch-x-xxxx-hf)
* `metrics/` — evaluation scripts (MRR, F1, METEOR)
* `README_reproducibility.md` (this file)

---

## 1. Environment (example)

Save below as `environment.yml` and create conda env:

```yaml
name: cdtp-repro
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch>=2.6.0       # IMPORTANT: transformers' torch.load safety requires torch >= 2.6 for weights-only loads
      - torchvision
      - torchaudio
      - transformers>=4.35
      - deepspeed==0.17.5
      - bitsandbytes       # optional: needed for quantized inference
      - accelerate
      - safetensors
      - sentence-transformers
      - datasets
      - wandb
      - tqdm
      - numpy
      - scikit-learn
      - sacrebleu
      - nltk
      - rouge-score
```

Create env:

```bash
conda env create -f environment.yml
conda activate cdtp-repro
```

> **Notes:**
>
> * We found `torch >= 2.6` required by `transformers` safe-loading (see CVE-related protection). If your system uses an older PyTorch, upgrade (or use `safetensors` format).
> * If you plan to use 4-bit inference, install a compatible `bitsandbytes` build for your CUDA version.

---

## 2. Data & preprocessing

Repository includes `data/preprocess/` with scripts that produce:

```
data/
  CDTP/
    train.jsonl
    val.jsonl
    test.jsonl
  ood/
    hotpotqa.jsonl
    yago3-10.jsonl
    webnlg.jsonl
```

Typical preprocessing command:

```bash
python data/preprocess/build_sft_and_eval_files.py \
  --raw-data-dir data/raw \
  --out-dir data/CDTP \
  --max-seq-len 2048 \
  --sft-version v2
```

This will generate `train.jsonl` / `val.jsonl` / `test.jsonl` in the format consumed by `run_finetune.py`.

---

## 3. Training (example)

Two representative setups used in the paper (we include both as reproducible examples).

### A. ZeRO Stage-3 + CPU offload (your working config that completed 3 epochs)

Create `configs/deepspeed_zero3_offload.json` (example):

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": "<computed-by-script>",
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  }
}
```

Train command (launcher):

```bash
deepspeed --num_gpus 8 run_finetune.py \
  --deepspeed \
  --deepspeed_config configs/deepspeed_zero3_offload.json \
  --exp_name qwen3_32b_text_gen_History_and_Politics \
  --data_path data/CDTP \
  --model_name_or_path /share/project/models/Qwen3-32B \
  --sft_data_version v2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 1 \
  --zero_stage 3 \
  --offload \
  --use_bf16 \
  --output_dir checkpoints/qwen3_32b_text_gen_History_and_Politics
```

### B. ZeRO Stage-2 (example)

`configs/deepspeed_zero2.json` similar but `"stage": 2` and no offload sections. Training command uses `--zero_stage 2` and larger `per_device_train_batch_size` if memory permits.

---

## 4. Resume training from a checkpoint

DeepSpeed checkpoints (ZeRO) can be used to resume training. Two common approaches:

### 4.1 Resume using DeepSpeed engine (recommended)

If you used DeepSpeed's `engine.save_checkpoint`, you can start the same command and add `--resume_from_checkpoint <CHECKPOINT_DIR>` (older deepspeed versions also inspect `args`):

```bash
deepspeed --num_gpus 8 run_finetune.py \
  ...same args... \
  --resume_from_checkpoint checkpoints/qwen3_32b_text_gen_History_and_Politics/epoch-1-1770-zero
```

**If your script does not parse CLI resume flags**, add a small snippet after DeepSpeed initialization to load latest checkpoint:

```python
# after model, optimizer, lr_scheduler = deepspeed.initialize(...)
# engine is the returned model object (DeepSpeedEngine)
ckpt_dir = "/path/to/checkpoint-dir"  # choose zero checkpoint directory
loaded = model.load_checkpoint(ckpt_dir)  # model is engine
# or: model.engine.load_checkpoint(...)
print("resume status:", loaded)
```

### 4.2 If you saved HF-format checkpoints (hf directories)

If you saved `*-hf` folders containing HuggingFace `pytorch_model.bin` or `safetensors`, use those to restart from HF weights (but for large models prefer Zero->HF conversion first).

---

## 5. Converting ZeRO checkpoints → HF (single-file) or safetensors

DeepSpeed ZeRO stage-3 stores partitioned shards. To run inference with HF `from_pretrained()` you must aggregate them.

Two options:

### Option A — Use `deepspeed` utility (if available and working)

```bash
python /path/to/deepspeed/utils/zero_to_fp32.py \
  --checkpoint_dir /path/to/epoch-2-2655-zero \
  --output_file /path/to/epoch-2-2655-zero-merged/pytorch_model.bin
```

> Note: In some environments the packaged `zero_to_fp32.py` may import local modules and cause circular-import errors. See Option B.

### Option B — Use a conversion helper script (recommended if deepspeed utility broken)

We include `scripts/convert_zero_to_hf_simple.py` in the repo. Example:

```bash
python scripts/convert_zero_to_hf_simple.py \
  /share/project/.../epoch-2-2655-zero \
  /share/project/.../epoch-2-2655-zero-2-hf
```

Then copy tokenizer and metadata:

```bash
cp /share/project/models/Qwen3-32B/tokenizer* /share/project/.../epoch-2-2655-zero-2-hf/
cp /share/project/models/Qwen3-32B/special_tokens_map.json /share/project/.../epoch-2-2655-zero-2-hf/ || true
cp /share/project/models/Qwen3-32B/generation_config.json /share/project/.../epoch-2-2655-zero-2-hf/ || true
```

**Which checkpoint to use?**

* The `*-zero` directory contains the full-precision partitioned parameters (large; correct).
* The `*-hf` folder that your training script produced may be *incomplete* (tiny `pytorch_model.bin` of a few MBs means only metadata or a failed dump). **Prefer `*-zero` and convert it**. After conversion, you get a proper HF-format model you can load with `AutoModelForCausalLM.from_pretrained()`.

**Recommendation:** Always verify the converted `pytorch_model.bin` or `pytorch_model.safetensors` shape sizes by a small test script (e.g., check embedding shapes). If sizes mismatch model config (vocab\_size, d\_model), the conversion failed.

---

## 6. Inference — example script

Example `inference.py` uses quantized loading to save VRAM:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("/path/to/epoch-2-2655-zero-2-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/epoch-2-2655-zero-2-hf",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=bnb_cfg  # optional
)
model.eval()
```

Then run inference similar to `qwen3_32b_Kg_HS.py` in this repo. **Important:** `device_map="auto"` + quantization requires proper `bitsandbytes` and `accelerate` setup and enough PCIe memory.

---

## 7. Evaluation scripts

We include `metrics/compute_mrr.py`, `metrics/compute_f1.py`, `metrics/compute_meteor.py`. Typical usage:

```bash
python metrics/compute_mrr.py --pred predictions.jsonl --gold gold.jsonl --out results/mrr.json
python metrics/compute_f1.py --pred predictions.jsonl --gold gold.jsonl --out results/f1.json
python metrics/compute_meteor.py --pred predictions.jsonl --gold gold.jsonl --out results/meteor.json
```

These scripts expect JSONL lines with fields `{ "id": ..., "model_answer": "...", "gold": ... }`.

---

## 8. OOD experiments

To reproduce RQ3 (OOD robustness):

1. Prepare OOD datasets:

   * KGC: YAGO3-10 split in `data/ood/yago3-10.jsonl`
   * QA: HotpotQA (Chinese-paraphrased / converted) in `data/ood/hotpotqa.jsonl`
   * T2T: WebNLG in `data/ood/webnlg.jsonl`

2. Run inference on base model and SFT model:

```bash
python scripts/infer_on_dataset.py --model /path/to/base --data data/ood/yago3-10.jsonl --out out/base_yago_preds.jsonl
python scripts/infer_on_dataset.py --model /path/to/sft_epoch2 --data data/ood/yago3-10.jsonl --out out/sft_yago_preds.jsonl
```

3. Compute metrics and produce plots (scripts included in `analysis/`).

> **Note on reporting:** In the paper we reported OOD results for the best-performing model (Yi\_9B) as the representative case. For reproducibility we include the evaluated checkpoint and the exact commands used.

---

## 9. Reproducing table / figure generation

We provide `analysis/` notebooks that read the raw metric outputs and generate the plots/tables used in the paper. Run:

```bash
jupyter nbconvert --execute analysis/plot_ood_results.ipynb --to html
```

---

## 10. Troubleshooting / common pitfalls

* **Tiny `*-hf/pytorch_model.bin` (e.g., 1.6MB)**: indicates the HF-dump did not include parameters (often a failure during save). Use the `*-zero` checkpoint and convert using `zero_to_fp32.py` or the conversion helper. Confirm shapes by loading the file with `torch.load()` and printing key shapes.

* **`RuntimeError: size mismatch for weight` when loading HF model**: usually means model config (hidden size / vocab size) differs from checkpoint — confirm `config.json` in the HF folder matches the original base model.

* **`ValueError: ... requires torch >= 2.6`**: upgrade PyTorch in your conda env. Transformers may refuse to call `torch.load` with older versions.

* **BitsAndBytes errors (`PackageNotFoundError: No package metadata was found for bitsandbytes`)**: ensure `bitsandbytes` is installed in the same conda env, and that it's a build compatible with your CUDA version.

* **DeepSpeed conversion script imports failing** (circular import): run conversion script within the same env used for training (matching DeepSpeed install) or use the conversion helper in `scripts/` we provide.

* **Resuming training**: use DeepSpeed `load_checkpoint` / `resume_from_checkpoint` semantics as shown above. Also ensure `--init_global_step` (if your script uses it) is consistent.

---

## 11. Checkpoint publishing plan (to include in final paper / appendix)

When the paper is accepted we will publish:

* CDTP dataset (train/val/test + OOD splits) under `<dataset_license>` at `<release_url>` (or Zenodo accession).
* Training configs, seed values, and scripts (in `scripts/`).
* Best-performing model checkpoints under permissive download (or upon request if license limits apply). We will provide both DeepSpeed `*-zero` directories and converted HF `safetensors` snapshots when possible.

*(Replace placeholders above with the final URLs / license text prior to camera-ready.)*

---

## 12. Minimal quick-check (toy run)

To verify end-to-end works in a short time:

1. Create a tiny dataset:

```bash
head -n 50 data/CDTP/train.jsonl > data/toy/train.jsonl
head -n 10 data/CDTP/val.jsonl > data/toy/val.jsonl
```

2. Start an extremely short training (1 step) on CPU/GPU:

```bash
deepspeed --num_gpus 1 run_finetune.py \
  --deepspeed_config configs/deepspeed_zero3_offload.json \
  --data_path data/toy \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 1 \
  --save_interval 1 \
  --output_dir checkpoints/toy_test
```

3. Confirm a `checkpoints/toy_test/epoch-0-...-zero` directory is created and convertible.

---

## Contact / Issues

If you encounter reproducibility problems, please open an issue in the repo (when public) or contact the authors via the submission system. Include: exact `training.log`, `parse_config.json`, `deepspeed` JSON, and the checkpoint directory path.

---

### Acknowledgements / License

When publishing, include the dataset license and model release license here. Example placeholder:

> CDTP dataset: CC BY-SA 4.0 (tentative)
> Code: MIT license (tentative)
> Model checkpoints: subject to third-party model license — check `LICENSE_in_repo.md`.
