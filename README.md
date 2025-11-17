# Compact Transformer – Indic Translation (From Scratch)

This repository implements an encoder–decoder Transformer **from scratch** in PyTorch for English → Hindi translation.  
It combines:

- A custom Transformer stack (`model.py`) with sinusoidal positions, multi-head attention, decoder cross-attention, and optional [Liger](https://github.com/linkedin/LigerKernel) fused loss.
- Data ingestion via 🤗 `datasets` using the [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) Hindi split (`data.py`).
- SentencePiece tokenization through [`ai4bharat/IndicBARTSS`](https://huggingface.co/ai4bharat/IndicBARTSS) (`tokenizer.py`).
- A feature-complete training loop with gradient accumulation, cosine LR schedule, mixed precision, checkpointing, and optional Weights & Biases logging (`trainer.py`).
- Text generation helpers (`inference.py`) for top‑k sampling and beam search.

> ⚠️ All recent experiments were **CPU-only shakedowns** to validate the pipeline. Meaningful training/inference should run on a GPU instance (e.g., `ml.g4dn.xlarge`, `ml.g5.xlarge`, or local CUDA hardware) with larger sequence lengths and more iterations.

---

## Repository Layout

| File / Dir               | Purpose                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------|
| `config.py`              | CLI + `ModelArgs` dataclass storing every hyperparameter, schedule, and derived values.           |
| `model.py`               | Transformer encoder + decoder modules, attention heads, embeddings, and projection layers.        |
| `data.py`                | Hugging Face dataset loader plus PyTorch `DataLoader` with custom `collate_fn`.                    |
| `tokenizer.py`           | Minimal wrapper around `AlbertTokenizer.from_pretrained("ai4bharat/IndicBARTSS")`.                |
| `trainer.py`             | Main training entry point (local/SageMaker) with LR scheduler, eval loop, gradient accumulation.  |
| `inference.py`           | Generation utilities (`topk_sampling`, `beam_search_corrected`, `save_text`).                     |
| `requirements.txt`       | Runtime Python dependencies (wandb, transformers, datasets, sentencepiece, tqdm).                 |
| `wandb/`, `__pycache__/` | Local artifacts (when present).                                                                   |

---

## Dataset & Tokenizer

- **Dataset**: `ai4bharat/samanantar`, Hindi split (`load_dataset("ai4bharat/samanantar", "hi", split="train")`).
- **Tokenizer**: `ai4bharat/IndicBARTSS` SentencePiece; both source & target reuse the same vocabulary.  
  `tokenizer.py` adds BOS/EOS manually and prepares decoder inputs for teacher forcing.

---
## Training

### Local GPU (recommended)

```bash
export HF_TOKEN=<your_hf_token>
export WANDB_DISABLED=1                   # flip to 0 when ready to log
python trainer.py \
  --device cuda \
  --batch_size 32 \
  --block_size 512 \
  --total_iters 10000 \
  --save_checkpoint_iter 500
```

- Mixed precision via `torch.cuda.amp.GradScaler`.
- Optional `torch.compile` (default on when CUDA available & `ENABLE_TORCH_COMPILE!=0`).
- Checkpoints saved under `checkpoints/snapshot_{step}.pt`.

### Local CPU sanity pass

Useful for pipeline verification; extremely slow for real training:

```bash
export HF_TOKEN=<your_hf_token>
export WANDB_DISABLED=1
export ENABLE_TORCH_COMPILE=0

python trainer.py \
  --device cpu \
  --batch_size 4 \
  --block_size 128 \
  --total_batch_size 8192 \
  --total_iters 1 \
  --eval_iters 100 \
  --eval_check 50 \
  --save_checkpoint_iter 1000
```

### SageMaker – PyTorch estimator (CPU shakedown)

The most recent run (11 Nov 2025) used `ml.c5.2xlarge` with the hyperparameters above (`total_iters=1`, `epochs=1`, etc.) to validate the training/eval loop.

```python
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::<account>:role/service-role/AmazonSageMaker-ExecutionRole-..."

hyperparams = dict(
    device="cpu",
    epochs=1,
    batch_size=4,
    block_size=128,
    total_batch_size=8192,
    total_iters=1,
    eval_iters=100,
    eval_check=50,
    save_checkpoint_iter=1000,
    hf_token="<HF_TOKEN>",
)

estimator = PyTorch(
    entry_point="trainer.py",
    source_dir="...",
    dependencies=["requirements.txt"],
    role=role,
    instance_type="ml.c5.2xlarge",
    instance_count=1,
    framework_version="2.2",
    py_version="py310",
    hyperparameters=hyperparams,
    environment={
        "WANDB_DISABLED": "1",
        "ENABLE_TORCH_COMPILE": "0",
    },
)

estimator.fit({"training": "s3://<bucket>/compact-transformer/"})
```

- Runtime: ~224 seconds (billable).
- Artifact: `/opt/ml/model/snapshot_0.pt` within `model.tar.gz` at  
  `s3://sagemaker-<region>-<account>/.../output/model.tar.gz`.

### SageMaker – GPU (recommended for real training)

Switch to the Hugging Face estimator, e.g.:

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="trainer.py",
    source_dir="...",
    instance_type="ml.g4dn.xlarge",   # or ml.g5.xlarge
    instance_count=1,
    role=role,
    transformers_version="4.36",
    pytorch_version="2.2",
    py_version="py310",
    hyperparameters=dict(
        device="cuda",
        epochs=1,
        batch_size=16,
        block_size=512,
        total_iters=10000,
        hf_token="<HF_TOKEN>",
    ),
    use_spot_instances=False,         # enable once you have spot quota
    environment={"WANDB_DISABLED": "1"},
)

estimator.fit({"training": "s3://<bucket>/compact-transformer/"})
```

---

## Checkpoints & Artifacts

- **Local**: `checkpoints/snapshot_{step}.pt`.
- **SageMaker**: everything saved to `/opt/ml/model` is automatically compressed into `model.tar.gz` and uploaded to the training job’s output S3 URI.

## Offline Inference (Notebook or Local)

```python
import torch
from config import ModelArgs
from tokenizer import initialize_tokenizer
from model import Transformer
from inference import topk_sampling

ckpt = torch.load("snapshot_0.pt", map_location="cpu")

args = ModelArgs(
    block_size=128,      # use training settings
    batch_size=4,
    device="cpu",
)
tokenizer = initialize_tokenizer(args.hf_token)
args.src_vocab_size = args.tgt_vocab_size = len(tokenizer)

model = Transformer(
    src_vocab_size=args.src_vocab_size,
    tgt_vocab_size=args.tgt_vocab_size,
    use_liger=args.use_liger,
)
model.load_state_dict(ckpt["MODEL_STATE"])
model.eval()

prompt = "Hello, how are you?"
translation = topk_sampling(
    model=model,
    prompt=prompt,
    tokenizer=tokenizer,
    device=args.device,
    max_length=args.block_size,
    top_k=50,
    temperature=0.8,
)
print("Prompt:", prompt)
print("Translation:", translation)
```

- `inference.py` also provides `beam_search_corrected` for deterministic decoding.
- `save_text()` writes sampled generations to `generated_data/`.

---

## Deploying a SageMaker Endpoint (optional)

1. Package an inference script (e.g., `serve.py`) implementing `model_fn`/`predict_fn` that:
   - Loads `snapshot_0.pt`, tokenizer files, and config from the model artifact.
   - Performs prompt tokenization & decoding without downloading assets at runtime.
2. Create the model:

   ```python
   from sagemaker.pytorch import PyTorchModel

   pytorch_model = PyTorchModel(
       model_data="s3://.../model.tar.gz",
       role=role,
       entry_point="serve.py",
       source_dir="...",
       framework_version="2.2",
       py_version="py310",
       env={"WANDB_DISABLED": "1"},
   )
   ```

3. Deploy & invoke:

   ```python
   predictor = pytorch_model.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.xlarge",   # CPU endpoint
       endpoint_name="compact-transformer-endpoint",
   )

   predictor.predict({"text": "Hello"})
   predictor.delete_endpoint()
   ```

> If invocations time out, ensure your tokenizer/model assets are bundled inside `model.tar.gz` and that inference code avoids long-running downloads.

---

## Next Steps

- Run long training sessions on GPU (larger block size/batch size, more iterations, multi-epoch).
- Enable W&B tracking with meaningful metrics & text samples.
- Experiment with additional Indic language pairs from Samanantar.
- Optimize inference (TorchScript, quantization, batching) once the model converges.
- Package a polished SageMaker endpoint/notebook for demo translations.

