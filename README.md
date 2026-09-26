# LoRA Forge
### 918 Technologies · Standalone fine-tuning workbench · v1.0.0

[Download the standalone package](downloads/LoRA-Forge-Standalone-v1.0.0.zip) · [Verified training dashboard](verification/dashboard-trained.png)

Train real Hugging Face PEFT LoRA adapters through a local browser dashboard or CLI. No Kaggle notebook, competition harness, solver, paid API or cloud service is required. Model weights and training data stay on the machine running Forge. Model weights, Python and ML dependencies are not bundled.

## Start

Extract the ZIP. Install Python 3.10 or newer (3.12 tested).

**Linux / Ubuntu:** open a terminal in the extracted folder and run:

```bash
bash launch.sh
```

**Windows:** double-click `launch.cmd`, or run it in Command Prompt. A Python installation with the `py` launcher is required.

The launcher creates a virtual environment, installs training dependencies on its first run, and opens **http://127.0.0.1:9180**. Keep the terminal open. Enter your model folder, dataset paths and a new output directory; click **Check inputs**, then **Start forging**. First-time installation needs internet or a separately prepared wheelhouse. Offline assets is enabled by default and controls model/tokenizer downloads, not dependency installation.

For a remote GPU server, use `bash launch.sh --no-browser` and forward localhost port 9180 through SSH. The application intentionally listens only on loopback. An Android browser can control a forwarded session, but the phone does not perform the GPU training. Jetson TX2's older CUDA/Python stack is not supported by this release.

## What is included

- Real LoRA training; optional CUDA NF4 QLoRA.
- Strict model, dataset, tokenizer and projection validation before optimization.
- JSONL, JSON, CSV and final-assistant chat input.
- Prompt masking, actual EOS supervision and weighted final-answer loss.
- Named training phases, separate replay corpora, and bounded learning-rate reductions when heldout loss rises.
- SQLite provenance and split ledger, JSONL logs, live losses and run reports.
- Saved adapters plus optimizer, gradient scaler and RNG checkpoints; resume into a new run directory.
- Cooperative stop after the current optimizer step.
- Export gates for completed training, changed finite weights, rank/dimensions, heldout loss and PEFT reload equality.
- A real offline tiny-model integration test. It is clearly identified as a test fixture.

## Hardware and model scope

The release supports one training device: NVIDIA CUDA or CPU float32 for small models. A model must be supported by the installed Transformers/PEFT stack and fit on that device. NF4 requires installing `bitsandbytes` in the same environment:

```bash
.venv/bin/python -m pip install -e '.[train,qlora]'
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. GPU drivers and an appropriate CUDA-enabled PyTorch installation are required. `python -m loraforge doctor` reports availability.

Use original Hugging Face BF16/FP16 model weights. GGUF, externally prequantized FP8/GPTQ/AWQ checkpoints, automatic CPU/disk offload and multi-GPU training are outside v1. Nemotron's model-specific custom code/runtime may need additional packages; set `trust_remote_code: true` only for code you intend to load. No large-model or CUDA run was executed in the build environment. Large-model memory capacity cannot be inferred from the tiny-model test.

## CLI

From the activated environment:

```bash
python -m loraforge doctor
python -m loraforge init config.json
# Edit model and train_files in config.json.
python -m loraforge validate config.json
python -m loraforge train config.json
python -m loraforge serve
```

`validate` checks dataset schemas, duplicates, isolation and runtime availability. `train` additionally loads the tokenizer and model, checks every sequence and LoRA target, then starts optimization. A successful input check is not a GPU memory guarantee.

Set `offline: false` for a Hugging Face model ID that needs downloading. Existing Hugging Face cache and normal authentication are used; never place tokens in the configuration. Local model identity includes actual weight-file SHA-256 hashes, so a large checkpoint takes time to audit.

Paths in a CLI JSON file are relative to that JSON file. Paths entered or imported in the browser are relative to the server's working directory shown in the dashboard. Absolute paths avoid ambiguity.

## Dataset format

One JSON object per line:

```json
{"prompt":"What is 17 + 25?","answer":"42","category":"arithmetic","group_id":"problem-001"}
{"prompt":"Explain the result of this program...","reasoning":"A supplied, checked explanation.","answer":"The result is 7.","weight":1.2}
```

Alternatively use `instruction` + optional `input` + `output`, `question` + `answer`, or `prompt` + `completion`. Numeric zero is a valid answer. CSV uses the same columns. JSON can contain an array, or `data`, `records`, `train`, `examples`, or `rows` containing an array.

Chat input:

```json
{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]}
```

Only the final assistant completion is supervised; earlier messages provide context. The tokenizer's chat template is used when available. Otherwise Forge uses explicit USER/ASSISTANT text boundaries. Completion ends in the tokenizer's EOS token; full multi-turn tool-call training and arbitrary special assistant-end tokens are not covered in v1.

Exact prompts after whitespace/Unicode normalization are deduplicated. Conflicting answers fail. `group_id` keeps related problems in the same split; supply consistent group IDs for paraphrases. Without explicit `eval_files`, a deterministic group split holds out 10%. Replay data must be separate from both train and validation inputs. Near-duplicate semantic leakage cannot be detected automatically; curate group IDs.

A sequence exceeding `max_seq_length` fails before training with its source row. Forge never silently chops off a final answer or drops a record. Increase the limit within the model/GPU capacity or shorten the example yourself.

`answer_weight` increases the answer's loss relative to an optional `reasoning` segment. If every token has the same multiplier, normalization cancels that multiplier; it does not magically improve training. A record's `weight` scales its contribution even with a singleton microbatch; token emphasis is normalized separately from record confidence. No confidence is invented from category names.

The provided `examples/sample.jsonl` and `glyph-replay.jsonl` are deterministic starter data, not a meaningful benchmark or production training corpus. Example configs must be edited to point to your model.

## Phases, replay and measured auto-tuning

`examples/phased-retention.json` demonstrates warmup, broad learning, retention and polish. When `phases` is nonempty, its step counts replace `max_steps`. Each phase specifies its learning rate and optional `replay_fraction`; replay examples are drawn only from `replay_files`. With no replay file, all draws use training records.

Validation runs at checkpoints and phase boundaries. With `auto_tune: true`, a loss increase over the best observed heldout loss halves subsequent learning rates, down to 1/16 of the configured rate. This is bounded feedback, not a claim of reaching a target accuracy. The validation set is used for tuning and gating; keep an additional untouched test set for an unbiased quality estimate.

The final adapter is exported only if its validation loss is at most `baseline_loss * (1 + max_eval_loss_increase)`. Default tolerance is 5%. This checks final loss against the initial adapter, not competition scores. A rejected export leaves trained checkpoints and an explicit failure report.

`memory.sqlite` is a provenance/split ledger. It does not inject remembered validation answers into training or implement a retrieval model. Dataset text remains in your original input files.

## Stop and resume

Use **Stop & checkpoint** or create a file named `STOP` inside the current output directory. A long model load or optimizer step finishes before a cooperative stop can be processed. Closing the dashboard's server terminal terminates the worker and may lose work since its last checkpoint; use Stop first.

Copy the run's `config.json` to a convenient path. Keep all settings identical except `output_dir`, which must name a new/empty folder. Run:

```bash
python -m loraforge train resumed-config.json --resume /path/to/run/checkpoint-000025
```

Use checkpoints you created and trust. Resume checks configuration, input data, model/tokenizer, package versions and device identity. It restores optimizer/scaler state and RNG state. The CPU integration test verifies exact equality with uninterrupted execution. GPU kernels can be nondeterministic; bitwise CUDA equality is not promised. Resume continues the original schedule; extending a completed schedule or warm-starting from arbitrary adapters is outside v1.

## Run artifacts

| File | Meaning |
|---|---|
| `adapter.zip` | Verified two-file PEFT adapter export; exists only after successful completion |
| `adapter/` | Actual saved adapter tensors and PEFT configuration |
| `tokenizer/` | Matching tokenizer assets saved separately |
| `checkpoint-*/` | Adapter, optimizer/scaler/RNG state, and configuration |
| `report.json` | Explicit completion, export, evaluation and failure status |
| `events.jsonl` | Structured training and evaluation events |
| `memory.sqlite` | Input provenance, split identities and content hashes |
| `config.json` | Resolved run configuration |

The ZIP is a generic PEFT adapter, not the full base model. It is not automatically an ARC-AGI submission, `agent.py`, `submission.parquet`, or competition-scored result. No game harness/solver is part of this standalone fine-tuner.

Load a trained adapter with the same base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('/path/to/base')
model = PeftModel.from_pretrained(base, '/path/to/run/adapter')
tokenizer = AutoTokenizer.from_pretrained('/path/to/run/tokenizer')
model.eval()
```

## Verify your installation

```bash
python -m loraforge smoke --output runs/my-smoke-test
python -m unittest discover -s tests -v
```

The smoke test creates a tiny random Llama locally, trains real LoRA weights, verifies frozen base weights, reloads the export, cancels a second run, resumes it and compares final tensors. It uses no downloaded model. `verification/` contains the build's measured results and test log. This establishes pipeline behavior, not model quality.

See `docs/SOURCE_AUDIT.md` for the exact relationship to the supplied notebooks. Official API references: [PEFT quicktour](https://huggingface.co/docs/peft/quicktour), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
