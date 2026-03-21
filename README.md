# QLoRA Fine-Tuning with PEFT

Fine-tuning a quantized language model using LoRA adapters — without enterprise hardware.

**Notebook:** [lab-qlora-tuning-peft.ipynb](lab-qlora-tuning-peft.ipynb)

---

## What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) combines two techniques:

- **Quantization** — model weights are compressed from 32-bit floats to 4-bit integers (NF4 format), drastically reducing VRAM usage with acceptable precision loss.
- **LoRA** — instead of updating all weights, two small low-rank matrices are injected into attention layers. Only those adapters are trained; the base model stays frozen.

The result: fine-tuning a multi-billion parameter model on a single consumer GPU. Only 0.39% of parameters were trained here.

---

## Experiment

| | |
|---|---|
| **Base model** | `EleutherAI/pythia-410m` |
| **Dataset** | `tatsu-lab/alpaca` (~52k instruction-response pairs) |
| **Quantization** | 4-bit NF4 + double quantization via `bitsandbytes` |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **Target modules** | `query_key_value` |
| **Trainable params** | ~1.57M / 406.9M total (0.39%) |
| **Epochs** | 1 |
| **Final train loss** | 1.83 |

`pythia-410m` was chosen for fast iteration while being large enough to show meaningful behavioral shifts.

---

## Before vs. After

Same prompt, same model — before and after fine-tuning on Alpaca:

> **Prompt:** `"I want you to act as a motivational coach."`

| | Output |
|---|---|
| **Base model** | *"You are going on an adventure with me, and I need your help. We will be traveling through the land of 'What If'..."* |
| **Fine-tuned** | *"I will provide some information about an individual or group of people who need motivation, and your role is to help them find the inspiration they require..."* |

The fine-tuned model responds in the Alpaca instruction-following style — more focused, task-oriented, and structurally clean.

---

## What I Observed

**Prompt format matters more than expected.** Structured prompts (`### Instruction / ### Response`) produced markedly cleaner completions after fine-tuning. The model learned the template boundary.

**Masking the response tokens was necessary.** Without it, the model memorized the template structure and looped — generating new instructions instead of answers. Response-only label masking fixed this. Low training loss alone did not explain model behavior; structural correctness required changing the objective.

**Base LM ability was preserved.** Plain continuation prompts (e.g. `"The capital of France is..."`) produced normal completions with no template injection, confirming the adapter didn't over-steer.

**LoRA doesn't add intelligence — it reallocates capacity.** Instruction tuning changed *behavior*, not world knowledge. The model follows formats better; it doesn't learn new facts. The ceiling is the base model's reasoning depth.

---

## Tradeoffs

| Tradeoff | Detail |
|---|---|
| Speed vs. precision | 4-bit quantization enables training on limited hardware but introduces irreversible precision loss |
| Rank (r) vs. compute | Higher r = more expressive adapters, but more parameters and longer training |
| Behavior vs. knowledge | Fine-tuning on Alpaca shifts format adherence, not world knowledge |
| Small model ceiling | pythia-410m has limited reasoning depth; alignment improvements are real but bounded |

---

## Stack

- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Transformers](https://github.com/huggingface/transformers) + [TRL](https://github.com/huggingface/trl)
- PyTorch + CUDA
