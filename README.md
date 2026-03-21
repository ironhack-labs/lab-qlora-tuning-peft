# QLoRA Fine-Tuning with PEFT

This technique makes it feasible to adapt large language models to specific tasks without access to enterprise-grade hardware.

## What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) combines two techniques:

- **Quantization**: model weights are compressed from 32-bit floats to 4-bit integers (NF4 format), drastically reducing VRAM usage with acceptable precision loss.
- **LoRA**: instead of updating all weights, two small low-rank matrices are injected into attention layers. Only those adapters are trained — the base model stays frozen.

The result: fine-tuning a multi-billion parameter model on a single consumer GPU.

You can find the code here: [lab-qlora-tuning-peft.ipynb](https://github.com/marcosfsousa/lab-qlora-tuning-peft/blob/main/lab-qlora-tuning-peft.ipynb)

## Experiment

| | |
|---|---|
| **Base model** | `EleutherAI/pythia-410m` |
| **Dataset** | `tatsu-lab/alpaca` (instruction-following, ~52k samples) |
| **Quantization** | 4-bit NF4 + double quantization via `bitsandbytes` |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **Target modules** | `query_key_value` |
| **Trainable params** | ~1.57M / 406.9M total (0.39%) |
| **Epochs** | 1 |
| **Final train loss** | 1.83 |

The model was chosen for fast iteration (small parameter count) while still being large enough to show meaningful behavioral shifts.

## Key Findings

**Prompt format matters more than you'd expect.** The base model responded reasonably to plain prompts, but after fine-tuning on Alpaca's instruction-response format, structured prompts (`### Instruction / ### Response`) produced markedly cleaner, more focused completions.

**Masking the response tokens was necessary.** Without masking the prompt portion of the loss, the model memorized the template structure and looped it — generating new instructions instead of responses. Applying response-only labels fixed this.

**Base LM ability was preserved.** Plain continuation prompts (e.g. "The capital of France is...") produced normal completions with no template injection, confirming the adapter didn't over-steer the model.

## Tradeoffs

| Tradeoff | Detail |
|---|---|
| Speed vs. capacity | 4-bit quantization enables training on limited hardware but introduces irreversible precision loss |
| Rank (r) vs. compute | Higher r = more expressive adapters, but more parameters and longer training |
| Instruction tuning vs. knowledge | Fine-tuning on Alpaca changes *behavior*, not world knowledge — the model follows formats better but doesn't learn new facts |
| Small model ceiling | pythia-410m has limited reasoning depth; alignment improvements are real but bounded by base capacity |

## Stack

- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Transformers](https://github.com/huggingface/transformers) + [TRL](https://github.com/huggingface/trl)
- PyTorch + CUDA
