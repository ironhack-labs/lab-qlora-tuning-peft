**QLoRA Fine-Tuning Report**

**Project Overview**
This project explored parameter-efficient fine-tuning using the QLoRA method on BLOOM models. The goal was to adapt large language models to the target task while fitting within the memory limits of a 16 GB GPU. Several configurations were tested to evaluate the impact of model size and LoRA hyperparameters on performance and feasibility.

---

## Experiment 1

**Model:** BLOOM-7B1
**LoRA rank (r):** 14
**LoRA alpha:** 20
**Bias setting:** lora_only
**Epochs:** 3

**Result:**
After inference with the fine-tuned model, the performance was worse than the base model. The responses were less coherent and less accurate than expected. This suggests that either the rank was too low for this model size, the bias configuration limited learning capacity, or the training setup was not sufficient to adapt a 7B-parameter model effectively within the available resources.

---

## Experiment 2

**Model:** BLOOM-7B1
**LoRA rank (r):** 20
**LoRA alpha:** 20
**Bias setting:** all
**Epochs:** 3

**Result:**
Training could not be completed due to insufficient GPU memory. Increasing the LoRA rank and enabling bias updates across the model significantly increased memory usage, exceeding the limits of the available 16 GB GPU.

---

## Experiment 3

**Model:** BLOOM-560M
**LoRA rank (r):** 20
**LoRA alpha:** 20
**Bias setting:** all
**Epochs:** 3

**Result:**
This configuration produced much better results. The model trained successfully within the available memory, and the fine-tuned outputs were noticeably more accurate and coherent compared to the earlier experiments. The smaller model size allowed the LoRA parameters to have a stronger relative impact, improving performance.

---

## Observations

1. **Model size strongly affects feasibility.**
   The 7B model quickly reached memory limits, especially when increasing LoRA rank or updating all biases.

2. **LoRA configuration matters.**
   Lower-rank settings with limited bias updates may reduce learning capacity, leading to poorer results.

3. **Smaller models can outperform poorly-configured large models.**
   The 560M model, with more flexible LoRA settings, achieved better practical performance under the same hardware constraints.

---

## What did I learn?

* QLoRA enables fine-tuning large models on limited hardware, but configuration choices are critical.
* Memory usage increases significantly with higher LoRA ranks and broader bias updates.
* A smaller model with a well-chosen configuration can outperform a larger model that is under-trained or memory-constrained.
* Iterative experimentation is necessary to find the best balance between model size, memory limits, and performance.

---

## Conclusion

The experiments showed that while BLOOM-7B1 is theoretically more powerful, practical limitations made it difficult to fine-tune effectively on a 16 GB GPU. The BLOOM-560M configuration provided the best balance between memory usage and performance, making it the most suitable setup for this environment.
