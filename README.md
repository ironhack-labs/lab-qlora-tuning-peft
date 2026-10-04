![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | QLoRA Tuning using PEFT

## Getting Started

Follow the instructions provided in the notebook.

Read the instructions for each cell and provide your answers. Make sure to test your answers in each cell and save. Jupyter Notebook should automatically save your work progress. But it's a good idea to periodically save your work manually just in case.

In the notebook, run the first dependency-install cell by itself, wait for `%pip` to finish, then restart the kernel before running the remaining cells. Do not select **Run All** until setup and restart are complete. The installer uses the pinned versions in `requirements-qlora.txt`, including `trl==0.9.6`, which provides the `SFTTrainer` imported by the lab.

In VS Code, select the `.venv (3.14.7)` notebook kernel for this workspace.

The lab uses a GPU with CUDA for actual 4-bit QLoRA. If CUDA is unavailable, it switches to a small pretrained GPT-2 model and runs LoRA in full precision so the workflow can still be practiced on CPU; that fallback is LoRA, not QLoRA. A local CUDA-enabled Windows setup is not detected by this workspace, so use a CUDA notebook runtime (for example, Colab) for the QLoRA path.

To train on your own expertise data, place `expert_data.jsonl` beside the notebook. Use one JSON object per line with `instruction`, optional `input`, and `output` string fields, for example:

```json
{"instruction":"Explain our return policy.","input":"Customer bought the item 20 days ago.","output":"Our policy allows returns within 30 days..."}
```

If the file is absent, the notebook uses 32 rows from the public Dolly instruction dataset as a working example.

## Deliverables

- Downloaded notebook with your responses to each of the exercises.

## Submission

- Upon completion, add your deliverables to git. 
- Then commit git and push your branch to the remote.
- Make a pull request and paste the PR link in the submission field in the Student Portal.

<br>

**Good luck!**
