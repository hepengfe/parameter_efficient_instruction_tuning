## Parameter efficient instruction tuning: an Empirical Study
This repository serves as an effort to systematically to compare different parameter efficient fine-tuning methods on instruction tuning task. We use the [NI dataset](https://github.com/allenai/natural-instructions) as the benchmark dataset.

PEFT method implementations are adapted from [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) and [peft](https://github.com/huggingface/peft).





## Training
* For bash scripts to run all experiments, refer to `scripts` folder.
* All experiments calling scripts are formatted by `scripts/hfai/hp_run.sh`. For example, to run a LoRA experimental in the development mode, run the following command.
```bash
bash scripts/hfai/hp_run.sh lora_peft dev
```
## Dataset
* We employ [SuperNI](https://github.com/allenai/natural-instructions) as our training and evaluation datasets.

## Setup
To install the package, do the following.

* `conda create -n peft python=3.8`
* `git clone https://github.com/hepengfe/peft-private.git` and checkout `git checkout release-v0.4.0-adapter` branch. Under `peft-private` folder, `pip install -e .` to install peft-private.
<!-- * under `adapater-transfeomers` folder, `pip install -e .` to install adapter-transformer.
* `pip uninstall transfomers` uninstall the original transformer installed by `peft` to use adapter-transformer instead. -->

* rouge-score: `pip install rouge-score`
* make sure GPT2 is under `cache/saved_pretrained/gpt2` for evaluation




## HPC Platform specific
The platform we use is [hfai HPC](https://www.high-flyer.cn/en/). Each node is equipped with A100x8 GPU, and each of our experiments is runing on a single node.

This codebase is highly optimized for hfai platform, and it supports the following functionalities:
* **Experiment Configuration and Submission**: The `hp_run.sh` scripts allows for flexible adjustments to experiment name, batch size and training framework based on fine-tuning methods. To launch multiple jobs based on `hp_run.sh`, refer to scripts under `scripts/hfai` folder.
* **Checkpoint Management**: Since the platform is pre-emptable, our codebase supports checkpoints saving upon suspension and resuming from the last checkpoint. Each experiment is assumed to be run until complete test dataset evaluation.
* **Training State Validation**: When saving checkpoint, we support checking the completeness of training state, training random states. Otherwise, it will delete the latest checkpoint and needs re-run the experiment loading second-to-last checkpoint.
* **System Message and Debugging**: We let most system message output by print statement because it's more suitable for multi-process debugging, and we suppress warnings that contain `error` string to avoid job killing.

Here are some extra notes for hfai platform:
* default ni dataset dir is `../../data` due to hfai compatibility.
* **Pytorch and CUDA compatibility**: hfai platform has CUDA version 11.3, and [peft](https://github.com/huggingface/peft) setup requires `torch>=1.13.0`. Therefore, we use the corresponding pytorch version `1.10.2+cu113` by peft setup.
