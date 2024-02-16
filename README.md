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
* default ni dataset dir is `../../data` due to hfai compatibility.

## Platform specific
* hfai platform has CUDA version 11.3, and we use the corresponding pytorch version `1.10.2+cu113`.  `torch>=1.13.0` by peft setup.
* for hfai run, refer to scripts under `scripts/hfai` folder.
* for hfai run, make sure supress warning that contains `error` string. Otherwise, cluster will kill the job.