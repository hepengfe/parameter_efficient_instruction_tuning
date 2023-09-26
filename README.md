repo for parameter efficient instruction tuning
currently adapted from [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) and [peft](https://github.com/huggingface/peft).


To install the package, do the following.

* `conda create -n peft python=3.8`
* `git clone https://github.com/hepengfe/peft-private.git` and checkout `git checkout release-v0.4.0-adapter` branch. Under `peft-private` folder, `pip install -e .` to install peft-private.
<!-- * under `adapater-transfeomers` folder, `pip install -e .` to install adapter-transformer.
* `pip uninstall transfomers` uninstall the original transformer installed by `peft` to use adapter-transformer instead. -->

* rouge-score: `pip install rouge-score`
* make sure GPT2 is under `cache/saved_pretrained/gpt2` for evaluation
* default ni dataset dir is `../../data` due to hfai compatibility.


* 1.10.2+cu113 by hfai platform.  "torch>=1.13.0" by peft setup.
* for hfai run, refer to scripts under `scripts/hfai` folder.
* for hfai run, make sure supress warning that contains `error` string. Otherwise, cluster will kill the job.


