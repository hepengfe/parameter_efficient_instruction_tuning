repo for parameter efficient instruction tuning
currently adapted from [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) and [peft](https://github.com/huggingface/peft).


To install the package, do the following.

* under `peft` folder, `pip install -e .`
* under `adapater-transfeomers` folder, `pip install -e .` to install adapter-transformer.
* `pip uninstall transfomers` uninstall the original transformer installed by `peft` to use adapter-transformer instead.
* rouge-score: `pip install rouge-score`


* 1.10.2+cu113 by platform.  "torch>=1.13.0" by peft setup.

* for hfai run, refer to scripts under `scripts/hfai` folder.
* for 