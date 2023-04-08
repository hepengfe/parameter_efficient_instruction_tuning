repo for parameter efficient instruction tuning
currently adapted from [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) and [peft](https://github.com/huggingface/peft).


To install the package, do the following.

* under `peft` folder, `pip install -e .`
* under `adapater-transfeomers` folder, `pip install -e .` to install adapter-transformer.
* `pip uninstall transfomers` uninstall the original transformer to use adapter-transformer instead.
* rouge-score: `pip install rouge-score`