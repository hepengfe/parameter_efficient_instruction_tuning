"""
TRANSFORMERS_OFFLINE=1 python scripts/util/save_models.py
To simulate cluster env, disable `cache/model` by rename it to `cache/model_`
TODO: do we really need `cache/model`? If not, we can remove it.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.adapters import AutoAdapterModel
from peft import PeftModelForSeq2SeqLM
import os


model_names = ("google/t5-small-lm-adapt", "google/t5-large-lm-adapt", "google/t5-xl-lm-adapt")
# cache_dir = "cache/model"
cache_dir="cache"
saved_pretrained_dir = os.path.join(cache_dir, "saved_pretrained")


for m in model_names:
    # check if the model exists in the path
    # priority: 1. potential_model_path 2. cache_dir 3. download from huggingface
    

    # model = AutoModelForSeq2SeqLM.from_pretrained(m, cache_dir=cache_dir)
    tokenizer =  AutoTokenizer.from_pretrained(m,use_fast=True,
                return_tensors="pt")

    # model.save_pretrained(f"model/{m}")
    tokenizer.save_pretrained(os.path.join(saved_pretrained_dir,m))


    # if os.path.exists(potential_model_path):
    # potential_model_path = os.path.join(saved_pretrained_dir, m)
    #     # the following two are for testing loading only, comment out when running the script
    #     # model=AutoAdapterModel.from_pretrained(potential_model_path)
    #     # model=PeftModelForSeq2SeqLM.from_pretrained(potential_model_path, "0")
        
    #     import pdb; pdb.set_trace()
    #     print('')
        
    #     model = AutoModelForSeq2SeqLM.from_pretrained(potential_model_path) # if potential model path exists, it should be able load it directly without downloading from huggingface and caching it in cache_dir
        
    # else: