"""
TRANSFORMERS_OFFLINE=1 python scripts/util/save_models.py
To simulate cluster env, disable `cache/model` by rename it to `cache/model_`
TODO: do we really need `cache/model`? If not, we can remove it.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForCausalLM, GPT2LMHeadModel
from transformers.adapters import AutoAdapterModel
from peft import PeftModelForSeq2SeqLM
import os


model_names = ("google/t5-small-lm-adapt", "google/t5-large-lm-adapt", "google/t5-xl-lm-adapt")

model_names = ("gpt2", )

model_names = ("facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b")# ,"facebook/opt-13b", )

model_names = ("facebook/opt-6.7b", )

# test llama
# tokenizer = AutoTokenizer.from_pretrained("cache/saved_pretrained/facebook/llama-7b")
# , use_fast=True, return_tensors="pt"
# model = AutoModelForSeq2SeqLM.from_pretrained("cache/saved_pretrained/facebook/llama-7b")

# exit()




# model_names = ("google/t5-xl-lm-adapt",)
# # cache_dir = "cache/model"
cache_dir="cache"
saved_pretrained_dir = os.path.join(cache_dir, "saved_pretrained")



# tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)
# tokenizer.save_pretrained(os.path.join(saved_pretrained_dir,"gpt2"))

# exit()

for m in model_names:
    # check if the model exists in the path
    # priority: 1. potential_model_path 2. cache_dir 3. download from huggingface
    
    if "gpt2" in m or "opt" in m or "llama" in m:
        model = AutoModelForCausalLM.from_pretrained(m, cache_dir=cache_dir)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(m, cache_dir=cache_dir)
    tokenizer =  AutoTokenizer.from_pretrained(m,
                                               use_fast=True,
                                                return_tensors="pt",
                                                truncation=True,
                                                max_length=512,
                                                )


    # AutoTokenizer.from_pretrained(os.path.join("cache/saved_pretrained", "gpt2"), max_length=1e5)

    model.save_pretrained(os.path.join(saved_pretrained_dir,m))
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