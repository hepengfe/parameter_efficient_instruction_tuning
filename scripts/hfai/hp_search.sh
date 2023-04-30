# bash scripts/hfai/hp_search.sh <hp_search_type>
# for example:
# bash scripts/hfai/hp_search.sh lora_lr       # rank=64, search for a starting lr and fix alpha lr
# bash scripts/hfai/hp_search.sh ft_data_size
# bash scripts/hfai/hp_search.sh adapter_lr     # rf=??
# bash scripts/hfai/hp_search.sh lora_rank
# bash scripts/hfai/hp_search.sh adapter_rf
# bash scripts/hfai/hp_search.sh adapter_data_size
# bash scripts/hfai/hp_search.sh lora_data_size


# lr hyper
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)


# peft hyper
lora_rank=(64 128 256 512 1024)
adapter_rf=(0.1 0.2 0.3 0.4 0.5)
eval_bss=(20 20 20 20 10 2) # for peft_hp only, higher trainable params, lower eval bs for 40GB GPU.
default_eval_bs=20


# data size
data_folders=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50")


# if else search_seq: peft_hp, lr, train_data_size
# length is 5 for alignment


# set search seq and set other hyperparameter to default
# TODO: generalize search seq condition, for example, lr -> search_seq=$lrs and make all other hyepers to be default 
if [ $1 == "lora_rank" ]; then
    search_seq=$lora_rank
elif [ $1 == "adapter_rf" ]; then
    search_seq=$adapter_rf
elif [ $1 == "lora_data_size" ]; then
    search_seq=$data_folders
elif [ $1 == "adapter_data_size" ]; then
    search_seq=$data_folders
elif [ $1 == "ft_data_size" ]; then
    search_seq=$data_folders
elif [ $1 == "adapter_lr" ]; then
    search_seq=$adapter_lr
elif [ $1 == "lora_lr" ]; then
    search_seq=$lora_lr
else
    echo "Wrong input"
    exit 1
fi

# two types of training mode
# deepspeed lower save/eval interval
# ddp higher save/eval interval




hfai workspace push  --force --no_zip


# expr name -> logging dir, run name prefix, hfai expr name
# expr name ->  train_data_size/peft_method/peft_hp/lr   needs to keep this constant for regex search
# this script can be generalized to other PEFT hyperparameter search
# for example, ia3 lr, prefix_tuning lr, prefix_tuning dimension, etc.

# debug script


# we can remove default element from search seq to avoid duplicate expr
# the following is the script for removing default element from search seq
# array=("apple" "banana" "cherry")
# remove_value="banana"
# for i in "${!array[@]}"; do
#     if [[ "${array[$i]}" == "${remove_value}" ]]; then
#         unset 'array[$i]'
#     fi
# done
# echo "${array[@]}"