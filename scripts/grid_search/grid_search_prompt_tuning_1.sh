# bash scripts/hfai/hp_search.sh lora_adapter lora_r hfai 0
# bash scripts/hfai/hp_search.sh lora_adapter lora_r hfai 1
# bash scripts/hfai/hp_search.sh lora_adapter lr hfai 0

# LORA_RANKS=(8 32 64 128 256 512)

# flip the order of the bottleneck sizes
PROMPT_LENS=(256 128 32 8)
# BOTTLENECK_SIZES=(1024 512 256)
# DATA_FOLDERS=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
script_mode=$1

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01


for prompt_len in "${PROMPT_LENS[@]}"; do
    export PROMPT_LEN=$prompt_len

    for lr in "${lrs[@]}"; do
        export LR=$lr
        export CMD_INDEX=$i
        bash scripts/hfai/hp_run.sh prompt_tuning $script_mode
        ((i++))
        if [ $script_mode == "dev" ];then
            break
        fi
    done
    if [ $script_mode == "dev" ];then
        break
    fi
done