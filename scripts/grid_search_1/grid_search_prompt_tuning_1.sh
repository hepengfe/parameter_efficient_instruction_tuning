# bash scripts/grid_search_1/grid_search_prompt_tuning_1.sh dev_cmd t5
# LORA_RANKS=(8 32 64 128 256 512)

# flip the order of the bottleneck sizes
PROMPT_LENS=(256 128 32 8)
# BOTTLENECK_SIZES=(1024 512 256)
# DATA_FOLDERS=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
script_mode=$1
model_name=$2

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01


for prompt_len in "${PROMPT_LENS[@]}"; do
    export PROMPT_LEN=$prompt_len

    for lr in "${lrs[@]}"; do
        export LR=$lr
        export CMD_INDEX=$i
        export MODEL_NAME=$model_name
        bash scripts/hfai/hp_run.sh prompt_tuning $script_mode &
        ((i++))
        if [ $script_mode == "dev" ];then
            break
        fi
    done
    if [ $script_mode == "dev" ];then
        break
    fi
done