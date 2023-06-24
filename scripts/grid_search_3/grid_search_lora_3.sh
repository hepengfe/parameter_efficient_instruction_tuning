MODEL_NAMES=("google/t5-xxl-lm-adapt" "facebook/opt-13b" "facebook/llama-7b" )
# lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
script_mode=$1


i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01
export LR=5e-4
export RANDOM_SEED=$random_seed
lora_r=32

for model_name in "${MODEL_NAMES[@]}"; do

    export LORA_RANK=$lora_r
    export CMD_INDEX=$i
    export MODEL_NAME=$model_name
    bash scripts/hfai/hp_run.sh lora_adapter $script_mode &
    ((i++))
    if [ $script_mode == "dev" ];then
        break
    fi

done

