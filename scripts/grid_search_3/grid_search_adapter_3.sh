# bash scripts/grid_search/grid_search_adapter_1.sh dev_cmd t5


MODEL_NAMES=("google/t5-xxl-lm-adapt" "google/t5-base-lm-adapt" "google/t5-large-lm-adapt" )
# temp changes
MODEL_NAMES=("google/t5-base-lm-adapt" "google/t5-xxl-lm-adapt")
MODEL_NAMES=("google/t5-xxl-lm-adapt")
script_mode=$1


i=0


export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01
export LR=1e-4
export RANDOM_SEED=42
adapter_size=256

for model_name in "${MODEL_NAMES[@]}"; do

    export ADAPATER_SIZE=$adapter_size
    export CMD_INDEX=$i
    export MODEL_NAME=$model_name
    bash scripts/hfai/hp_run.sh adapter_peft $script_mode &
    ((i++))
    if [ $script_mode == "dev" ];then
        break
    fi

done
