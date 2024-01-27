
MODEL_NAMES=("google/t5-base-lm-adapt" "google/t5-large-lm-adapt" "google/t5-xl-lm-adapt" "google/t5-xxl-lm-adapt")
script_mode=$1
random_seed=$2

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0
export WEIGHT_DECAY=0
export LR=1e-5
export RANDOM_SEED=$random_seed

for model_name in "${MODEL_NAMES[@]}"; do
    export CMD_INDEX=$i
    export MODEL_NAME=$model_name
    bash scripts/hfai/hp_run.sh fine_tuning $script_mode &
    ((i++))
    if [ $script_mode == "dev" ];then
        break
    fi
done
