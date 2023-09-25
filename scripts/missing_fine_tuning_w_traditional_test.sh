# search best learning rate for fine tuning
if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
    hfai workspace push  --force --no_zip
fi

script_mode=$1
model_name=$2

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0
export WEIGHT_DECAY=0
export RANDOM_SEED=127

export CMD_INDEX=$i
export MODEL_NAME=$model_name
export LR=1e-5

bash scripts/hfai/hp_run.sh fine_tuning $script_mode