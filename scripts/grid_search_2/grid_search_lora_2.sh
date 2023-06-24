LORA_RANKS=(8 32 64 128 256 512)
DATA_FOLDERS=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50")
# lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
# temp
DATA_FOLDERS=("default_train_8_val_50")
LORA_RANKS=(8)
script_mode=$1
model_name=$2
random_seed=$3

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01
export LR=5e-4
export RANDOM_SEED=$random_seed


for data_folder in "${DATA_FOLDERS[@]}"; do
    for lora_r in "${LORA_RANKS[@]}"; do
        export LORA_RANK=$lora_r
        export CMD_INDEX=$i
        export MODEL_NAME=$model_name
        export DATA_FOLDER=$data_folder
        bash scripts/hfai/hp_run.sh lora_adapter $script_mode &
        ((i++))
        if [ $script_mode == "dev" ];then
            break
        fi
    done
done

