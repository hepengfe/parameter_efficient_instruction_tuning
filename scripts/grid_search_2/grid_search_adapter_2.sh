# bash scripts/grid_search/grid_search_adapter_1.sh dev_cmd t5

ADAPATER_SIZES=(8 32 64 128 256)
DATA_FOLDERS=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50")
ADAPATER_SIZES=(256)
DATA_FOLDERS=("default_train_32_val_50")
script_mode=$1
model_name=$2
random_seed=$3

i=0


export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01
export LR=1e-4
export RANDOM_SEED=$random_seed

for data_folder in "${DATA_FOLDERS[@]}"; do
    for adapter_size in "${ADAPATER_SIZES[@]}"; do
        export ADAPATER_SIZE=$adapter_size
        export CMD_INDEX=$i
        export MODEL_NAME=$model_name
        export DATA_FOLDER=$data_folder
        bash scripts/hfai/hp_run.sh adapter_peft $script_mode &
        ((i++))
        if [ $script_mode == "dev" ];then
            break
        fi
    done
    if [ $script_mode == "dev" ];then
        break
    fi
done




# bash scripts/hfai/hp_search.sh lora_adapter lr dev_cmd 0
# bash scripts/hfai/hp_search.sh lora_adapter lr dev_cmd 1