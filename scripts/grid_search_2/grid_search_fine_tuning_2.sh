# learning rate search is finished
DATA_FOLDERS=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
lrs=(1e-5)
script_mode=$1
model_name=$2
random_seed=$3

i=0
export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0
export WEIGHT_DECAY=0
export RANDOM_SEED=$random_seed

lr=1e-5

for data_folder in "${DATA_FOLDERS[@]}"; do
    
    export LR=$lr
    export CMD_INDEX=$i
    export MODEL_NAME=$model_name
    export DATA_FOLDER=$data_folder
    bash scripts/hfai/hp_run.sh fine_tuning $script_mode &
    ((i++))
    if [ $script_mode == "dev" ];then
        break
    fi
done
