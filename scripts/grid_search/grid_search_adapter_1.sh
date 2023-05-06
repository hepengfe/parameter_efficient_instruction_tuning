# bash scripts/hfai/hp_search.sh lora_adapter lora_r hfai 0
# bash scripts/hfai/hp_search.sh lora_adapter lora_r hfai 1
# bash scripts/hfai/hp_search.sh lora_adapter lr hfai 0
TRAINING_SETTINGS=(0 1)
# LORA_RANKS=(8 32 64 128 256 512)
ADAPATER_SIZES=(8 32 64 128 256)
# DATA_FOLDERS=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
script_mode=$1

i=0
for training_setting in "${TRAINING_SETTINGS[@]}"; do
    if [ $training_setting == 0 ]; then
        export LABEL_SMOOTHING_FACTOR=0
        export DROPOUT_RATE=0
        export WEIGHT_DECAY=0
    elif [ $training_setting == 1 ]; then
        export LABEL_SMOOTHING_FACTOR=0.1
        export DROPOUT_RATE=0.1
        export WEIGHT_DECAY=0.01
    else
        echo "Wrong training_setting"
        exit 1
    fi

    for lr in "${lrs[@]}"; do
        export LR=$lr
        for adapter_size in "${ADAPATER_SIZES[@]}"; do
            export ADAPATER_SIZE=$adapter_size
            export CMD_INDEX=$i
            bash scripts/hfai/hp_run.sh adapter $script_mode
            ((i++))
            if [ $script_mode == "dev" ];then
                break
            fi
        done
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