# bash scripts/grid_search/grid_search_adapter_1.sh dev_cmd t5


MODEL_NAMES=("google/t5-xxl-lm-adapt" "google/t5-base-lm-adapt" "google/t5-large-lm-adapt" )
# temp changes
MODEL_NAMES=("google/t5-base-lm-adapt" "google/t5-xxl-lm-adapt")
script_mode=$1


i=0


export LABEL_SMOOTHING_FACTOR=0
export DROPOUT_RATE=0.1
export WEIGHT_DECAY=0.01
export LR=1e-4
adapter_size=256

for model_name in "${MODEL_NAMES[@]}"; do

    export ADAPATER_SIZE=$adapter_size
    export CMD_INDEX=$i
    export MODEL_NAME=$model_name
    bash scripts/hfai/hp_run.sh adapter $script_mode &
    ((i++))
    if [ $script_mode == "dev" ];then
        break
    fi

done




# bash scripts/hfai/hp_search.sh lora_adapter lr dev_cmd 0
# bash scripts/hfai/hp_search.sh lora_adapter lr dev_cmd 1
hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hfai_peft.yaml prompt_tuning.py --model_name_or_path google/t5-xxl-lm-adapt  --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 5000 --save_steps 1000  --tuning_mode lora_peft --lora_r 32 --lora_alpha 32 --learning_rate 5e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --predict_with_generate  --gradient_accumulation_steps 2 --do_train  --logging_steps 500 --run_name ni_default_train_707_val_50_google_t5-xxl-lm-adapt_lora_peft_r_32_alpha_32_modules_qv_lr_5e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03 --logging_dir ni/default_train_707_val_50/google_t5-xxl-lm-adapt/lora_peft/r_32_alpha_32_modules_qv/lr_5e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03 --is_cluster -- --nodes 1 --no_inherit --force --name ni_default_train_707_val_50_google_t5-xxl-lm-adapt_lora_peft_r_32_alpha_32_modules_qv_lr_5e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03