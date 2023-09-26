# # search best learning rate for fine tuning
# if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
#     hfai workspace push  --force --no_zip
# fi

# script_mode=$1


# i=0
# export LABEL_SMOOTHING_FACTOR=0
# export DROPOUT_RATE=0
# export WEIGHT_DECAY=0
# export RANDOM_SEED=127

# export CMD_INDEX=$i
# export MODEL_NAME="t5"
# export LR=1e-5

# bash scripts/hfai/hp_run.sh fine_tuning $script_mode


# accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp_2gpu_1.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_traditional_test --gradient_accumulation_steps 8 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_fine_tuning_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_127 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/fine_tuning/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_127 --random_seed 127 --expr_dir /media/nvme_1

# what's needs to be changed: --expr_dir is the output dir
# --config_file should be ddp
# --eval_steps should be 5000 * (8/NUM_GPUS) to match 8XA100 evaluation steps