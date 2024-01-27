# # search best learning rate for fine tuning
# if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
#     hfai workspace push  --force --no_zip
# fi

# script_mode=$1
# model_name=$2

# i=0
# export LABEL_SMOOTHING_FACTOR=0
# export DROPOUT_RATE=0
# export WEIGHT_DECAY=0
# export RANDOM_SEED=127

# export CMD_INDEX=$i
# export MODEL_NAME=$model_name
# export ADAPATER_SIZE=512
# export LR=1e-4

# bash scripts/hfai/hp_run.sh adapter_peft $script_mode

# increased eval batch size from 1 to 4
hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt  --per_device_train_batch_size 1 --per_device_eval_batch_size 15 --eval_steps 5000 --save_steps 1000  --tuning_mode adapter_peft --adapter_size 512  --learning_rate 1e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_train  --do_traditional_test --gradient_accumulation_steps 2  --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_894 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/adapter_peft/sz_512/lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_894  --random_seed 894 --expr_dir cache/tmp --is_cluster -- --nodes 1 --no_inherit --force --name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_894

hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt  --per_device_train_batch_size 1 --per_device_eval_batch_size 15 --eval_steps 5000 --save_steps 1000  --tuning_mode adapter_peft --adapter_size 512  --learning_rate 1e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_train --do_traditional_test --gradient_accumulation_steps 2 --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/adapter_peft/sz_512/lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42  --random_seed 42 --expr_dir cache/tmp --is_cluster -- --nodes 1 --no_inherit --force --name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42

hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt  --per_device_train_batch_size 1 --per_device_eval_batch_size 15 --eval_steps 5000 --save_steps 1000  --tuning_mode adapter_peft --adapter_size 512  --learning_rate 1e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_train  --do_traditional_test --gradient_accumulation_steps 2  --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_127 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/adapter_peft/sz_512/lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_127  --random_seed 127 --expr_dir cache/tmp --is_cluster -- --nodes 1 --no_inherit --force --name ni_default_train_707_val_50_google_t5-xl-lm-adapt_adapter_peft_sz_512_lr_1e-4_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_127

