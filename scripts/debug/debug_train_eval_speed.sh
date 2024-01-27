accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp_2gpu_1.yaml prompt_tuning.py --model_name_or_path google/t5-base-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-base-lm-adapt_fine_tuning_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-base-lm-adapt/fine_tuning/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


# no dev mode
accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp_2gpu_1.yaml prompt_tuning.py --model_name_or_path google/t5-base-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 10 --save_steps 10 --tuning_mode fine_tuning --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-base-lm-adapt_fine_tuning_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-base-lm-adapt/fine_tuning/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --random_seed 42 --expr_dir cache/tmp


# self.training_args.num_train_epochs


hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hfai_ft.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt  --per_device_train_batch_size 1 --per_device_eval_batch_size 50 --eval_steps 5000 --save_steps 1000  --tuning_mode off_the_shelf --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 0 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_test --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_off_the_shelf_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/off_the_shelf/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42  --random_seed 42 --expr_dir cache/tmp --is_cluster -- --nodes 1 --no_inherit --force --name ni_default_train_707_val_50_google_t5-xl-lm-adapt_off_the_shelf_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42

accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp_3gpu_0.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 5000 --save_steps 1000 --tuning_mode lora_peft --lora_r 512 --lora_alpha 512 --lora_modules q,v --learning_rate 1e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-large-lm-adapt_lora_peft_r_512_alpha_512_modules_q,v_lr_1e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-large-lm-adapt/lora_peft/r_512_alpha_512_modules_q,v/lr_1e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


accelerate launch --config_file configs/accelerate_rtx3090/default_config_no_dist.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 1000 --save_steps 1000 --tuning_mode lora_peft --lora_r 512 --lora_alpha 512 --lora_modules q,v --learning_rate 1e-4 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-large-lm-adapt_lora_peft_r_512_alpha_512_modules_q,v_lr_1e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-large-lm-adapt/lora_peft/r_512_alpha_512_modules_q,v/lr_1e-4_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


* test prefix tuning
accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 5000 --save_steps 1000 --tuning_mode prefix_tuning --prefix_len 8 --bottleneck_size -1 --learning_rate 1e-5 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_prefix_tuning_prefix_len_8_bottleneck_size_1024_lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/prefix_tuning/prefix_len_8_bottleneck_size_1024/lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train

* test prefix tuning on single gpu
accelerate launch --config_file configs/accelerate_rtx3090/default_config_no_dist.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 5000 --save_steps 1000 --tuning_mode prefix_tuning --prefix_len 8 --bottleneck_size -1 --learning_rate 1e-5 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_prefix_tuning_prefix_len_8_bottleneck_size_1024_lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/prefix_tuning/prefix_len_8_bottleneck_size_1024/lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


* test bitfit
accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 10 --eval_steps 5000 --save_steps 1000 --tuning_mode bitfit --bias_name encoder_decoder_bias --learning_rate 1e-3 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_bitfit_None_lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/bitfit/None/lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


* 
accelerate launch --config_file configs/accelerate_rtx3090/default_config_no_dist.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 10 --eval_steps 5000 --save_steps 1000 --tuning_mode bitfit --bias_name encoder_decoder_bias --learning_rate 1e-3 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_bitfit_None_lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/bitfit/None/lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train


* ia3
accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 10 --eval_steps 5000 --save_steps 1000 --tuning_mode ia3 --learning_rate 1e-3 --scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.01 --label_smoothing_factor 0 --dropout_rate 0.1 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_ia3_None_lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/ia3/None/lr_1e-5_weight_decay_0.01_dropout_rate_0.1_label_smoothing_factor_0_scheduler_linear_warmup_ratio_0.03_random_seed_42 --random_seed 42 --expr_dir cache/tmp --dev_train --overwrite_output_dir





accelerate launch --config_file configs/accelerate_rtx3090/default_config_deepspeed_peft.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_traditional_test --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_fine_tuning_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/fine_tuning/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --random_seed 42 --expr_dir cache/tmp 

accelerate launch --config_file configs/accelerate_rtx3090/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --scheduler_type constant --warmup_ratio 0 --weight_decay 0 --label_smoothing_factor 0 --dropout_rate 0 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_707_val_50 --task_dir ../../data/tasks --do_traditional_test --gradient_accumulation_steps 2 --do_train --logging_steps 250 --run_name ni_default_train_707_val_50_google_t5-xl-lm-adapt_fine_tuning_None_lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --logging_dir ni/default_train_707_val_50/google_t5-xl-lm-adapt/fine_tuning/None/lr_1e-5_weight_decay_0_dropout_rate_0_label_smoothing_factor_0_scheduler_constant_warmup_ratio_0_random_seed_42 --random_seed 42 --expr_dir cache/tmp
