
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29556 --config_file configs/accelerate_rtx3090/default_config_deepspeed_dev.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_adapter --lora_r 508 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate --dev_train
