accelerate launch --num_processes 2 --gpu_ids='4,5' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 5000 --tuning_mode lora_peft --lora_r 508 --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_train