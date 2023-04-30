
# 0,1

# peft

# 3,4

CUDA_VISIBLE_DEVICES=3,4 accelerate launch --main_process_port 29556 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_peft --lora_r 77 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate

# 2,5
CUDA_VISIBLE_DEVICES=2,5 accelerate launch --main_process_port 29557 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_peft --lora_r 77 --learning_rate 0.001 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate



# 2,5
CUDA_VISIBLE_DEVICES=2,5 accelerate launch --main_process_port 29557 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_adapter --lora_r 77 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate



# 0,1 lora 381
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29557 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_adapter --lora_r 381 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 16 --do_train --use_accelerate

# 2,5 lora 255
CUDA_VISIBLE_DEVICES=2,5 accelerate launch --main_process_port 29558 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_adapter --lora_r 255 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 16 --do_train --use_accelerate



# 3,4  lora 508, lower bs, higher gradient acc
CUDA_VISIBLE_DEVICES=3,4 accelerate launch --main_process_port 29559 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_peft --lora_r 508 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 16 --do_train --use_accelerate --overwrite_output_dir


# single gpu testing

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29556 --config_file configs/accelerate_rtx3090/default_config_deepspeed_dev.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode lora_adapter --lora_r 508 --learning_rate 5e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate --dev_train


# since lora computation graph is keeped in memory, the memory usage is not reduced. 