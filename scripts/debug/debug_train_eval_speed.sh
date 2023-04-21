eval_save_steps=1
model="google/t5-small-lm-adapt"
per_device_eval_batch_size=36
# mode="lora"
mode="fine_tuning"
max_steps=320
# evaluation_strategy="epoch"
evaluation_strategy="steps"

# 20 steps, save one checkpoint
CUDA_VISIBLE_DEVICES=5  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --save_steps $eval_save_steps  --eval_steps $eval_save_steps --tuning_mode $mode --learning_rate 3e-4 --max_steps $max_steps --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir "../../data/tasks" --predict_with_generate  --max_num_instances_per_eval_task 50 --bf16 True --gradient_accumulation_steps 8 --evaluation_strategy $evaluation_strategy --dev True  --lr_scheduler_type constant

# --bf16  
# --dev True

# ft
#  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 1 --save_steps 1 --tuning_mode lora --learning_rate 3e-4 --max_steps 20 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train


#  epoch
#  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 1 --save_steps 1 --tuning_mode lora --learning_rate 3e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train

# TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0  python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 1 --save_steps 10 --tuning_mode lora --learning_rate 3e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train

# ACCELERATE_LOG_LEVEL=

# TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 10 --save_steps 5 --tuning_mode lora --learning_rate 3e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --dev_run



# dev training
# TRANSFORMERS_OFFLINE=1 accelerate launch --multi_gpu  --num_processes 3 prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 500 --save_steps 500 --tuning_mode lora --lora_r 10 --learning_rate 3e-4 --num_train_epochs 3 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --dev_run


# CUDA_VISIBLE_DEVICES=1,2,3   TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline accelerate launch --multi_gpu  --num_processes 3 prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 500 --save_steps 500 --tuning_mode lora --lora_r 508 --learning_rate 3e-4 --num_train_epochs 3 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --dev --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train



# fine tuning single gpu
# CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline accelerate launch prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 1000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --dev_run

# TRANSFORMERS_OFFLINE=1 cannot be used in multi-gpu, I think it's related to saved pretrained
WANDB_MODE=offline accelerate launch --multi_gpu  --num_processes 2 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 1000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_run

# bs = 3 x 6 = 18
WANDB_MODE=offline accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 15 --eval_steps 2000 --save_steps 2000 --tuning_mode lora --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 1 --do_train --dev_train


# ft train
# bs = 2 * 6 * 2 = 24
accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 1000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_train


# A6000 machine test
accelerate launch --multi_gpu  --num_processes 3 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 10 --eval_steps 1000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_train


accelerate launch --num_processes 6 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 5 --eval_steps 5000 --save_steps 5000 --tuning_mode lora --lora_r 255 --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_run


accelerate launch --num_processes 1 --gpu_ids='2,3' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 5000 --tuning_mode lora --lora_r 381 --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train


accelerate launch --num_processes 2 --gpu_ids='4,5' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 5000 --save_steps 5000 --tuning_mode lora --lora_r 508 --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train 


accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 1000 --save_steps 1000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --dev_run



# 500 steps reached 4.1524
# per step bs: 2 * 2 = 4
# num_of_data = 4 * 500 = 2000



# I think this version, training is not right (without acclerator)
# 3k 3.8669
# 6k 0.3286, predict token '0' for all instances, it's pretty similar to PEFT training + accelerator results 
# it's no acc launch and training incorrect
# PEFT one is acc launch and training incorrect 
# all though it's diff, there are some similarities
# 9k   
CUDA_VISIBLE_DEVICES=1 python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 6000 --save_steps 6000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train



cur_metric_val

# 6.7237
# /home/murphy/.cache/huggingface/accelerate/no_dist_config_gpu0.yaml
# low bs, high grad acc, acc launch
# 3k 5.6388
# 6k 6.8115
# 9k 8.9531
# 12k 13.2877
# 15k 14.0093
accelerate launch --config_file /home/murphy/.cache/huggingface/accelerate/no_dist_config_gpu0.yaml prompt_tuning.py  --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 False --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 4 --do_train --use_accelerate


# gpu 4 5
# /home/murphy/.cache/huggingface/accelerate/dist_config_gpu45.yaml
# 3k 15.3302
# 6k 17.7426
# 9k 17.4281   no improvement?
accelerate launch  --config_file /home/murphy/.cache/huggingface/accelerate/dist_config_gpu45.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 2 --do_train --use_accelerate


# validate grad acc assumption
WANDB_MODE=disabled accelerate launch  --config_file /home/murphy/.cache/huggingface/accelerate/dist_config_gpu45.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 8 --do_train --use_accelerate




# /home/murphy/.cache/huggingface/accelerate/dist_config_gpu23.yaml
# gradient acc = 1 + dist
# 3k 16.1147    it has reasonable output (might be inaccurate but answer format is correct)
# 6k 18.6458
# 9k
accelerate launch --main_process_port 29555 --config_file /home/murphy/.cache/huggingface/accelerate/dist_config_gpu23.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 1 --do_train --use_accelerate

# home/murphy/.cache/huggingface/accelerate/dist_config_gpu_all.yaml
# 3k 18.7513 (expected to be around 20 as it actually have bs=6 -> it's same as 9k steps for other training)
# 6k  20.2314
# 9k  21.1136 (22?) the training is improving but it's not as good as I expecte like ~35. 
# 12k 21.1174
# 15k 22.5378
# 18k 21.7561
WANDB_MODE=disabled accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/dist_config_gpu_all.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 1 --do_train --use_accelerate


# test deepspeed run
WANDB_MODE=disabled accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 1 --do_train --use_accelerate --dev_run

# official deepspeed run
  WANDB_MODE=offline accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 2000 --save_steps 2000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --use_accelerate 




accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 2000 --save_steps 2000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run

accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/dist_config_gpu_all.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 2000 --save_steps 2000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run


# single gpu save test


 accelerate launch --num_processes 1 --gpu_ids='0' --config_file  /home/murphy/.cache/huggingface/accelerate/no_dist_config_gpu0.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 2000 --save_steps 2000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run

# reduced eval bs to 5
accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml  prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 5 --eval_steps 2000 --save_steps 2000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 50 --gradient_accumulation_steps 6 --do_train --dev_run



