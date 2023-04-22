#  WANDB_MODE=disabled hfai python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps --tuning_mode $mode --learning_rate 3e-4 --max_steps $max_steps --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --max_num_instances_per_eval_task 50 --bf16 False --gradient_accumulation_steps 8 --



# 
HFAI_SIMULATE=1,WORLD_SIZE=1,RANK=0,MASTER_IP=127.0.0.1,MASTER_PORT=29510,MARSV2_WHOLE_LIFE_STATE=0 HF_ENV_NAME=peit3 hfai python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps --tuning_mode $mode --learning_rate 3e-4 --max_steps $max_steps --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --max_num_instances_per_eval_task 50 --bf16 False --gradient_accumulation_steps 8 --

# local test hfai with accelerate

HF_ENV_NAME=peit3 hfai accelerate launch --multi_gpu  --num_processes 6 --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run --



/hf_shared/hfai_envs/wangyizhong/peit3_0


conda deactivate;conda deactivate;conda activate peft2;source haienv peit3

# local test hfai without accelerate
hfai workspace push  --force --no_zip; HF_ENV_NAME=peit3 hfai python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run --is_cluster -- --nodes 1 --no_inherit ;hfai logs -f prompt_tuning.py 



python -m accelerate prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000 --tuning_mode adapter --reduction_factor 2.29  --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run --is_cluster