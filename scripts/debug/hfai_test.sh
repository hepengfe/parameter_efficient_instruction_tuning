#  WANDB_MODE=disabled hfai python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps --tuning_mode $mode --learning_rate 3e-4 --max_steps $max_steps --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --max_num_instances_per_eval_task 50 --bf16 False --gradient_accumulation_steps 8 --




HFAI_SIMULATE=1,WORLD_SIZE=1,RANK=0,MASTER_IP=127.0.0.1,MASTER_PORT=29510,MARSV2_WHOLE_LIFE_STATE=0 HF_ENV_NAME=peit hfai python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps --tuning_mode $mode --learning_rate 3e-4 --max_steps $max_steps --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --max_num_instances_per_eval_task 50 --bf16 False --gradient_accumulation_steps 8 --
