# test reduction factor
# CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 2 --tuning_mode adapter --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev --max_num_instances_per_eval_task 10  > debug_save_load_eval.log

# --overwrite_cache

# # test lora
# CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2 --tuning_mode lora --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10 > debug_save_load_eval.log

# # test compactor
# CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2 --tuning_mode compactor --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10 --reduction_factor 20  > debug_save_load_eval.log


# test prefix_tuning
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2 --tuning_mode prefix_tuning --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10 --prefix_len 10  > debug_save_load_eval.log

# ia3
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2 --tuning_mode ia3 --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10


# parallel_adapter
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2 --tuning_mode parallel_adapter --lr 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10 --reduction_factor 20

# model path will be after ':' in the last line of the log file
# read log file and extract model path
model_path=$(tail -n 1 debug_save_load_eval.log | cut -d ':' -f 2)


echo "loading model from $model_path"

# load previous checkpoint
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path $model_path --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 2 --save_steps 2  --tuning_mode adapter --lr 3e-4 --max_steps 2 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev




# test wandb
# CUDA_VISIBLE_DEVICES=0  python prompt_tuning.py --model google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 2 --tuning_mode adapter --lr 3e-4 --max_steps 10 --dataset_name ni --data_dir data/splits/default_train_707_val_50 --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --max_num_instances_per_eval_task 10





# CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path ../peft_cache/google-t5-small-lm-adapt-ni-bf16-adapter-lr_0.0003_reduction_factor_20.0000_20230408-131152data_splits_default_max_num_instances_per_task_100/checkpoint-2 --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 1 --tuning_mode adapter --lr 3e-4 --max_steps 2 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev