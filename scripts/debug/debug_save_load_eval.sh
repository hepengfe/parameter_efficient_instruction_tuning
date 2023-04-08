# test reduction factor
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 2 --mode adapter --lr 3e-4 --max_steps 10 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev --max_num_instances_per_eval_task 10 --overwrite_cache &> debug_save_load_eval.log

# --overwrite_cache



# model path will be after ':' in the last line of the log file
# read log file and extract model path
model_path=$(tail -n 1 debug_save_load_eval.log | cut -d ':' -f 2)


echo $model_path

# load previous checkpoint
CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model $model_path --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 1 --mode adapter --lr 3e-4 --max_steps 2 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev





# CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model ../peft_cache/google-t5-large-lm-adapt-ni-bf16-adapter-lr_0.0003_reduction_factor_20.0000_20230407-010248_max_num_instances_per_task_100/checkpoint-2 --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 1 --mode adapter --lr 3e-4 --max_steps 2 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev


# CUDA_VISIBLE_DEVICES=0  python prompt_tuning.py --model google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 2 --mode adapter --lr 3e-4 --max_steps 10 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --dev --max_num_instances_per_eval_task 10
CUDA_VISIBLE_DEVICES=0  python prompt_tuning.py --model google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_save_steps 2 --mode adapter --lr 3e-4 --max_steps 10 --dataset_name ni --data_dir data/splits/default_train_707_val_50 --task_dir data/tasks --predict_with_generate  --bf16 --reduction_factor 20 --max_num_instances_per_eval_task 10