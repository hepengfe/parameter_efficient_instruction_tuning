eval_save_step=1000
# gpu 0,1 group1

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29566 --config_file configs/accelerate_rtx3090/default_config_deepspeed_2gpu.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps $eval_save_step --save_steps $eval_save_step --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 8 --do_train --use_accelerate


# gpu 3,4 group2

CUDA_VISIBLE_DEVICES=3,4 accelerate launch --main_process_port 29567 --config_file configs/accelerate_rtx3090/default_config_deepspeed_dev.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps $eval_save_step --save_steps $eval_save_step --tuning_mode fine_tuning --learning_rate 1e-4 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 8 --do_train --use_accelerate



CUDA_VISIBLE_DEVICES=2,5 accelerate launch --main_process_port 29568 --config_file configs/accelerate_rtx3090/default_config_deepspeed_dev.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 30 --eval_steps $eval_save_step --save_steps $eval_save_step --tuning_mode fine_tuning --learning_rate 1e-3 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 8 --do_train --use_accelerate --dev_run