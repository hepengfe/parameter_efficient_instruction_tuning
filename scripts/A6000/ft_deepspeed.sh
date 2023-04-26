# gpu 0,2 single group
CUDA_VISIBLE_DEVICES=0,2 accelerate launch  --config_file configs/accelerate_A6000/default_config_deepspeed.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 4 --per_device_eval_batch_size 30 --eval_steps 3000 --save_steps 3000 --tuning_mode fine_tuning --learning_rate 1e-5 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 10 --gradient_accumulation_steps 4 --do_train --use_accelerate 



