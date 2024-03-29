GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


accelerate launch --config_file configs/accelerate/default_config_deepspeed.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 3 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000 --tuning_mode fine_tuning  --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --dev_run
accelerate launch --multi_gpu  --num_processes $GPU_COUNT --gpu_ids='all' --config_file /home/murphy/.cache/huggingface/accelerate/default_deepspeed_config.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000 --tuning_mode fine_tuning  --learning_rate 5e-4 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train



# no offload  3s/iter, bs=1,
# offload bs=2, 4s/iter
# ddp, 1s/iter, bs=1