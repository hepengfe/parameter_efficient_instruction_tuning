# bash scripts/hfai/ft.sh expr_name
expr_name=$1

# python accelerate_cli.py  launch --config_file /weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit/configs/hfai/default_config_deepspeed.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 5 --per_device_eval_batch_size 100 --eval_steps 6000 --save_steps 6000  --tuning_mode fine_tuning  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --is_cluster 



hfai workspace push  --force --no_zip

hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hf.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 6000 --save_steps 3000  --tuning_mode fine_tuning  --learning_rate 5e-5 --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500 --dev_train -- --nodes 1 --no_inherit --name $expr_name

hfai logs -f $expr_name



# python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_mig.yaml prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 5 --per_device_eval_batch_size 100 --eval_steps 6000 --save_steps 6000  --tuning_mode fine_tuning  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --is_cluster --dev_run

# python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hf.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 20 --eval_steps 6000 --save_steps 6000  --tuning_mode fine_tuning  --learning_rate 1e-5 --num_train_epochs 2 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 6 --do_train --is_cluster --dev_train