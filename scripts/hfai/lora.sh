# bash scripts/hfai/lora.sh lora_expr_49 49
# bash scripts/hfai/lora.sh lora_expr_145 145
# bash scripts/hfai/lora.sh lora_expr_484_ddp 484
expr_name=$1
peft_hp=$2

# hfai workspace push  --force --no_zip

# hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hf.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 3000 --save_steps 1000  --tuning_mode lora_peft  --learning_rate 5e-4 --lora_r $peft_hp --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500  -- --nodes 1 --no_inherit --name $expr_name

# hfai logs -f $expr_name

hfai workspace push  --force --no_zip

hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 3000 --save_steps 1000  --tuning_mode lora_peft  --learning_rate 5e-4 --lora_r $peft_hp --num_train_epochs 4 --dataset_name ni --data_dir "../../data/splits/default_train_707_val_50" --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500  -- --nodes 1 --no_inherit --name $expr_name

hfai logs -f $expr_name



