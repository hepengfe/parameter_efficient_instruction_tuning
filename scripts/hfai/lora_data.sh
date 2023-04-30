# bash scripts/hfai/lora_data.sh lora_train_task 77
expr_name=$1
peft_hp=$2
run_name_prefix="lora_num_train_task"

search_seq=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50") # train data 

hfai workspace push  --force --no_zip

for ((i=0; i<${#search_seq[@]}; i++))
    do
        # name is expr name + train data size
        hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 3000 --save_steps 1000  --tuning_mode lora_peft  --learning_rate 5e-4 --lora_r $peft_hp --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${search_seq[i]} --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500 --run_name $run_name_prefix -- --nodes 1 --no_inherit --name ${expr_name}_${search_seq[i]}
done