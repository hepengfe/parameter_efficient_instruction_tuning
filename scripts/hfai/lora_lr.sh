# bash scripts/hfai/lora_lr.sh dev

default_data_folder="default_train_707_val_50"
default_lr=5e-4
search_seq=(1e-5 5e-5 1e-4 5e-4 1e-3) # train data
peft_method="lora_peft"
default_peft_hp=64
# train_data_size/peft_method/peft_hp/lr 
default_eval_bs=20



if [ $1 != "dev" ]; then
    hfai workspace push  --force --no_zip
fi




for ((i=0; i<${#search_seq[@]}; i++))
    do
        data_folder=$default_data_folder
        peft_hp=$default_peft_hp
        data_folder=$default_data_folder
        lr=${search_seq[i]}
        # dataset/dataset_config/model/tuning_mode/tuning_config/lr
        # e.g.
        # ni/default_train_707_val_50/t5-xl-lm-adapt/ft/no_config/lora_r_64_alpha_32/5e-4
        # cot/xxx/llama-7b/lora_peft/lora_r_64/5e-4
        # we can uniform run_name/logging_dir in bash script rather than in python script
        expr_dir=${data_folder}/${peft_method}/lora_r_${peft_hp}/${lr}
        expr_name=${expr_dir//\//_} # replace "/" with "_"
        
        tuning_setting="--tuning_mode ${peft_method} --lora_r ${peft_hp} --learning_rate ${lr}"

        # variables unrelated to name
        eval_bs=$default_eval_bs

        launch_commad="hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_ddp.yaml"
        launch_suffix="--is_cluster -- --nodes 1 --no_inherit --name $expr_name"

        if [ $1 == "dev" ]; then
            launch_commad="accelerate launch --config_file configs/accelerate_A6000/default_config_ddp.yaml"
            launch_suffix="--dev_train"
        fi
        # name is expr name + train data size
        ${launch_commad} prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size $eval_bs --eval_steps 3000 --save_steps 1000  ${tuning_setting} --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${data_folder} --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --logging_steps 500 --run_name $expr_name --logging_dir $expr_dir $launch_suffix

        if [ $1 == "dev" ];then
            break
        fi
done
