# bash scripts/hfai/ft_data.sh dev -> for local run
# bash scripts/hfai/ft_data.sh dev_cmd -> for print command and check
# bash scripts/hfai/ft_data.sh cluster -> for cluster run

default_data_folder="default_train_707_val_50"
default_lr=1e-5

peft_method="fine_tuning"
default_peft_hp=64 # for lora rank only
# train_data_size/peft_method/peft_hp/lr 
default_eval_bs=15
default_model="google/t5-xl-lm-adapt"
default_dataset="ni"

# lora
default_lora_r=64
default_alpha=32
default_lora_modules="qv"


search_seq=(1e-5 5e-5 1e-4 5e-4 1e-3) # train data
search_seq=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
defaulat_eval_save_steps=1500
default_eval_step=3000
default_save_step=100

if [ $1 == "cluster" ]; then
    hfai workspace push  --force --no_zip
fi




for ((i=0; i<${#search_seq[@]}; i++))
    do
        
        # data
        data_folder="${search_seq[i]}"
        dataset=$default_dataset
        
        
        # training/evaluation
        eval_bs=$default_eval_bs
        eval_save_steps=$defaulat_eval_save_steps
        eval_step=$default_eval_step
        save_step=$default_save_step
        lr=$default_lr

        # peft config
        # ft

        tuning_setting="--tuning_mode ${peft_method} --learning_rate ${lr}"


        # expr name
        model_name=${default_model//\//_} # flatten "/" 
        # dataset/dataset_config/model/tuning_mode/tuning_config/lr
        # e.g.
        # ni/default_train_707_val_50/t5-xl-lm-adapt/ft/no_config/lora_r_64_alpha_32/5e-4
        # cot/xxx/llama-7b/lora_peft/lora_r_64/5e-4
        # we can uniform run_name/logging_dir in bash script rather than in python script
        tuning_config="r_${peft_hp}_alpha_${alpha}_modules_${lora_modules}" # for lora_peft
        tuning_config="None" # 
        # if hp search alpha, thne tuning_config="r_${lora_r}_alpha_${peft_hp}_modules_${lora_modules}" 
        expr_dir=${dataset}/${data_folder}/${model_name}/${peft_method}/${tuning_config}/lr_${lr}
        expr_name=${expr_dir//\//_} # replace "/" with "_"
        


        

        launch_prefix="hfai python hfai_accelerate.py  launch --config_file configs/hfai/default_config_deepspeed_hf.yaml"
        launch_suffix="--is_cluster -- --nodes 1 --no_inherit --name $expr_name"

        if [ $1 == "dev" ]; then
            launch_prefix="accelerate launch --config_file configs/accelerate_A6000/default_config_deepspeed_2gpu.yaml"
            launch_suffix="--dev_train"
        fi
        launch_command="${launch_prefix} prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size $eval_bs --eval_steps ${eval_step} --save_steps ${save_step}  ${tuning_setting} --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${data_folder} --task_dir ../../data/tasks --predict_with_generate  --gradient_accumulation_steps 2 --do_train --logging_steps 100 --run_name $expr_name --logging_dir $expr_dir $launch_suffix"

        if [ $1 == "dev_cmd" ];then
            echo "---------------cmd $i-----------------"
            echo "expr_name: $expr_name"
            echo "expr_dir: $expr_dir"
            echo "launch command: $launch_command"
            echo -e "\n\n"
        elif [[ $1 == "cluster" || $1 == "dev" ]];then
            echo $launch_command
            eval $launch_command
        fi

        if [ $1 == "dev" ];then
            break
        fi
done
