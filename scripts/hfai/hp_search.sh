# bash scripts/hfai/hp_search.sh <tuning_method>  <hp_to_search> <script_mode>
# bash scripts/hfai/hp_search.sh fine_tuning data_size cluster
# bash scripts/hfai/hp_search.sh lora_peft lr dev_cmd
# bash scripts/hfai/hp_search.sh adapter adapter_size dev
# bash scripts/hfai/hp_search.sh lora_peft lora_r dev_cmd

tuning_mode=$1
hp_to_search=$2
script_mode=$3

default_data_folder="default_train_707_val_50"







# train_data_size/peft_method/peft_hp/lr 

default_model="google/t5-xl-lm-adapt"
default_dataset="ni"

# lora
default_lora_r=64
default_lora_alpha=32
default_lora_modules="qv"

# adapter
default_adapter_size=64


eval_bss=(20 20 20 10 2) # for peft_hp only, higher trainable params, lower eval bs for 40GB GPU.
# two types of training mode
# deepspeed lower save/eval interval
if [ $tuning_mode == "fine_tuning" ]; then
    lr=1e-5
    config_file="configs/hfai/default_config_deepspeed_hf.yaml"
    default_eval_step=5000
    default_eval_bs=15
elif [ $tuning_mode == "lora_peft" ]; then
    lr=5e-4
    # lr=1e-4
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    default_eval_bs=20 # adapter < 15, lora < 20
    eval_bss=(20 20 20 10 2) # for peft_hp only, higher trainable params, lower eval bs for 40GB GPU.
elif [ $tuning_mode == "adapter" ]; then
    lr=5e-4
    # lr=1e-4
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    default_eval_bs=15 # adapter < 15, lora < 20
    eval_bss=(15 15 10 5 1) # for peft_hp only, higher trainable params, lower eval bs for 40GB GPU.
fi



# hyper seq
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
lora_ranks=(64 128 256 512 1024)
adapter_rf=(0.1 0.2 0.3 0.4 0.5)
adapter_szs=(64 128 256 512 1024)



data_folders=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")




# default settings if it's not in hp to search



# if else search_seq: peft_hp, lr, train_data_size
# length is 5 for alignment


# set search seq and set other hyperparameter to default
# TODO: generalize search seq condition, for example, lr -> search_seq=$lrs and make all other hyepers to be default 

if [ $hp_to_search == "lr" ]; then
    search_seq=("${lrs[@]}")
    eval_bs=$default_eval_bs
elif [ $hp_to_search == "data_size" ]; then
    search_seq=("${data_folders[@]}")
    eval_bs=$default_eval_bs
else
    if [ $tuning_mode == "lora_peft" ]; then
        if [ $hp_to_search == "lora_r" ]; then
            search_seq=("${lora_ranks[@]}")
            eval_bs=$default_eval_bs
        else
            echo "Wrong input"
            exit 1
        fi
    elif [ $tuning_mode == "adapter" ]; then
        if [ $hp_to_search == "adapter_size" ]; then
            search_seq=("${adapter_szs[@]}")
            eval_bs=$default_eval_bs
        else
            echo "Wrong input"
            exit 1
        fi
    else
        echo "Wrong tuning_mode"
        exit 1
    fi
fi



# ddp higher save/eval interval
default_save_step=$((default_eval_step/5)) # 5000/5=1000
defualt_logging_steps=$((default_eval_step/20)) # 5000/20=250

if [ $script_mode == "cluster" ]; then
    hfai workspace push  --force --no_zip
fi


for ((i=0; i<${#search_seq[@]}; i++))
    do
        
        # data
        data_folder=$default_data_folder
        dataset=$default_dataset
        # training/evaluation
        eval_bs=$default_eval_bs

        # set default hp
        # peft config
        # lora
        lora_r=$default_lora_r
        lora_alpha=$default_lora_alpha
        lora_modules=$default_lora_modules

        # adapter
        adapter_size=$default_adapter_size

        # reset hp in search seq
        if [ $hp_to_search == "lr" ]; then
            lr=${search_seq[i]}
        elif [ $hp_to_search == "lora_r" ]; then
            lora_r=${search_seq[i]}
            eval_bs=${eval_bss[i]}
        elif [ $hp_to_search == "adapter_size" ]; then
            adapter_size=${search_seq[i]}
            eval_bs=${eval_bss[i]}
        elif [ $hp_to_search == "data_size" ]; then
            data_folder=${search_seq[i]}
        else
            echo "Wrong input"
            exit 1
        fi



        # expr name
        model_name=${default_model//\//_} # flatten "/" 
        # dataset/dataset_config/model/tuning_mode/tuning_config/lr
        # e.g.
        # ni/default_train_707_val_50/t5-xl-lm-adapt/ft/no_config/lora_r_64_alpha_32/5e-4
        # cot/xxx/llama-7b/lora_peft/lora_r_64/5e-4
        # we can uniform run_name/logging_dir in bash script rather than in python script

        # after hp are determined, set tuning_config

        if [ $tuning_mode == "lora_peft" ]; then
            tuning_config="r_${lora_r}_alpha_${lora_alpha}_modules_${lora_modules}" # for lora_peft
            tuning_args="--tuning_mode ${tuning_mode} --lora_r ${lora_r} --lora_alpha ${lora_alpha} --learning_rate ${lr}"
        elif [ $tuning_mode == "adapter" ]; then
            tuning_config="sz_${adapter_size}" # for adapter_peft
            tuning_args="--tuning_mode ${tuning_mode} --adapter_size ${adapter_size} --learning_rate ${lr}"
        elif [ $tuning_mode == "fine_tuning" ]; then
            tuning_config="None"
            tuning_args="--tuning_mode ${tuning_mode} --learning_rate ${lr}"
        else
            tuning_config="None"
        fi

        expr_dir=${dataset}/${data_folder}/${model_name}/${tuning_mode}/${tuning_config}/lr_${lr}
        expr_name=${expr_dir//\//_} # replace "/" with "_"
        


        

        launch_prefix="hfai python hfai_accelerate.py  launch --config_file ${config_file}"
        launch_suffix="--is_cluster -- --nodes 1 --no_inherit --force --name $expr_name"

        if [ $script_mode == "dev" ]; then
            launch_prefix="accelerate launch --config_file configs/accelerate_A6000/default_config_ddp.yaml"
            launch_suffix="--dev_train"
        fi
        launch_command="${launch_prefix} prompt_tuning.py --model_name_or_path google/t5-xl-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size $eval_bs --eval_steps ${default_eval_step} --save_steps ${default_save_step}  ${tuning_args} --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${data_folder} --task_dir ../../data/tasks --predict_with_generate  --gradient_accumulation_steps 2 --do_train --logging_steps ${defualt_logging_steps} --run_name $expr_name --logging_dir $expr_dir $launch_suffix"

        if [ $script_mode  == "dev_cmd" ];then
            echo "---------------cmd $i-----------------"
            echo "expr_name: $expr_name"
            echo "expr_dir: $expr_dir"
            echo "launch command: $launch_command"
            echo -e "\n\n"
        elif [[ $script_mode == "cluster" || $script_mode == "dev" ]];then
            echo $launch_command
            eval $launch_command
        fi

        if [ $script_mode == "dev" ];then
            break
        fi
done
