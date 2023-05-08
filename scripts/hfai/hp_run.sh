# 1. this script requires pass hp by environment variables or relying on a script that set the hp
# 2. hp will have default values in this script 
# 3. 

# bash scripts/hfai/hp_search.sh <tuning_method>  <script_mode>


tuning_mode=$1
script_mode=$2


default_data_folder="default_train_707_val_50"
default_model="google/t5-xl-lm-adapt"
default_dataset="ni"

# lora
default_lora_modules="qv"



default_scheduler="linear"

# optimizer and scheduler
default_warmup_ratio=0.03

# two types of training mode
# deepspeed lower save/eval interval


declare -A model_name2path
declare -A lora_rank2bs
declare -A adapter_size2bs
adapter_size2bs=(["8"]=20 ["32"]=20 ["64"]=15 ["128"]=10 ["256"]=2)
model_name2path=(["t5"]="google/t5-xl-lm-adapt" ["opt"]="facebook/opt-13b" ["llama"]="facebook/llama-7b" ["gpt2"]="gpt2")
model_name2arch=(["t5"]="encoder-decoder" ["opt"]="decoder" ["llama"]="decoder")
lora_rank2bs=(["8"]=15 ["32"]=15 ["64"]=15 ["128"]=15 ["256"]=10 ["512"]=5)


model=${model_name2path[$MODEL_NAME]}
model_arch=${model_name2arch[$MODEL_NAME]}
scheduler=$default_scheduler

# tuning mode fixed setup
if [ $tuning_mode == "fine_tuning" ]; then
    config_file="configs/hfai/default_config_deepspeed_hf.yaml"
    default_eval_step=5000
    scheduler="constant"
elif [[ $tuning_mode == "lora_peft" || $tuning_mode == "lora_adapter" ]]; then
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    scheduler="linear"
    eval_bs=${lora_rank2bs[$LORA_RANK]}
elif [ $tuning_mode == "adapter" ]; then
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    eval_bs=${adapter_size2bs[$ADAPATER_SIZE]}
    scheduler="linear"
elif [ $tuning_mode == "prefix_tuning" ]; then
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    eval_bs=10
    scheduler="linear"
elif [ $tuning_mode == "prompt_tuning" ]; then
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    eval_bs=20
    scheduler="linear"
elif [ $tuning_mode == "ia3" ]; then
    config_file="configs/hfai/default_config_ddp.yaml"
    default_eval_step=5000
    eval_bs=20
    scheduler="linear"
else
    echo "tuning_mode ${tuning_mode} is not supported"
    exit 1
fi






# ddp higher save/eval interval
default_save_step=$((default_eval_step/5)) # 5000/5=1000
defualt_logging_steps=$((default_eval_step/20)) # 5000/20=250

if [ $script_mode == "hfai" ]; then
    hfai workspace push  --force --no_zip
fi

 
# data
data_folder=$default_data_folder
dataset=$default_dataset
# training/evaluation



# set default hp
# peft config
# lora

lora_modules=$default_lora_modules





# expr name
model_name=${default_model//\//_} # flatten "/" 
# dataset/dataset_config/model/tuning_mode/tuning_config/lr
# e.g.
# ni/default_train_707_val_50/t5-xl-lm-adapt/ft/no_config/lora_r_64_alpha_32/5e-4
# cot/xxx/llama-7b/lora_peft/lora_r_64/5e-4
# we can uniform run_name/logging_dir in bash script rather than in python script

# after hp are determined, set tuning_config

if [[ $tuning_mode == "lora_peft" || $tuning_mode == "lora_adapter" ]]; then
    tuning_config="r_${LORA_RANK}_alpha_${LORA_RANK}_modules_${lora_modules}" # for lora_peft
    tuning_args="--tuning_mode ${tuning_mode} --lora_r ${LORA_RANK} --lora_alpha ${LORA_RANK}"
elif [ $tuning_mode == "adapter" ]; then
    tuning_config="sz_${ADAPATER_SIZE}" # for adapter_peft
    tuning_args="--tuning_mode ${tuning_mode} --adapter_size ${ADAPATER_SIZE} "
elif [ $tuning_mode == "fine_tuning" ]; then
    tuning_config="None"
    tuning_args="--tuning_mode ${tuning_mode}"
elif [ $tuning_mode == "prefix_tuning" ]; then
    tuning_config="prefix_len_${PREFIX_LEN}_bottleneck_size_${BOTTLENECK_SIZE}"
    tuning_args="--tuning_mode ${tuning_mode} --prefix_len ${PREFIX_LEN} --bottleneck_size ${BOTTLENECK_SIZE} "
elif [ $tuning_mode == "prompt_tuning" ]; then
    tuning_config="prompt_len_${PROMPT_LEN}"
    tuning_args="--tuning_mode ${tuning_mode} --prompt_len ${PROMPT_LEN}"
elif [ $tuning_mode == "ia3" ]; then
    tuning_config="None"
    tuning_args="--tuning_mode ${tuning_mode}"
else
    echo "tuning_mode ${tuning_mode} is not supported"
    exit 1
fi

tuning_args+=" --learning_rate ${LR} --scheduler ${scheduler} --warmup_ratio ${default_warmup_ratio} --weight_decay ${WEIGHT_DECAY} --label_smoothing_factor ${LABEL_SMOOTHING_FACTOR} --dropout_rate ${DROPOUT_RATE}"


# expr_dir=${dataset}/${data_folder}/${model_name}/${tuning_mode}/${tuning_config}/lr_${lr}_label_smoothing_factor_${label_smoothing_factor}_scheduler_${scheduler}_warmup_steps_${warmup_steps}
expr_dir=${dataset}/${data_folder}/${model_name}/${tuning_mode}/${tuning_config}/lr_${LR}_weight_decay_${WEIGHT_DECAY}_dropout_rate_${DROPOUT_RATE}_label_smoothing_factor_${LABEL_SMOOTHING_FACTOR}_scheduler_${scheduler}_warmup_ratio_${default_warmup_ratio}

expr_name=${expr_dir//\//_} # replace "/" with "_"





launch_prefix="hfai python hfai_accelerate.py  launch --config_file ${config_file}"
launch_suffix="--is_cluster -- --nodes 1 --no_inherit --force --name $expr_name"

if [ $script_mode == "dev" ]; then
    launch_prefix="accelerate launch --config_file configs/accelerate_A6000/default_config_ddp.yaml"
    launch_suffix="--dev_train"
fi
launch_command="${launch_prefix} prompt_tuning.py --model_name_or_path ${model} --model_arch ${model_arch} --per_device_train_batch_size 1 --per_device_eval_batch_size $eval_bs --eval_steps ${default_eval_step} --save_steps ${default_save_step}  ${tuning_args} --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${data_folder} --task_dir ../../data/tasks --predict_with_generate  --gradient_accumulation_steps 2 --do_train --logging_steps ${defualt_logging_steps} --run_name $expr_name --logging_dir $expr_dir $launch_suffix"

if [ $script_mode  == "dev_cmd" ];then
    echo "---------------cmd $CMD_INDEX-----------------"
    echo "expr_name: $expr_name"
    echo "expr_dir: $expr_dir"
    echo "launch command: $launch_command"
    echo -e "\n\n"
elif [[ $script_mode == "hfai" || $script_mode == "dev" ]];then
    echo $launch_command
    eval $launch_command
fi
