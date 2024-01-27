dataset="ni"
model="google/t5-large-lm-adapt"
num_instances_per_task=100
data_dir="../../data/splits/default_train_707_val_50"
task_dir="../../data/tasks"
# create a folder for each dataset
out_dir="out/${dataset}/${model}/"
mkdir -p $out_dir
per_device_eval_batch_size=16
gradient_accumulation_steps=8
max_num_instances_per_eval_task=50
eval_save_steps=2500
num_train_epochs=3
# start_cuda_idx=$2
# end_cuda_idx=$3
# cuda_device_seq=($(seq $start_cuda_idx $end_cuda_idx))

# bash run_t5_large_adapt_200_instances_by_lora_rank.sh 200 0 2
# bash run_t5_large_adapt_200_instances_by_lora_rank.sh 300 3 5
# create a seq with 8/32/64/128/256/512
lora_r_seq=(26 77 255 381 508)
percents=(0.01 0.03 0.1 0.15 0.20)
cuda_device_seq=(0 1 2 3 4)




# iterate
# for ((i=0; i<${#cuda_device_seq[@]}; i++))
#     do
#         device="${cuda_device_seq[i]}"
#         lora_r="${lora_r_seq[i]}"
#         percent="${percents[i]}"
#         sleep 1
#         # print command
#         CUDA_VISIBLE_DEVICES=$device python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps  --tuning_mode lora --learning_rate 5e-4 --dataset_name ni --data_dir $data_dir --task_dir $task_dir --predict_with_generate  --bf16 True --lora_r $lora_r --max_num_instances_per_task $num_instances_per_task --gradient_accumulation_steps $gradient_accumulation_steps --max_num_instances_per_eval_task $max_num_instances_per_eval_task --num_train_epochs $num_train_epochs>   "${out_dir}/lora_${percent}_lora_r_${lora_r}_num_instances_per_task_${num_instances_per_task}_valid_50_$(date +%Y%m%d_%H%M%S).log" &
# done
per_device_eval_batch_size=32
CUDA_VISIBLE_DEVICES=5 python prompt_tuning.py --model_name_or_path $model --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps $eval_save_steps --save_steps $eval_save_steps  --tuning_mode fine_tuning --learning_rate 1e-5 --dataset_name ni --data_dir $data_dir --task_dir $task_dir --predict_with_generate  --bf16 True --max_num_instances_per_task 100 --gradient_accumulation_steps $gradient_accumulation_steps --max_num_instances_per_eval_task $max_num_instances_per_eval_task --num_train_epochs $num_train_epochs >  "${out_dir}/fine_tuning_num_instances_per_task_${num_instances_per_task}_valid_50_$(date +%Y%m%d_%H%M%S).log" &


# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled  python prompt_tuning.py --model_name_or_path google/t5-large-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --eval_steps 1 --save_steps 5000  --tuning_mode fine_tuning --learning_rate 1e-5 --dataset_name ni --data_dir "data/splits/default_train_707_val_50" --task_dir data/tasks --predict_with_generate --max_num_instances_per_task 100