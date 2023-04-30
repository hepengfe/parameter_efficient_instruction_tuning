# bash scripts/hfai/ft_data.sh ft_train_task
expr_name=$1


train_data_seq=("default_train_8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50")

for ((i=0; i<${#train_data_seq[@]}; i++))
    do
        # name is expr name + train data size
        python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 6000 --save_steps 3000  --tuning_mode fine_tuning  --learning_rate 1e-5 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/${train_data_seq[i]} --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500 --dev_run &
    done



# cd /weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit; source haienv peit3; python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 1 --per_device_eval_batch_size 100 --eval_steps 6000 --save_steps 3000  --tuning_mode fine_tuning  --learning_rate 1e-5 --num_train_epochs 4 --dataset_name ni --data_dir ../../data/splits/default_train_256_val_50 --task_dir ../../data/tasks --predict_with_generate  --bf16 True --max_num_instances_per_eval_task 100 --gradient_accumulation_steps 2 --do_train --is_cluster --logging_steps 500 --dev_run


# default_train_8_val_50
# default_train8_val_50