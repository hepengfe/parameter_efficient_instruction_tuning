
max_steps=30
eval_steps=5

# train one steps and eval, save the checkpoint.  max steps 2

# compute the approximate time to stop it 

CUDA_VISIBLE_DEVICES=0  WANDB_MODE=disabled python prompt_tuning.py --model_name_or_path google/t5-small-lm-adapt --model_arch encoder-decoder --per_device_train_batch_size 2 --per_device_eval_batch_size $per_device_eval_batch_size --eval_steps 2 --save_steps 2 --tuning_mode lora --learning_rate 3e-4 --max_steps 3 --dataset_name ni --data_dir data/splits/default --task_dir data/tasks --predict_with_generate  --bf16  --dev --max_num_instances_per_eval_task 10 

# continue previous latest checkpoint, the new current step should be 2 




# current test evaluation issues, pred is more than than # references 
# # references is controlled by --max_num_instances_per_eval_task, we should 


# trainer option -- load best checkopoint   -- load latest checkpoint
# --load best  -> used when evaluation
# --load latest -> turn on by default during training for suspending recoverage
