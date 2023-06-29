
DATA_FOLDERS=("default_train8_val_50" "default_train_32_val_50" "default_train_64_val_50" "default_train_128_val_50" "default_train_256_val_50" "default_train_512_val_50" "default_train_707_val_50")
peft_methods=("lora_adapter" "adapter" "fine_tuning")
model_names=("google/t5-base-lm-adapt" "google/t5-large-lm-adapt" "google/t5-xl-lm-adapt" "google/t5-xxl-lm-adapt")


# iterate all model names and peft_methods 
for model_name in "${model_names[@]}"; do
    for peft_method in "${peft_methods[@]}"; do
        echo "model_name ${model_name} ${peft_method}"
        python expr_analysis/expr_extract.py --dataset ni/default_train_707_val_50 --peft_method $peft_method --model $model_name
    done
done
