# search best learning rate for fine tuning
if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
    hfai workspace push  --force --no_zip
fi

random_seeds=(42)
for random_seed in "${random_seeds[@]}"; do
    export RANDOM_SEED=$random_seed
    bash scripts/grid_search_1/grid_search_lora_1.sh $1 t5
    if [ $script_mode == "dev" ];then
        break
    fi
done