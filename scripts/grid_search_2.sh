# search correlation between PEFT hyperparameters and the number of tasks
if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
    hfai workspace push  --force --no_zip
fi
script_mode=$1
random_seeds=(42 127 894)
for random_seed in "${random_seeds[@]}"; do
    for file in $(ls -d scripts/grid_search_2/*); do
        bash $file $script_mode t5 $random_seed &
    done
    if [ $script_mode == "dev" ];then
        break
    fi
done
