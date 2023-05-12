# iterate all files under gird_search_1
if [[ $1 == "hfai" || $1 == "hfai_rm" ]]; then
    hfai workspace push  --force --no_zip
fi

for file in $(ls -d scripts/grid_search_1/*); do
    bash $file $1 t5 &
done