# iterate all files under gird_search_1
hfai workspace push  --force --no_zip
for file in $(ls -d scripts/grid_search_1/*); do
    bash $file $1 t5 &
done