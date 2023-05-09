# iterate all files under gird_search_1
for file in $(ls -d scripts/grid_search_1/*); do
    bash $file hfai t5 &
done