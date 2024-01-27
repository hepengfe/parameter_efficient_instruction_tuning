# since the push could fail so this command keep push n times
n=$1 # number of times to push file changes


# # assume current env is peit3
# bash scripts/util/non_editable_install.sh
for ((i=1; i<=$n; i++))
do
    hfai venv push peit3  # zip files that needs to be uploaded
done
