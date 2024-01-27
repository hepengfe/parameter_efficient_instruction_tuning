# since the push could fail so this command keep push n times

n=$1 # number of times to push file changes

for ((i=1; i<=n; i++))
do
    hfai workspace push --force --no_zip  # zip files that needs to be uploaded
done

