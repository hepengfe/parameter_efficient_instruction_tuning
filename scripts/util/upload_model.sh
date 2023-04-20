# given that ~/cache is ignored
# it's better to compress files locally ahead as it's not 
# tar -czvf hf_model2.tar.gz $1

# upload the tar file, keep push by 10 times
sh scripts/util/keep_push_workspace.sh 10


# on the cloud 
# tar -xzvf xxx.tar.gz


# TODO: only compress a target model and decompress it at the target folder
