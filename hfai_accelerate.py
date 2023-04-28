# this file is adapted for hfai run
# do not use this file to run training tasks locally
import haienv
# haienv.set_env('202111') # base env?
# haienv.set_env('peit4')
haienv.set_env('peit3')
from argparse import ArgumentParser
import os
import sys
import subprocess
from accelerate.commands.config import get_config_parser
from accelerate.commands.env import env_command_parser
from accelerate.commands.launch import launch_command_parser
from accelerate.commands.test import test_command_parser
from accelerate.commands.tpu import tpu_command_parser


def main():
    parser = ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]", allow_abbrev=False)
    subparsers = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    get_config_parser(subparsers=subparsers)
    env_command_parser(subparsers=subparsers)
    launch_command_parser(subparsers=subparsers)
    tpu_command_parser(subparsers=subparsers)
    test_command_parser(subparsers=subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)
    print("sys path: ", sys.path)
    # pre-check deepspeed packages
    # subprocess.check_output('ninja --version'.split())

    hfai_proj_dir = "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit/"

    if args.config_file is not None:
        args.config_file = os.path.join(hfai_proj_dir, args.config_file)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()

# submitted
# ['.', '/weka-jd/prod/marsV2/hf_venvs/hf_env', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8/site-packages', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8/lib-dynload', '/weka-jd/prod/marsV2/hf_venvs/python3.8/202207/lib/python3.8/site-packages', '/weka-jd/prod/marsV2/hf_venvs/haienv']

# /weka-jd/prod/marsV2/hf_venvs/python3.8/202207/bin/ninja


# peit4 container
# ['', '/ceph-jd/pub/jupyter/wangyizhong/home', '/weka-jd/prod/marsV2/hf_venvs/hf_env', '/opt/alpha-lib-linux/build/pyarmor_3', '/opt/alpha-lib-linux/build/out/lib', '/opt/alpha-lib-linux/home', '/opt/alpha-lib-linux/home/share', '/opt/alpha-lib-linux/home/share/store_tools', '/opt/apex', '/opt/alpha_packs', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python38.zip', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8/lib-dynload', '/hf_shared/hfai_envs/wangyizhong/peit4_0/lib/python3.8/site-packages', '/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit/adapter-transformers/src', '/weka-jd/prod/platform_team/system_env/py38-202207_0/lib/python3.8/site-packages', '/weka-jd/prod/marsV2/hf_venvs/haienv']







# peit3 container
# ['', '/ceph-jd/pub/jupyter/wangyizhong/home', '/weka-jd/prod/marsV2/hf_venvs/hf_env', '/opt/alpha-lib-linux/build/pyarmor_3', '/opt/alpha-lib-linux/build/out/lib', '/opt/alpha-lib-linux/home', '/opt/alpha-lib-linux/home/share', '/opt/alpha-lib-linux/home/share/store_tools', '/opt/apex', '/opt/alpha_packs', '/hf_shared/hfai_envs/wangyizhong/peit3_0/lib/python38.zip', '/hf_shared/hfai_envs/wangyizhong/peit3_0/lib/python3.8', '/hf_shared/hfai_envs/wangyizhong/peit3_0/lib/python3.8/lib-dynload', '/hf_shared/hfai_envs/wangyizhong/peit3_0/lib/python3.8/site-packages']