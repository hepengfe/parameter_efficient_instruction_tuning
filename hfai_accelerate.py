# this file is adapted for hfai run
# do not use this file to run training tasks locally
import haienv
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
    
    # subprocess.run(['alias',
    #     'ninja=/opt/hf_venvs/python3.8/202111/bin/ninja'], shell=True)
    
    sys.path.append('/opt/hf_venvs/python3.8/202111/bin/ninja')
    sys.path.append('/opt/hf_venvs/python3.8/202111/bin')

    subprocess.check_output('/opt/hf_venvs/python3.8/202111/bin/ninja --version'.split())
    subprocess.check_output('ninja --version'.split())

    hfai_proj_dir = "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit/"

    if args.config_file is not None:
        args.config_file = os.path.join(hfai_proj_dir, args.config_file)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()
