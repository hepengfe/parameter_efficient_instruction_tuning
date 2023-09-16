import os
from utils import flatten, get_latest_checkpoint
import json
import argparse
log_dir = "cache/tmp"

"logs/ni/default_train_707_val_50/google_t5-xl-lm-adapt/lora_adapter/r_8_alpha_8_modules_qv"
dataset="ni/default_train_707_val_50"
model="facebook/opt-350m"
model=flatten(model, "/-")
peft_methods=["lora_adapter"]

def extract_expr(log_dir, dataset, model, peft_method, rand_seeds, expr_type = None):
    """
    Extract peft method evaluation results with all random seeds of all peft parameters (rank or adapter size) into a dictionary.

    Example dictionary:
    It could be lora or adapter with rank/size 8 and 16.
    The three values in the lists are the rougeL scores of the three random seeds.
    d = {
        "8": [28, 29, None]
        "16": [30, 31, 32]
    }
    """
    model=flatten(model, "/-")

    
    search_dir = os.path.join(log_dir, dataset, model, peft_method)
    
    if not os.path.exists(search_dir):
        print("search dir not exist: ", search_dir)
        return None
    folders = os.listdir(search_dir) # it also include all random seeds
    
    
    train_state = None
    best_train_state = {}
    best_test_score = -1
    d = {}
    for rand_seed in rand_seeds:
        rand_seed_folders = [f for f in folders if str(rand_seed) in f]
        for f in rand_seed_folders:
            # if no random seed provided
            # if rand_seed is None or str(rand_seed) not in f:
            #     pass
            # else:
            #     # skip if random seed not match
            #     if str(rand_seed) != f.split("_")[f.split("_").index("seed")+1]:
            #         continue
            

            # extract potential keys
            # get the number after "r", "sz"
            if "lora" in peft_method:
                size_or_rank = f.split("_")[f.split("_").index("r")+1]
            elif "adapter" in peft_method:
                size_or_rank = f.split("_")[f.split("_").index("size")+1]
            elif "prompt_tuning" in peft_method:
                size_or_rank = f.split("_")[f.split("_").index("len")+1]
            elif "fine_tuning" in peft_method:
                size_or_rank = "None"
            else:
                raise ValueError(f"peft method {peft_method} not supported")
            f_path = os.path.join(search_dir, f)
            print("searched path: ", f_path)
            try:
                lastest_cp = get_latest_checkpoint(f_path)
                if lastest_cp is None:
                    raise ValueError(f"Latest checkpoint is None from {f_path}" )
            except Exception as e:
                print(e)
                continue
            # assert lastest_cp is not None, f"no checkpoint found in {f_path}"
            
            lr = f.split("_")[f.split("_").index("lr")+1]


            # determine key based on expr_type
            if expr_type in ["2", "3", "single", "4"]:
                key = size_or_rank
            elif expr_type == "1":
                key = lr
            else:
                raise NotImplementedError(f"expr_type {expr_type} not implemented")

            last_train_state = os.path.join(lastest_cp, "training_state.json")
            
            if os.path.exists(last_train_state):
                try:
                    train_state = json.load(open(last_train_state))
                except Exception as e:
                    print(f"error loading {last_train_state}")
                    continue
                # if train_state["state_dict"]["test_eval_finished"] == False:
                #     # print(f"test eval not finished for {train_state['training_args/run_name']}")
                #     print(f"test eval not finished for {f}")
                #     d[size_or_rank] = d.get(size_or_rank, []) + [None]
                #     continue
                if "test/rougeL" not in train_state["state_dict"]:
                    print(f"test eval not finished for {f}")
                    d[key] = d.get(key, []) + [None]
                    continue 
                if "ni" in dataset:
                    best_metric_val = train_state["state_dict"]["test/rougeL"]
                    d[key] = d.get(key, []) + [best_metric_val]
                    if args.show_all:
                        # print_state_info(train_state)
                        if args.peft_method == "lora_adapter" or args.peft_method == "lora_peft":
                            r_index = train_state["training_args"]['training_args/run_name'].split("_").index("r")
                            lora_r = train_state["training_args"]['training_args/run_name'].split("_")[r_index+1]
                            
                            print(f"test rougeL: {best_metric_val}, lora_r: {lora_r}")
                        elif args.peft_method == "adapter":
                            sz_index = train_state["training_args"]['training_args/run_name'].split("_").index("size")
                            adapter_size = train_state["training_args"]['training_args/run_name'].split("_")[sz_index+1]
                            print(f"test rougeL: {best_metric_val}, adapter_size: {adapter_size}")
                        else:
                            print_state_info(train_state)
                elif "alpaca" in dataset:
                    
                    # extract all k-v pair containing "mmlu" from train_state
                    for k, v in train_state["state_dict"].items():
                        print(k)
                        if "mmlu" in k:
                            d[k] = v
                    # train_state["mmlu/0-shot/subcat/xxx"]
                    # train_state["mmlu/5-shot/subcat/xxx"]

                    
                
                
                # if best_metric_val > best_test_score:
                #     best_train_state = train_state
                #     best_test_score = best_metric_val
        # also verify test_eval_finished

        # analyze best train state
        # print log dir (for easy search)
        # print best rougeL
        # print peft method and its peft config
        # tranable params
        # or just copy the best train state to a new folder?
        # print(f"------- best train state found for {model}, {peft_method} -------")
        # # print(f"best rougeL: {best_train_state['state_dict']['test/rougeL']}")
        # if best_train_state:
        #     print_state_info(best_train_state)
        # else:
        #     print("no best train state found")
        #     return None
    return d

def print_state_info(state_d):
    
    print("test rougel: ", state_d["state_dict"]["test/rougeL"])
    print("best eval rougel: ", state_d["state_dict"]["best_metric_val"])
    print("best_metric_step: ",  state_d["state_dict"]["best_metric_step"])
    print(
        f" trainable_params: {state_d['state_dict']['trainable_params']}"
        )
    print(f" total_model_params: {state_d['state_dict']['total_model_params']}")
    print(f" trainable ratio: {state_d['state_dict']['trainable_ratio']}")
    print(f" run name: {state_d['training_args']['training_args/run_name']}")
    print("--------------------\n")

# python expr_analysis/expr_extract.py --expr_type 1 --peft_method fine_tuning --model google/t5-xl-lm-adapt

# python expr_analysis/expr_extract.py --expr_type 2 --peft_method prompt_tuning --model google/t5-large-lm-adapt
# python expr_analysis/expr_extract.py --expr_type 2 --peft_method adapter_peft --model google/t5-xl-lm-adapt
# python expr_analysis/expr_extract.py --expr_type 2 --peft_method lora_peft --model google/t5-xl-lm-adapt
# python expr_analysis/expr_extract.py --expr_type 2 --peft_method adapter --model google/t5-xl-lm-adapt
# python expr_analysis/expr_extract.py --expr_type 2 --peft_method lora_adapter --model google/t5-xl-lm-adapt
# python expr_analysis/expr_extract.py --expr_type 3 --peft_method adapter_peft --model google/t5-xxl-lm-adapt
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_dir", type=str, default="cache/tmp")
    arg_parser.add_argument("--dataset", type=str, default="ni/default_train_707_val_50")
    arg_parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
    arg_parser.add_argument("--peft_method", type=str, default="lora_adapter")
    arg_parser.add_argument("--random_seeds", type=int, nargs="+", default=[42, 127, 894])
    arg_parser.add_argument("--expr_type", type=str, default="single")
    arg_parser.add_argument("--show_all", action="store_true")
    args = arg_parser.parse_args()
    assert args.expr_type in ["single", "1", "2", "3", "4"]
    search_dir = os.path.join(args.log_dir, args.dataset, args.model, args.peft_method)
    peft_methods = [args.peft_method]
    d = {}
    if args.expr_type == "single":
        assert len(args.random_seeds) == 1
        d = extract_expr(args.log_dir, args.dataset, args.model, args.peft_method, args.random_seeds, expr_type = "single")
    elif args.expr_type == "1":
        dataset = "ni/default_train_707_val_50"
        model = "google/t5-xl-lm-adapt"
        lrs = ("1e-5", "5e-5", "1e-4", "5e-4", "1e-3")

        d = extract_expr(args.log_dir, dataset, model, args.peft_method, args.random_seeds, expr_type = "1")
    elif args.expr_type == "2":
        # dictionary structure
        # dataset -> peft method -> random seed 
        print("args.dataset is not used")
        for dataset in ["ni/default_train_707_val_50", "ni/default_train_512_val_50", "ni/default_train_256_val_50", "ni/default_train_128_val_50", "ni/default_train_64_val_50", "ni/default_train_32_val_50", "ni/default_train_8_val_50"]:
            d[dataset] = extract_expr(args.log_dir, dataset, args.model, args.peft_method, args.random_seeds, expr_type = "2")
    elif args.expr_type == "3":
        for model in [
            "google/t5-base-lm-adapt","google/t5-large-lm-adapt","google/t5-xl-lm-adapt","google/t5-xxl-lm-adapt",
        ]:
            print('args.model is not used')
            d[model] = extract_expr(args.log_dir, args.dataset, model, args.peft_method, args.random_seeds, expr_type = "3")
            
    elif args.expr_type == "4":
        for model in [
            "google/t5-xxl-lm-adapt", "facebook/opt-13b", "facebook/llama-7b"
        ]:
            print('args.model is not used')
            d[model] = extract_expr(args.log_dir, args.dataset, model, args.peft_method, args.random_seeds, expr_type = "4")
            
    else:
        raise NotImplementedError(
            f"expr_type {args.expr_type} not implemented"
        )
    print(d)