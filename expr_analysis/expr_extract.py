import os
from utils import flatten, get_latest_checkpoint
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
log_dir = "cache/tmp"

"logs/ni/default_train_707_val_50/google_t5-xl-lm-adapt/lora_adapter/r_8_alpha_8_modules_qv"
dataset="ni/default_train_707_val_50"
model="facebook/opt-350m"
model=flatten(model, "/-")
peft_methods=["lora_adapter"]



category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}
eval_category_keys = ["test/" + f"{metric}_for_{category}" + "_test" for category, metric in list(category_metrics.items())]

def remove_extra_brace(file_path):
    """
    Some json file could have extra brace at the end possibly due to concurrent writing.
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    if content.endswith("}}"):
        print("removing extra brace in ", file_path)
        content = content[:-1]
    
    with open(file_path, "w") as f:
        f.write(content)

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

    # rename all files to have random seed suffix
    # for f in folders:
    #     if "seed" in f:
    #         continue
    #     os.rename(os.path.join(search_dir, f), os.path.join(search_dir, f"{f}_random_seed_{42}"))

    train_state = None
    best_train_state = {}
    best_test_score = -1
    d = {}
    # if expr_type == "1":
    #     # no random seed in expr type 1
    #     rand_seeds = [""]
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
                if expr_type == "3" and size_or_rank != "512":
                    continue
                # if expr_type == "4":
                #     continue
            elif args.peft_method == "adapter" or args.peft_method == "adapter_peft":
                size_or_rank = f.split("_")[f.split("_").index("size")+1]
                if expr_type == "3" and size_or_rank != "256":
                    continue
            elif "prompt_tuning" in peft_method:
                size_or_rank = f.split("_")[f.split("_").index("len")+1]
            elif "fine_tuning" in peft_method:
                size_or_rank = "None"
            elif "prefix_tuning" in peft_method:
                prefix_len = f.split("_")[f.split("_").index("len")+1]
                bottleneck_size = f.split("_")[f.split("_").index("size")+1]
                size_or_rank = f"prefix_len_{prefix_len}_bottleneck_size_{bottleneck_size}"
            elif "ia3" in peft_method or "bitfit" in peft_method:
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
            if expr_type in ["1", "2", "3", "single", "4", "5"]:
                key = size_or_rank
            # elif expr_type == "1":
            #     key = lr
            else:
                raise NotImplementedError(f"expr_type {expr_type} not implemented")
            d[key] = d.get(key, {})
            last_train_state = os.path.join(lastest_cp, "training_state.json")
            
            if os.path.exists(last_train_state):
                remove_extra_brace(last_train_state)
                try:
                    train_state = json.load(open(last_train_state))
                except Exception as e:
                    print(f"error loading {last_train_state}")
                    # continue
                    raise e

                if "test/rougeL" in train_state["state_dict"] and train_state["state_dict"]["test/rougeL"] is not None and train_state["state_dict"]["test_eval_finished"] == False:
                    train_state["state_dict"]["test_eval_finished"] = True
                    print(f"correcting test_eval_finished to True for {last_train_state}")
                    with open(last_train_state, "w") as f:
                        json.dump(train_state, f)

                    
                # if train_state["state_dict"]["test_eval_finished"] == False:
                #     # print(f"test eval not finished for {train_state['training_args/run_name']}")
                #     print(f"test eval not finished for {f}")
                #     d[size_or_rank] = d.get(size_or_rank, []) + [None]
                #     continue
                if "ni" in dataset:
                    if expr_type == "1":
                        metric_key = "test/rougeL"
                        if metric_key not in train_state["state_dict"]:
                            print(f"test eval not finished for {last_train_state}")
                            print(f"its run name is {train_state['training_args']['training_args/run_name']}")
                            d[key][lr] = d[key].get(lr, []) + [None]
                            continue 
                        best_metric_val = train_state["state_dict"][metric_key]
                        d[key][lr] = d[key].get(lr, []) + [best_metric_val]
                    else:
                        # filter non-optimal lr which are not needed for expr type other than 1
                        lr = train_state["training_args"]['training_args/run_name'].split("_")[train_state["training_args"]['training_args/run_name'].split("_").index("lr")+1]
                        if args.peft_method == "lora_adapter" or args.peft_method == "lora_peft":
                            if lr != args.lr:
                                # d[key][metric_key] = d[key].get(metric_key, []) + [None]
                                print(f"lr {lr} not optimal for {args.peft_method}, skip results...")
                                continue
                        if args.peft_method == "adapter" or args.peft_method == "adapter_peft":
                            if lr != args.lr:
                                print(f"lr {lr} not optimal for {args.peft_method}, skip results...")
                                continue

                        for metric_key in ["test/rougeL", "task", "category"]:
                            if metric_key == "test/rougeL":
                                if metric_key not in train_state["state_dict"]:
                                    print(f"test eval not finished for {last_train_state}")
                                    print(f"its run name is {train_state['training_args']['training_args/run_name']}")
                                    d[key][metric_key] = d[key].get(metric_key, []) + [None]
                                    continue
      
                                
                                best_metric_val = train_state["state_dict"][metric_key]
                                d[key][metric_key] = d[key].get(metric_key, []) + [best_metric_val]
                                print(f"parameter count: {train_state['state_dict']['trainable_params']} trainable ratio {train_state['state_dict']['trainable_ratio']} and total model params {train_state['state_dict']['total_model_params']}")
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
                            elif metric_key == "task":
                                for k in train_state["state_dict"]:
                                    if "task" in k and k.startswith("test/") and k not in eval_category_keys:
                                        test_task_k = k
                                        val = train_state["state_dict"][test_task_k]
                                        d[key][test_task_k] = d[key].get(test_task_k, []) + [val]
                            elif metric_key == "category":
                                
                                for k in train_state["state_dict"]:
                                    if k in eval_category_keys:
                                        test_category_k = k
                                        category_metric_val = train_state["state_dict"][test_category_k]
                                        d[key][test_category_k] = d[key].get(test_category_k, []) + [category_metric_val]

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

def convert(s):
    d = {
        "0.001": "1e-3",
        "0.0005": "5e-4",
        "0.0001": "1e-4",
        "0.00001": "5e-5",
        "0.00005": "1e-5",
        "ni/default_train_707_val_50":707, # "$n=707$",
        "ni/default_train_512_val_50":512, # "$n=512$",
        "ni/default_train_256_val_50":256, # "$n=256$",
        "ni/default_train_128_val_50":128, # "$n=128$",
        "ni/default_train_64_val_50":64, # "$n=64$",
        "ni/default_train_32_val_50":32, # "$n=32$",
        "ni/default_train_8_val_50": 8, # "$n=8$",
        "google/t5-base-lm-adapt": "T5-base",
        "google/t5-large-lm-adapt": "T5-large",
        "google/t5-xl-lm-adapt": "T5-xl",
        "google/t5-xxl-lm-adapt": "T5-xxl",
    }
    if s in d:
        return d[s]
    else:
        print(f"no match for {s} whiling converting")
        return s



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
    arg_parser.add_argument("--plot_interest", type=str, default=None)
    arg_parser.add_argument("--lr", type=str, default="1e-4")
    arg_parser.add_argument("--show_all", action="store_true")
    args = arg_parser.parse_args()
    assert args.expr_type in ["single", "1", "2", "3", "4", "5"]
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
    elif args.expr_type in ["3", "5"]:
        for model in [
            "google/t5-base-lm-adapt","google/t5-large-lm-adapt","google/t5-xl-lm-adapt"
        ]:
            print('args.model is not used')
            d[model] = extract_expr(args.log_dir, args.dataset, model, args.peft_method, args.random_seeds, expr_type = args.expr_type )
            
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
    # k:  model, v: peft method dict
    # k2:  peft method  v2: result dict
    # k3:  result name (such as RougeL)  v3: list of values
    # compute average for each value
    rows = []
    cat_rows = []
    task_rows = []
    test_rougeL_rows = []



    peft_setups = []
    if args.expr_type in ["2", "3", "4", "5"]:
        
        for model_k, v in d.items():
            
            for peft_k, v2 in v.items():
                for cat_task_metric, v3 in v2.items():
                    if all(item is None for item in v3):
                        continue
                    filtered_v3 = [item for item in v3 if item is not None]
                    avg = sum(filtered_v3) / len(filtered_v3)
                    assert len(filtered_v3) <= 3, f"more than 3 random seeds for {model_k}, {peft_k}, {cat_task_metric}"
                    rows.append({"model": model_k, "peft_k": peft_k, "cat_task_metric": cat_task_metric, "avg": avg})
                    if cat_task_metric in eval_category_keys:
                        cat_rows.append({"model": model_k, "peft_k": peft_k, "cat_task_metric": cat_task_metric, "avg": avg})
                    if "task" in cat_task_metric:
                        task_rows.append({"model": model_k, "peft_k": peft_k, "cat_task_metric": cat_task_metric, "avg": avg})
                    if cat_task_metric == "test/rougeL":
                        test_rougeL_rows.append({"model": model_k, "peft_k": peft_k, "cat_task_metric": cat_task_metric, "avg": avg, "metric_vals": filtered_v3})
                        
    elif args.expr_type == "1":
        for peft_k, v in d.items(): # since it's lr search, only one xl model
            for lr_k, v2 in v.items():
                if all(item is None for item in v2):
                    continue
                filtered_v2 = [item for item in v2 if item is not None]
                avg = sum(filtered_v2) / len(filtered_v2)
                test_rougeL_rows.append({"lr": lr_k, "peft_k": peft_k, "cat_task_metric": "test/rougeL", "avg": avg, "metric_vals": filtered_v2})
    else:
        raise NotImplementedError(f"expr_type {args.expr_type} not implemented")
    if args.expr_type == "1":
        assert len(test_rougeL_rows) > 0, "no test_rougeL_rows extracted"
    else:
        assert len(rows) > 0, "no rows extracted"
        assert len(task_rows) > 0, "no task_rows extracted"
        assert len(cat_rows) > 0, "no cat_rows extracted"
        assert len(test_rougeL_rows) > 0, "no test_rougeL_rows extracted"


    # df = pd.DataFrame(rows)
    # df = df.pivot_table(index='model', columns=['peft_k', 'cat_task_metric'], values='avg')
    out_dir=f"results/expr_type_{args.expr_type}/{args.peft_method}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # file_name = os.path.join(out_dir, f"{args.peft_method}.csv")
    # print(df)
    # print(f"writing to {file_name}")
    # df.to_csv(file_name)

    if args.expr_type == "1":
        test_rougeL_df = pd.DataFrame(test_rougeL_rows)
        pivoted_test_rougeL_df = test_rougeL_df.pivot_table(index='peft_k', columns=['lr',], values='avg')
        test_rougeL_file_name = os.path.join(out_dir, f"{args.peft_method}_{args.lr}_test_rougeL.csv")
        print(pivoted_test_rougeL_df)
        print(f"writing test_rougeL df to {test_rougeL_file_name}")
        pivoted_test_rougeL_df.to_csv(test_rougeL_file_name)
    elif args.expr_type in ["2"]:
        test_rougeL_df = pd.DataFrame(test_rougeL_rows)
        pivoted_test_rougeL_df = test_rougeL_df.pivot_table(index='model', columns=['peft_k', 'cat_task_metric'], values='avg')
        test_rougeL_file_name = os.path.join(out_dir, f"{args.peft_method}_{args.lr}_test_rougeL.csv")
        print(pivoted_test_rougeL_df)
        print(f"writing test_rougeL df to {test_rougeL_file_name}")
        pivoted_test_rougeL_df.to_csv(test_rougeL_file_name)

    elif args.expr_type in ["3", "5"]: # model centric
        # convert to data frame and save as csv
        task_df = pd.DataFrame(task_rows)
        task_df = task_df.pivot_table(index='model', columns=['peft_k', 'cat_task_metric'], values='avg')
        task_file_name = os.path.join(out_dir, f"{args.peft_method}_task_{args.lr}.csv")
        print(task_df)
        print(f"writing task df to {task_file_name}")
        task_df.to_csv(task_file_name)

        cat_df = pd.DataFrame(cat_rows)
        cat_df = cat_df.pivot_table(index='model', columns=['peft_k', 'cat_task_metric'], values='avg')
        cat_file_name = os.path.join(out_dir, f"{args.peft_method}_cat_{args.lr}.csv")
        print(cat_df)
        print(f"writing cat df to {cat_file_name}")
        cat_df.to_csv(cat_file_name)

        test_rougeL_df = pd.DataFrame(test_rougeL_rows)
        pivoted_test_rougeL_df = test_rougeL_df.pivot_table(index='model', columns=['peft_k', 'cat_task_metric'], values='avg')
        test_rougeL_file_name = os.path.join(out_dir, f"{args.peft_method}_{args.lr}_test_rougeL.csv")
        print(pivoted_test_rougeL_df)
        print(f"writing test_rougeL df to {test_rougeL_file_name}")
        pivoted_test_rougeL_df.to_csv(test_rougeL_file_name)

    if args.plot_interest is None:
        exit()
    # plt.rcParams['font.family'] = 'Times'
    plt.rcParams['font.family'] = "Times New Roman"
    if args.expr_type == "1":
        if args.plot_interest == "peft_k":
            # found the rank, lr and training stability
            # x is rank and legends are lr
            assert args.plot_interest is None or args.plot_interest == "peft_k", "only peft_k plot supported for expr_type 1"
            
            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            peft_setups = test_rougeL_df['peft_k'].unique()
            peft_setups = sorted(peft_setups, key=lambda x: int(x) if x != "None" else 0)

            num_lr = len(test_rougeL_df['lr'].unique())
            colors = np.random.rand(num_lr, 3)
            color_map = {lr: color for lr, color in zip(test_rougeL_df['lr'].unique(), colors)}
            x_pos = np.zeros(3)
            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                # draw by iterating rank
                subset = test_rougeL_df[test_rougeL_df['peft_k'] == setup]
                
                for index, row in subset.iterrows():
                    # print(row)
                    # if row['model'] != "ni/default_train_707_val_50":
                    #     print("skip non full dataset due to peft_k plot")
                    #     continue
                    print(row)
                    color = color_map[row['lr']]
                    # for each lr and each rank, draw one data point
                    num_data = 1
                    ys = row['metric_vals'][:num_data]
                    
                    # size = len(row['metric_vals'])
                    plt.scatter((x_pos)[:num_data], ys, color=color, label=f"{convert(row['lr'])}", s=18)
                x_pos += 1
            plt.xticks(list(position_map.values()), peft_setups)
            if "lora" in args.peft_method.lower():
                plt.xlabel('LoRA Rank')
            elif args.peft_method == "adapter" or args.peft_method == "adapter_peft":
                plt.xlabel('Adapter Size')
            else:
                raise NotImplementedError(f"peft method {args.peft_method} not supported for expr_type 1")
            plt.ylabel('RougeL Score')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            # sort legend by lr
            by_label = {k: by_label[k] for k in sorted(by_label.keys(), key=lambda x: float(x))}
            

            plt.legend(by_label.values(), by_label.keys())


    elif args.expr_type in ["2", "3", "5"]: # only 2 and 3 have plots
        if args.plot_interest == "peft_k":
            x_pos = np.zeros(3)
            jitter_values = [-0.1, 0, 0.1]
            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            peft_setups = test_rougeL_df['peft_k'].unique()
            peft_setups = sorted(peft_setups, key=lambda x: int(x) if x != "None" else 0)
            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                subset = test_rougeL_df[test_rougeL_df['peft_k'] == setup]
                
                for index, row in subset.iterrows():
                    if row['model'] != "ni/default_train_707_val_50":
                        print("skip non full dataset due to peft_k plot")
                        continue
                    print(row)
                    color = np.random.rand(3,)
                    size = len(row['metric_vals'])
                    plt.scatter((x_pos + jitter_values)[:size], row['metric_vals'], color=color, label=f"{convert(row['model'])} {convert(row['peft_k'])}")
                x_pos += 1
            plt.xticks(list(position_map.values()), peft_setups)
            if "lora" in args.peft_method.lower():
                plt.xlabel('LoRA Rank')
            elif args.peft_method == "adapter" or args.peft_method == "adapter_peft":
                plt.xlabel('Adapter Size')
            plt.ylabel('RougeL Score')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        elif args.plot_interest == "data_size":
            x_pos = np.zeros(3)

            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            # it's data size actually
            peft_setups = test_rougeL_df['model'].unique()
            peft_setups = sorted(peft_setups, key=lambda x: int(x.split('_')[2]) if x != "None" else 0)
            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                subset = test_rougeL_df[test_rougeL_df['model'] == setup]
                
                for index, row in subset.iterrows():
                    if (args.peft_method == "adapter" or args.peft_method == "adapter_peft") and row['peft_k'] != "256":
                        continue
                    if "lora" in args.peft_method and row['peft_k'] != "512":
                        continue
                    print(row)
                    color = np.random.rand(3,)
                    size = len(row['metric_vals'])
                    plt.scatter( x_pos[:size], row['metric_vals'], color=color, label=f"{convert(row['model'])} {convert(row['peft_k'])}")
                x_pos += 1
            plt.xticks(list(position_map.values()), [convert(item) for item in list(peft_setups)])
            plt.xlabel('Data Size')
            plt.ylabel('RougeL Score')


            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys())

        elif args.plot_interest == "data_size_n_peft_k":
            assert args.expr_type  == "2", "only expr_type 3 has model size and peft k plot"
            pivoted_test_rougeL_df = test_rougeL_df.pivot_table(index='peft_k', columns=['model', 'cat_task_metric'], values='avg')
            x_pos = 0

            num_data_size = len(test_rougeL_df['model'].unique())# it's actually data size
            colors = np.random.rand(num_data_size, 3)
            color_map = {data_size: color for data_size, color in zip(test_rougeL_df['model'].unique(), colors)}
            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            peft_setups = test_rougeL_df['peft_k'].unique()
            peft_setups = sorted(peft_setups, key=lambda x: int(x) if x != "None" else 0)

            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                subset = test_rougeL_df[test_rougeL_df['peft_k'] == setup]
                for index, row in subset.iterrows():
                    color = color_map[row['model']]
                    
                    plt.scatter(x_pos, row['avg'], color=color, label=f"$n=${convert(row['model'])}")
                x_pos += 1
            plt.xticks(list(position_map.values()), peft_setups)
            plt.xlabel('LoRA rank')
            plt.ylabel('RougeL Score')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        elif args.plot_interest == "model_size":
            assert args.expr_type  == "3", "only expr_type 3 has model size plot"
            x_pos = np.zeros(3)

            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            peft_setups = test_rougeL_df['model'].unique()

            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                subset = test_rougeL_df[test_rougeL_df['model'] == setup]
                
                
                for index, row in subset.iterrows():
                    color = np.random.rand(3,)
                    size = len(row['metric_vals'])
                    plt.scatter((x_pos)[:size], row['metric_vals'], color=color, label=f"{convert(row['model'])}  $r={convert(row['peft_k'])}$")
                x_pos += 1
            plt.xticks(list(position_map.values()), [convert(item) for item in list(peft_setups)])
            plt.xlabel('Model Size')
            plt.ylabel('RougeL Score')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        elif args.plot_interest ==  "model_size_n_peft_k":
            assert args.expr_type  == "5", "only expr_type 3 has model size and peft k plot"
            x_pos = 0
            test_rougeL_df = pd.DataFrame(test_rougeL_rows) # reset dataframe
            peft_setups = test_rougeL_df['model'].unique()

            position_map = {setup: i for i, setup in enumerate(peft_setups)}
            for setup in peft_setups:
                subset = test_rougeL_df[test_rougeL_df['model'] == setup]
                for index, row in subset.iterrows():
                    color = np.random.rand(3,)
                    plt.scatter(x_pos, row['avg'], color=color, label=f"{row['model']} {row['peft_k']}")
                x_pos += 1
            plt.xticks(list(position_map.values()), peft_setups)
            plt.xlabel('Model Size')
            plt.ylabel('RougeL Score')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
    plot_path = os.path.join(out_dir, f"{args.peft_method}_test_rougeL_{args.plot_interest}_{args.lr}.pdf")
    plt.savefig(plot_path)
    print(f"saved plot to {plot_path}")

