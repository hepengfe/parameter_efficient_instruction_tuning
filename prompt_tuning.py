from transformers import AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, logging
from peft_trainer import PEFTTrainer
import datetime
from arguments import TrainerArguments
import argparse
import os
import torch
import transformers

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()


    arg_parser.add_argument("--model", type=str, default="google/t5-large-lm-adapt")
    arg_parser.add_argument("--model_arch", type=str, default="encoder-decoder")
    arg_parser.add_argument("--mode", type=str, default="prompt_tuning")
    arg_parser.add_argument("--dataset_name", type=str, default=None)
    arg_parser.add_argument("--dataset_config_name", type=str, default=None)
    arg_parser.add_argument("--dev", action="store_true")
    arg_parser.add_argument("--max_steps", type=int, default=None)
    arg_parser.add_argument("--eval_save_steps", type=int, default=2500)
    arg_parser.add_argument("--per_device_train_batch_size", type=int, default=12)
    arg_parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    arg_parser.add_argument("--lr", type=float, default=0.3)
    arg_parser.add_argument("--num_soft_tokens", type=int, default=10)
    arg_parser.add_argument("--early_stop", action="store_true")
    arg_parser.add_argument("--test_train", action="store_true")
    arg_parser.add_argument("--paper_mode", action="store_true")
    arg_parser.add_argument("--eval", action="store_true")
    # predict_with_generate
    arg_parser.add_argument("--predict_with_generate", action="store_true")
    # gradient_accumulation_steps
    arg_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # ni specific args
    arg_parser.add_argument("--data_dir", type=str, default=None)
    arg_parser.add_argument("--task_dir", type=str, default=None)
    arg_parser.add_argument("--max_num_instances_per_task", type=int, default=100)
    arg_parser.add_argument("--max_num_instances_per_eval_task", type=int, default=100)
    arg_parser.add_argument("--add_task_name", action="store_true")
    arg_parser.add_argument("--add_task_definition", action="store_true")
    arg_parser.add_argument("--num_pos_examples", type=int, default=0)
    arg_parser.add_argument("--num_neg_examples", type=int, default=0)
    arg_parser.add_argument("--add_explanation", action="store_true")
    arg_parser.add_argument("--tk_instruct", action="store_true")
    arg_parser.add_argument("--max_source_length", type=int, default=256)
    arg_parser.add_argument("--max_target_length", type=int, default=8)
    
    # peft args
    arg_parser.add_argument("--lora_r", type=int, default=None)
    arg_parser.add_argument("--prefix_len", type=int, default=None)
    
    arg_parser.add_argument("--overwrite_cache", action="store_true")
    arg_parser.add_argument("--trainable_params_percentage", type=float, default=None)
    arg_parser.add_argument("--reduction_factor", type=float, default=32)
    
    arg_parser.add_argument("--layer_name", type=str, default=None)
    arg_parser.add_argument("--bias_name", type=str, default=None)
    
    arg_parser.add_argument("--use_pelt_gate", action="store_true")
    # lora_modules
    arg_parser.add_argument("--lora_modules", type=str, default=None)
    # phm_dimension
    arg_parser.add_argument("--phm_dimension", type=int, default=2)
    
    arg_parser.add_argument("--fp16", action="store_true")
    arg_parser.add_argument("--bf16", action="store_true")
    # num_train_epochs
    arg_parser.add_argument("--num_train_epochs", type=float, default=2.0)
    # num_training_tasks
    # arg_parser.add_argument("--num_training_tasks", type=int, default=None)
    args = arg_parser.parse_args()
    # args.models = args.models.split(",")
    # args.models = [m.strip() for m in args.models]
    assert args.dataset_name is not None, "dataset name is required"
    if args.dataset_name == "ni":
        assert args.predict_with_generate, "predict_with_generate is required for ni"
    
    if args.trainable_params_percentage is None:
        if args.mode == "lora":
            assert args.lora_r is not None, "lora_r is required for lora if trainable_params_percentage is not specified"
        if args.mode == "prefix_tuning":
            assert args.prefix_len is not None, "prefix_len is required for prefix_tuning"

    
    if args.mode == "layer_tuning":
        assert args.layer_name is not None, "layer_name should be specified for layer tuning mode"
    
    if args.mode == "bitfit":
        if args.bias_name is None:
            args.bias_name = "encoder_decoder_bias"
            print("bias_name is set to encoder_decoder_bias since args.bias_name is not specified")
        
    if args.mode == "fine_tuning":
        args.lr = 1e-5
        print("lr is set to 1e-5 due to fine_tuning mode")
        
    
    num_training_tasks = None
    # extract suffix number from data_dir
    if args.data_dir is not None:
        import re
        result = re.findall(r'\d+', args.data_dir)
        if len(result) == 0:
            num_training_tasks = None
        else:
            num_training_tasks = int(result[-1])


    cache_path = "~/tmp/cache"
    EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
    # add random number into output_path
    import random
    output_path += "_" + str(random.randint(0, 10000))
    default_optimizer_n_scheduler = False
    # if args.dev:
    #     default_optimizer_n_scheduler = True
    
    if args.dataset_name == "ni":
        assert args.data_dir is not None, "data_dir is required for ni"
        assert args.task_dir is not None, "task_dir is required for ni"
        
        args.max_source_length = 1024
        args.max_target_length = 128
        print("max_source_length is set to 1024")
        print("max_target_length is set to 128")
    
    
    # run name
    # run_name = args.models[0] + "-" + args.dataset_name 
    # if args.mode == "lora" and args.trainable_params_percentage is None:
    #     run_name += "-lora_r-" + str(args.lora_r)
    # elif args.mode == "prefix_tuning":
    #     run_name += "-prefix_len-" + str(args.prefix_len)
    
    # if args.fp16:
    #     run_name += "-fp16"
    # elif args.bf16:
    #     run_name += "-bf16"
    # run_name += args.mode + "_" + time # differentiate diff runs
    # rewrite the logic above into run name list
    run_name_list = []
    run_name_list.append(args.model)
    run_name_list.append(args.dataset_name)
    if args.trainable_params_percentage is None: # if use preset config
        if args.mode == "lora":
            run_name_list += ["lora_r", str(args.lora_r)]
        if args.mode == "prefix_tuning":
            run_name_list += ["prefix_len", str(args.prefix_len)]
    if args.fp16:
        run_name_list.append("fp16")
    elif args.bf16:
        run_name_list.append("bf16")
    run_name_list.append(args.mode)
    run_name_list.append("lr_" + str(args.lr))
    run_name = "-".join(run_name_list)
    # either max_steps or num_train_epochs should be specified
    assert args.max_steps is not None or args.num_train_epochs is not None, "either max_steps or num_train_epochs should be specified"
    
    
    
    trainer_args = TrainerArguments(
        model_names_or_path = args.model,
        model_arch = args.model_arch,
        dataset_name = args.dataset_name,
        dataset_config_name =args.dataset_config_name,
        evaluation_strategy="steps",
        eval_steps =  args.eval_save_steps,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=1,
        save_steps = args.eval_save_steps,
        max_steps=-1 if args.max_steps is None else args.max_steps, # it will override num_train_epochs if specified
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        output_dir = output_path,
        pad_to_max_length = True,
        predict_with_generate = args.predict_with_generate,
        model_parallel_gpus = 6,
        eval_accumulation_steps = 1,
        gradient_accumulation_steps = 1,
        cache_dir = cache_path,
        mode = args.mode,
        num_soft_tokens = args.num_soft_tokens,
        dev = args.dev,
        learning_rate = args.lr,
        default_optimizer_n_scheduler = default_optimizer_n_scheduler,
        report_to = "wandb",
        # label_names = ["label", "labels"],
        label_names = ["labels"],
        data_dir = args.data_dir,
        task_dir = args.task_dir,
        max_source_length = args.max_source_length,
        max_target_length = args.max_target_length,
        max_num_instances_per_eval_task= args.max_num_instances_per_eval_task,
        load_best_model_at_end=True,
        save_strategy = "steps",
        fp16 = args.fp16,
        bf16 = args.bf16,
        lora_r = args.lora_r,
        prefix_len = args.prefix_len,
        overwrite_cache= args.overwrite_cache,
        run_name = run_name,
        trainable_params_percentage = args.trainable_params_percentage,
        reduction_factor = args.reduction_factor,
        layer_name = args.layer_name,
        bias_name = args.bias_name,
        num_training_tasks= num_training_tasks,
        use_pelt_gate=args.use_pelt_gate,
        max_num_instances_per_task=args.max_num_instances_per_task,
        lora_modules=args.lora_modules,
        phm_dimension=args.phm_dimension,
    )
    trainer = PEFTTrainer(trainer_args)
    
    transformers.logging.set_verbosity_warning()

    trainer.train()
