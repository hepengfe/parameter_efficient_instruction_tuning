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

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()


    arg_parser.add_argument("--models", type=str, default="google/t5-large-lm-adapt")
    arg_parser.add_argument("--model_arch", type=str, default="encoder-decoder")
    arg_parser.add_argument("--mode", type=str, default="prompt_tuning")
    arg_parser.add_argument("--dataset_name", type=str, default=None)
    arg_parser.add_argument("--dataset_config_name", type=str, default=None)
    arg_parser.add_argument("--dev", action="store_true")
    arg_parser.add_argument("--max_steps", type=int, default=30000)
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
    arg_parser.add_argument("--num_pos_examples", type=int, default=2)
    arg_parser.add_argument("--num_neg_examples", type=int, default=0)
    arg_parser.add_argument("--add_explanation", action="store_true")
    arg_parser.add_argument("--tk_instruct", action="store_true")
    arg_parser.add_argument("--max_source_length", type=int, default=256)
    arg_parser.add_argument("--max_target_length", type=int, default=8)
    
    # peft args
    arg_parser.add_argument("--lora_r", type=int, default=None)
    arg_parser.add_argument("--prefix_len", type=int, default=None)
    
    arg_parser.add_argument("--overwrite_cache", action="store_true")
    
    
    arg_parser.add_argument("--layer_name", type=str, default=None)
    
    
    arg_parser.add_argument("--fp16", action="store_true")
    arg_parser.add_argument("--bf16", action="store_true")
    
    args = arg_parser.parse_args()
    args.models = args.models.split(",")
    args.models = [m.strip() for m in args.models]
    assert args.dataset_name is not None, "dataset name is required"
    if args.dataset_name == "ni":
        assert args.predict_with_generate, "predict_with_generate is required for ni"
    # else:
        
    #     assert not args.predict_with_generate, "predict_with_generate is not required for non-ni"
    
    if args.mode == "lora":
        assert args.lora_r is not None, "lora_r is required for lora"
    if args.mode == "prefix_tuning":
        assert args.prefix_len is not None, "prefix_len is required for prefix_tuning"
    
    if args.mode == "layer_tuning":
        assert args.layer_name is not None, "layer_name should be specified for layer tuning mode"
    
    cache_path = "~/tmp/cache"
    EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
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
    run_name = args.models[0] + "-" + args.dataset_name 
    if args.mode == "lora":
        run_name += "-lora_r-" + str(args.lora_r)
    elif args.mode == "prefix_tuning":
        run_name += "-prefix_len-" + str(args.prefix_len)
    elif args.mode == "layer_tuning":
        run_name += "-" + str(args.layer_name)
    if args.fp16:
        run_name += "-fp16"
    elif args.bf16:
        run_name += "-bf16"
    
    # TODO: rewrite the above run_name by concatenating a list of string
        
    run_name += args.mode + "_" + time # differentiate diff runs
    
    trainer_args = TrainerArguments(
        model_names_or_paths = args.models,
        model_arch = args.model_arch,
        dataset_name = args.dataset_name,
        dataset_config_name =args.dataset_config_name,
        evaluation_strategy="steps",
        eval_steps =  args.eval_save_steps,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=1,
        save_steps = args.eval_save_steps,
        max_steps=args.max_steps,
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
        load_best_model_at_end=True,
        save_strategy = "steps",
        fp16 = args.fp16,
        bf16 = args.bf16,
        lora_r = args.lora_r,
        prefix_len = args.prefix_len,
        overwrite_cache= args.overwrite_cache,
        run_name = run_name,
        layer_name = args.layer_name,
    )
    trainer = PEFTTrainer(trainer_args)
    import transformers
    transformers.logging.set_verbosity_warning()

    trainer.train()
