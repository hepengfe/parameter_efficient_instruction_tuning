from transformers import AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments
from peft_trainer import PEFTTrainer
import datetime
from arguments import TrainerArguments
import argparse
import os
import torch

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    # arg_parser.add_argument("--config", type=str, default="configs/gpt2.yaml")
    arg_parser.add_argument("--dataset", type=str, default="wikitext")
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
    args = arg_parser.parse_args()
    args.models = args.models.split(",")
    args.models = [m.strip() for m in args.models]
    assert args.dataset_name is not None, "dataset name is required"
    cache_path = "~/tmp/cache"
    EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
    default_optimizer_n_scheduler = False
    # if args.dev:
    #     default_optimizer_n_scheduler = True
    
    
    

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
        predict_with_generate = False,
        # predict_with_generate = True,
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
        load_best_model_at_end=True,
        save_strategy = "steps"
    )
    trainer = PEFTTrainer(trainer_args)
    trainer.train()
