from transformers import AutoModelForSeq2SeqLM
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptTuningConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments
from peft_trainer import PEFTTrainer
import datetime
from arguments import TrainerArguments
import argparse
import os
model_name_or_path = "t5-large"
tokenizer_name_or_path = "t5-large"

# peft_config = PromptTuningConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM,num_virtual_tokens=10, inference_mode=False
# )

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)

def get_soft_prompt_token_list(num_soft_prompt_tokens):
    """
    Get soft prompt tokens from vocab.
    """
    return [f"<|softprompt{i}|>" for i in range(num_soft_prompt_tokens)]

# def _format_prompts(prefix, source_strs, verbal_targets, include_labels_in_input = False):
#     prompts = [""] * len(source_strs)
#     prompts = [""]* len(source_strs)
#     formatted_inputs = [f"{prefix} {s_1} {prompt} " for s_1, prompt in zip(source_strs, prompts)]
#     formatted_inputs = [f"{input} " for input in formatted_inputs]

#     if include_labels_in_input:
#         labels = ",".join(verbal_targets)
#         formatted_inputs = [f"{input} Decide the label in {self.verbalizers}." for input in formatted_inputs]    

#     return formatted_inputs, verbal_targets


# raw_datasets = load_dataset("sst2")
# def preprocess(examples, class_ids = [0,1], evaluation=False):
#     num_soft_tokens = 10
#     prefix_prompt = "".join(get_soft_prompt_token_list(num_soft_tokens))
#     if num_soft_tokens ==0:
#         assert prefix_prompt == ""

#     inputs =["Sentence: " + sent + "Sentiment:" for sent, label_id in zip(examples["sentence"], examples["label"]) if label_id in class_ids]
#     # verbalize the sentiment id to tokens
#     # it's not used for t5 evaluation though (label id is used instead)
#     verbal_targets = [verbalizers[l]
#             for l in examples["label"] if l in class_ids]
#     formatted_inputs, verbal_targets =\
#                 _format_prompts(prefix_prompt,
#                                     inputs,
#                                     # [self.prefix_prompt]*len(inputs),
#                                     verbal_targets,
#         )

#     model_inputs = tokenizer(
#         formatted_inputs,
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#         return_tensors='pt'
#     )
    
#     # build label input ids
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             verbal_targets,
#             return_tensors='pt', padding="max_length", 
#             max_length=128,
#             # padding=self.padding,
#       =      truncation=True,
#         )

#     labels["input_ids"][labels["input_ids"]==0] = -100
#         # labels["input_ids"] = [
#         #     [(l if l != self.tokenizer.pad_token_id else -100)
#         #     for l in label]
#         #     for label in labels["input_ids"]
#         # ]
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
# verbalizers = ['terrible', 'great']

# column_names = raw_datasets["train"].column_names
# train_dataset = raw_datasets["train"].map(
#                 preprocess,
#                 batched=True,
#                 remove_columns= column_names,
#                 num_proc=1,
#                 # load_from_cache_file=self.arguments.dataset_cache,
#                 fn_kwargs = {"evaluation": False},
#                 # fn_kwargs = {"evaluation": True},
#                 desc="Running tokenizer on train dataset",
#             )
# eval_dataset = raw_datasets["validation"].map(
#                 preprocess,
#                 batched=True,
#                     remove_columns=column_names,
#                     num_proc=1,
#                     # load_from_cache_file=self.arguments.dataset_cache,
#                     # fn_kwargs = {"evaluation": False},  # hf internal validation
#                     fn_kwargs = {"evaluation": True},
#                     desc="Running tokenizer on validation dataset",
#                 )

# dataset_names = ("sst2", None)
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    # arg_parser.add_argument("--config", type=str, default="configs/gpt2.yaml")
    arg_parser.add_argument("--dataset", type=str, default="wikitext")
    arg_parser.add_argument("--models", type=str, default="google/t5-large-lm-adapt")
    arg_parser.add_argument("--mode", type=str, default="prompt_tuning")
    arg_parser.add_argument("--dev", action="store_true")
    arg_parser.add_argument("--max_steps", type=int, default=30000)
    arg_parser.add_argument("--eval_save_steps", type=int, default=2500)
    arg_parser.add_argument("--per_device_train_batch_size", type=int, default=12)
    arg_parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    arg_parser.add_argument("--lr", type=float, default=0.3)
    arg_parser.add_argument("--num_soft_tokens", type=int, default=5)
    arg_parser.add_argument("--early_stop", action="store_true")
    arg_parser.add_argument("--test_train", action="store_true")
    arg_parser.add_argument("--paper_mode", action="store_true")
    arg_parser.add_argument("--eval", action="store_true")
    args = arg_parser.parse_args()
    args.models = args.models.split(",")
    args.models = [m.strip() for m in args.models]
    dataset_names= ("sst2", None)
    # dataset_names= ("super_glue","boolq")
    cache_path = "~/tmp/cache"
    EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
    default_optimizer_n_scheduler = False
    if args.dev:
        default_optimizer_n_scheduler = True
    
    trainer_args = TrainerArguments(
        model_names_or_paths = args.models,
        dataset_name = dataset_names[0],
        dataset_config_name = dataset_names[1],
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
    )
    trainer = PEFTTrainer(trainer_args)
    # train_dataset.set_format(type="torch")
    # trainer.train_dataset = train_dataset
    # eval_dataset.set_format(type="torch")
    # trainer.eval_dataset = eval_dataset

    trainer.train()