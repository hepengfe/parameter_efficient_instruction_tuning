from transformers import AutoModelForSeq2SeqLM, HfArgumentParser
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, logging
from peft_trainer import PEFTTrainer
import datetime
import argparse
import os
import torch
import transformers
from dataclasses import dataclass, field

# import Optional
from typing import Optional, List




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_arch: str = field(
        default="encoder-decoder",
        metadata={"help": "model architecture"}
    )
    tuning_mode: str = field(
        default="lora",
        metadata={"help": "tuning mode, either fine tuning or peft"}
    )

    

@dataclass
class PeftArguments:
    """
    Arguments pertaining to peft
    """
    # general peft
    trainable_params_percentage: float = field(
        default=None,
        metadata={"help": "percentage of trainable parameters of peft methods w.r.t to the original pre-trained model"}
    )
    
    # lora
    lora_r: int = field(
        default=1,
        metadata={"help": "lore r. default to 1 due to compatibility with ia3."}
    )

    lora_modules: str = field(
        default="qv",
        metadata={"help": "lore modules to be reparameterized."}
    )
    
    
    # adaptor
    reduction_factor: int = field(
        default=None,
        metadata={"help": "reduction factor for adaptor"}
    )
    
    # compactor
    phm_dimension: int = field(
        default=2,
        metadata={"help": "dimension of phm"}
    )
    
    # prompt tuning
    num_soft_tokens: int = field(
        default=10,
        metadata={"help": "number of soft tokens"}
    )
    
    
    # prefix tuning
    prefix_len: int = field(
        default=None,
        metadata={"help": "prefix length"}
    )
    
    

    # bitfit
    bias_name: str = field(
        default=None,
        metadata={"help": "bias name to be tuned"}
    )
    
    # pelt
    use_pelt_gate: bool = field(
        default=False,
        metadata={"help": "whether to use pelt gate"}
    )
    
    
    # layer tuning
    layer_name: str = field(
        default=None,
        metadata={"help": "layer name to be tuned"}
    )
    
    module_device: int = field(
        default=0,
        metadata={"help": "device id of the module to be tuned"}
    )
    
    
    
    
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={ "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded." },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={ "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded." },
    )
    max_num_instances_per_eval_task: Optional[int] = field(
        default=100,
        metadata={ "help": "The maximum number of instances per eval task. If there are more instances than this number, we will sample this number of instances from the eval task." },
    )
    max_num_instances_per_task: Optional[int] = field(
        default=100,
        metadata={ "help": "The maximum number of instances per task. If there are more instances than this number, we will sample this number of instances from the task." },
    )
    
    # num_pos_examples
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={ "help": "The number of positive examples per task." },
    )
    # num_neg_examples
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={ "help": "The number of negative examples per task." },
    )
    add_task_name : bool = field(
        default=True,
        metadata={ "help": "Whether to add task name to the input." },
    )
    add_task_definition: bool = field(
        default=False,
        metadata={ "help": "Whether to add task definition to the input." },
    )
    add_explanation: bool = field(
        default=False,
        metadata={ "help": "Whether to add explanation to the input." },
    )
    pad_to_max_length: bool = field(
        default=True,   metadata={"help": "Whether to pad all samples to model maximum sentence length."}
    )
    tk_instruct: bool = field(
        default=False,  metadata={"help": "Whether to tokenize instructions."}
    )

    num_training_tasks: Optional[int] = field(
        default=None, metadata={"help": "Number of training tasks."}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    dev: bool = field(
        default=False,
        metadata={ "help": "Whether to use dev set." },
    )
    
    eval_steps: int = field(
        default=5000,
        metadata={ "help": "The number of steps to evaluate the model." },
    )
    
    save_steps: int = field(
        default=5000,
        metadata={ "help": "The number of steps to save the model." },
    )
    
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    lr: float = field(
        default=1e-5, metadata={"help": "The initial learning rate."}
    )
    
    full_determinism: bool = field(
        default=True,
        metadata={ "help": "Whether to use full determinism." },
    )
    
    seed: int = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    
    predict_with_generate: bool = field(
        default=True,
        metadata={ "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)." },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={ "help": "Number of updates steps to accumulate before performing a backward/update pass." },
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "whether to use bf16"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "whether to use fp16"}
    )
    
    eval_accumulation_steps: int = field(
        default=1,
        metadata={ "help": "Number of eval steps to accumulate before performing a backward/update pass." },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models."}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the checkpoints."}
    )
    
    default_optimizer_n_scheduler: bool = field(
        default=False, metadata={"help": "Whether to use default optimizer and scheduler."}
    )
    logging_steps: int = field(
        default=30, metadata={"help": "Log every X updates steps."}
    )
    model_parallel_gpus: int = field(
        default=6, metadata={"help": "Number of GPUs to use for model parallel training."}
    )
    learning_rate: float = field(
        default=5e-4, metadata={"help": "The initial learning rate."}
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    label_names: Optional[str] = field(
        default="labels", metadata={"help": "The list of labels."}
    )
    logging_strategy: str = field(
        default="steps", metadata={"help": "The logging strategy."}
    )
    
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "The evaluation strategy."}
    )
    eval_steps: int = field(
        default=5000, metadata={"help": "Run an evaluation every X steps."}
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "The save strategy."}
    )
    save_steps: int = field(
        default=5000, metadata={"help": "Save checkpoint every X steps."}
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Whether to load the best model found during training at the end of training."}
    )
    
    
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": "The maximum total amount of checkpoints to save."}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "If set, the training will override num_train_epochs and stop after max_steps."}
    )
    num_train_epochs: float = field(
        default=2.0, metadata={"help": "Total number of training epochs to perform."}
    )
    
    run_name: Optional[str] = field(
        default="", metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    

    

    
if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, PeftArguments, DataArguments, TrainingArguments))
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert data_args.dataset_name is not None, "dataset name is required"
    if data_args.dataset_name == "ni":
        assert training_args.predict_with_generate, "predict_with_generate is required for ni"
    
    if peft_args.trainable_params_percentage is None:
        if model_args.tuning_mode == "lora":
            assert peft_args.lora_r is not None, "lora_r is required for lora if trainable_params_percentage is not specified"
        if model_args.tuning_mode == "prefix_tuning":
            assert peft_args.prefix_len is not None, "prefix_len is required for prefix_tuning"

    
    if model_args.tuning_mode == "layer_tuning":
        assert peft_args.layer_name is not None, "layer_name should be specified for layer tuning mode"
    
    if model_args.tuning_mode == "bitfit":
        if peft_args.bias_name is None:
            peft_args.bias_name = "encoder_decoder_bias"
            print("bias_name is set to encoder_decoder_bias since args.bias_name is not specified")
        
    if model_args.tuning_mode == "fine_tuning":
        training_args.lr = 1e-5
        print("lr is set to 1e-5 due to fine_tuning mode")
        
    

    # extract suffix number from data_dir
    if data_args.data_dir is not None:
        import re
        result = re.findall(r'\d+', data_args.data_dir)
        if len(result) != 0:
            data_args.num_training_tasks = int(result[-1])


    cache_path = "~/tmp/cache"
    EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
    training_args.cache_dir = cache_path
    training_args.output_dir = output_path
    # add random number into output_path
    import random
    output_path += "_" + str(random.randint(0, 10000))
    default_optimizer_n_scheduler = False

    
    if data_args.dataset_name == "ni":
        assert data_args.data_dir is not None, "data_dir is required for ni"
        assert data_args.task_dir is not None, "task_dir is required for ni"
        data_args.max_source_length = 1024
        data_args.max_target_length = 128
        print("max_source_length is set to 1024")
        print("max_target_length is set to 128")
    

    run_name_list = []
    run_name_list.append(model_args.tuning_mode)
    run_name_list.append(data_args.dataset_name)
    if peft_args.trainable_params_percentage is None: # if use preset config
        if model_args.tuning_mode == "lora":
            run_name_list += ["lora_r", str(peft_args.lora_r)]
        if model_args.tuning_mode == "prefix_tuning":
            run_name_list += ["prefix_len", str(peft_args.prefix_len)]
    if training_args.fp16:
        run_name_list.append("fp16")
    elif training_args.bf16:
        run_name_list.append("bf16")
    run_name_list.append(model_args.tuning_mode)
    run_name_list.append("lr_" + str(training_args.lr))
    run_name = "-".join(run_name_list)
    # either max_steps or num_train_epochs should be specified
    assert training_args.max_steps is not None or training_args.num_train_epochs is not None, "either max_steps or num_train_epochs should be specified"
    training_args.label_names = [training_args.label_names]
    trainer = PEFTTrainer(training_args, data_args, model_args, peft_args)
    
    transformers.logging.set_verbosity_warning()

        
    if training_args.do_eval:
        trainer.evaluate()
    else:
        trainer.train()

