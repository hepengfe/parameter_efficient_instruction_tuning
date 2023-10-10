from transformers import HfArgumentParser
from transformers import Seq2SeqTrainingArguments
from peft_trainer import PEFTTrainer
import os
from dataclasses import dataclass, field
import shutil
# import Optional
from typing import Optional, List
from utils import flatten, build_peft_config_name, eval_hf_model
import logging
from logging import getLogger
import accelerate
logger = getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_arch: Optional[str] = field(
        default=None,
        metadata={"help": "model architecture"}
    )

    tuning_mode: str = field(
        default="adapter",
        metadata={"help": "tuning mode, either fine tuning or peft"}
    )


@dataclass
class PeftArguments:
    """
    Arguments pertaining to peft
    """

    # lora
    lora_r: int = field(
        default=1,
        metadata={"help": "lore r. default to 1 due to compatibility with ia3."}
    )
    
    lora_alpha: int = field(
        default=32,
        metadata={"help": "lore alpha."}
    )

    lora_modules: str = field(
        default="q,v",
        metadata={"help": "lore modules to be reparameterized."}
    )
    
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "dropout rate"}
    )

    # adaptor
    adapter_size: int = field(
        default=32,
        metadata={"help": "adapter size"}
    )
    
    reduction_factor: float = field(
        default=None,
        metadata={"help": "reduction factor for adaptor"}
    )

    # compactor
    phm_dimension: int = field(
        default=2,
        metadata={"help": "dimension of phm"}
    )

    # prompt tuning
    prompt_len: int = field(
        default=10,
        metadata={"help": "number of soft tokens"}
    )

    # prefix tuning
    prefix_len: int = field(
        default=None,
        metadata={"help": "prefix length"}
    )
    
    bottleneck_size: int = field(
        default=32,
        metadata={"help": "bottleneck size"}
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

    trainable_params_percentage: Optional[float] = field(
        default=None,
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
        default=False,
        metadata={ "help": "Whether to add task name to the input." },
    )
    add_task_definition: bool = field(
        default=True,
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
    per_device_test_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size per GPU/TPU core/CPU for testing."}
    )

    full_determinism: bool = field(
        default=True,
        metadata={ "help": "Whether to use full determinism." },
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
        default=False,
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
        default="cache", metadata={"help": "Where do you want to store the pretrained models."}
    )

    logging_dir: Optional[str] = field(
        default="logs", metadata={"help": "a suffix logging_dir can be passed in. Such as lora/lr_5e-4 and it will be further appended to the actual logging_dir under different training environments."}
    )

    output_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the checkpoints."}
    )

    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory."}
    )
    
    default_optimizer_n_scheduler: bool = field(
        default=False, metadata={"help": "Whether to use default optimizer and scheduler."}
    )
    logging_steps: int = field(
        default=30, metadata={"help": "Log every X updates steps."}
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

    checkpoint_save_total_limit: Optional[int] = field(
        default=3, metadata={"help": "The maximum total amount of checkpoints to save. Defaults to 3."}
    )
    
    best_checkpoint_save_total_limit:  Optional[int] = field(
        default=1, metadata={"help": "The maximum total amount of best checkpoints to save."}
    )

    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "If set, the training will override num_train_epochs and stop after max_steps."}
    )

    num_train_epochs: int = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
    
    run_name: Optional[str] = field(
        default="", metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )


    do_train: bool = field(
        default=False, metadata={"help": "Whether to run training."}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Whether to run test."}
    )
    do_traditional_test: bool = field(
        default=False, metadata={"help": "Whether to run traditional test."}
    )

    
    expr_dir : str = field(
        default="cache/tmp/", metadata={"help": "The directory for all experiments logs, checkpoints, and results."}
    )

    saved_pretrained_model_path: str = field(
        default="cache/saved_pretrained", metadata={"help": "The directory for saved pretrained model. It has a higher priority than model_cache_path."}
    )

    model_cache_path: str = field(
        default="cache/model", metadata={"help": "The directory for model cache."}
    )
    
    log_level: str = field(
        default="warning",
        metadata={ "help": "The logging level." },
    )

    eval_metric: str = field(
        default="rougeL",
    )

    dev_run: bool = field(
        default=False,
        metadata={ "help": "Whether to run in dev mode." },
    )
    
    dev_train: bool = field(
        default=False,
        metadata={ "help": "Whether to run in dev mode." },
    )
        
    dev_offline: bool = field(
        default=False,
        metadata={ "help": "Whether to run in dev mode." },
    )

    dev_eval: bool = field(
        default=False,
    )

    dev_test: bool = field(
        default=False,
        metadata={ "help": "Whether to run in test mode which check evaluation on test dataset" },
    )

    is_cluster: bool = field(
        default = False,
        metadata={ "help": "Whether to run on the cluster." },
    )


    label_smoothing_factor: float = field(
        default=0.0,
        metadata={ "help": "The label smoothing factor." },
    )

    weight_decay: float = field(
        default=0.0,
        metadata={ "help": "The weight decay." },
    )

    scheduler_type : str = field(
        default="constant",
        metadata={ "help": "The scheduler type." },
    )

    warmup_ratio: float = field(
        default=0.0,
        metadata={ "help": "The warmup ratio." },
    )
    
    
    random_seed: int = field(
        default=42,
        metadata={ "help": "The random seed." },
    )

    early_exit : bool = field(
        default=False,
        metadata={ "help": "Whether to use early exit after dataset processing." },
    )

    load_best_checkpoint: bool = field(
        default=True,
        metadata={ "help": "Whether to load best checkpoint." },
    )

ENCODER_DECODER_MODEL_NAMES = ["t5"]
DECODER_MODEL_NAMES = ["opt", "llama", "gpt2"]
def main():
    parser = HfArgumentParser((ModelArguments, PeftArguments, DataArguments, TrainingArguments))
    
    
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    accelerate.utils.set_seed(training_args.random_seed)
    if any([m in model_args.model_name_or_path for m in ENCODER_DECODER_MODEL_NAMES]):
        model_args.model_arch = "encoder-decoder"
        print(f"model {model_args.model_name_or_path} is encoder-decoder")
    elif any([m in model_args.model_name_or_path for m in DECODER_MODEL_NAMES]):
        model_args.model_arch = "decoder"
        print(f"model {model_args.model_name_or_path} is decoder")
    else:
        if model_args.model_arch is None:
            raise ValueError(f"model name or path {model_args.model_name_or_path} is not categorized into encoder-decoder or decoder. If it's model path, please specify model_arch.")


    training_args._frozen = False
    if training_args.is_cluster:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ['HF_DATASETS_OFFLINE']= "1"
        os.environ['HF_DATASETS_CACHE'] = "cache"
        os.environ["WANDB_MODE"] = "offline"
        # logging_dir
        training_args.logging_dir = os.path.join(
            "/ceph-jd/pub/jupyter/wangyizhong/notebooks/", training_args.logging_dir)
        
        training_args.expr_dir = os.path.join(
            "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit", training_args.expr_dir)
        training_args.cache_dir = os.path.join(
            "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/workspaces/peit", training_args.cache_dir)
        data_args.task_dir = "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/data/tasks"
        data_args.data_dir = "/weka-jd/prod/public/permanent/group_wangyizhong/wangyizhong/data/splits/" + data_args.data_dir.split("/")[-1]
        print("---- cluster mode ----")
        print("logging_dir: ", training_args.logging_dir)
        print("expr_dir: ", training_args.expr_dir)
        print("cache_dir: ", training_args.cache_dir)
        print("task_dir: ", data_args.task_dir)
        print("data_dir: ", data_args.data_dir)
        print("---- cluster mode ----")
        logging.getLogger().setLevel(logging.ERROR) # set all logging to error to prevent error message in warnings
    else:
        training_args.logging_dir = os.path.join("./logs", training_args.logging_dir)
        
    
    if training_args.dev_run:
        # no adjustable variables
        os.environ["WANDB_MODE"] = "disabled"
        training_args.dev_run_data_size = 2000
                # # debug logging
        training_args.save_steps = 30
        training_args.eval_steps = 30
        training_args.num_train_epochs = 4

        # adapter
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 35
        training_args.per_device_test_batch_size = 2

        # fine_tuning
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 10 # can be increased for offload
        training_args.per_device_test_batch_size = 2



        # training_args.per_device_eval_batch_size = 1
        # training_args.dev_run_data_size = 16
        # # model_args.tuning_mode = "fine_tuning"
        
        # # debug high validation rougeL
        # training_args.per_device_train_batch_size = 1
        # training_args.per_device_eval_batch_size = 2
        # training_args.per_device_test_batch_size = 10
        # training_args.dev_run_data_size = 40

        # test evaluation
        training_args.dev_run_data_size = 500
        training_args.save_steps = 50
        training_args.eval_steps = 50
        training_args.num_train_epochs = 4
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 10 # can be increased for offload
        training_args.per_device_test_batch_size = 10
        

    if training_args.dev_train:
        # dev issues such as OOM, training loss decreasing
        os.environ["WANDB_MODE"] = "disabled"
        eval_logger = logging.getLogger("compute_metrics.py")
        eval_logger.setLevel(logging.DEBUG)
        # training_args.learning_rate = 0.01
        # try to adjust train/eval bs during dev run
        training_args.dev_train_data_size = 30

        

        # test overfitting
        training_args.logging_steps = 10
        # async eval and save
        training_args.num_train_epochs = 5
        training_args.save_steps = 60
        training_args.eval_steps = 60

        training_args.save_steps = 600
        training_args.eval_steps = 200
        training_args.dev_train_data_size = 30
        training_args.per_device_eval_batch_size = 1
        # training_args.per_device_train_batch_size = 1
        training_args.per_device_train_batch_size = 1

        
        # test evaluation
        # training_args.dev_train_data_size = 100
        # training_args.save_steps = 20
        # training_args.eval_steps = 20
        # training_args.num_train_epochs = 20
        # training_args.per_device_train_batch_size = 2
        # training_args.per_device_eval_batch_size = 10 # can be increased for offload
        # training_args.per_device_test_batch_size = 10


    if training_args.dev_test:
        # test save and test eval OOM, also whether eval and test results are same
        # save at 4, 8, 10(epoch) steps
        training_args.num_train_epochs = 1
        training_args.dev_test_data_size = 50
        training_args.save_steps = 10
        training_args.eval_steps = 10
        training_args.logging_steps = 10
        training_args.per_device_eval_batch_size = 4
        training_args.per_device_train_batch_size = 1

    if training_args.dev_eval:
        # dev issues such as empty prediction (although it's mostly likely a generation issue)
        pass

    # pre tuning check
    assert data_args.dataset_name is not None, "dataset name is required"
    assert training_args.logging_steps > 0, "logging_steps should be larger than 0"

    
    if data_args.dataset_name == "ni":
        assert training_args.predict_with_generate, "predict_with_generate is required for ni"
        assert data_args.data_dir is not None, "data_dir is required for ni"
    if data_args.dataset_name == "alpaca":
        assert data_args.data_dir is None
    if model_args.tuning_mode == "layer_tuning":
        assert peft_args.layer_name is not None, "layer_name should be specified for layer tuning mode"

    if model_args.tuning_mode == "bitfit":
        if peft_args.bias_name is None:
            peft_args.bias_name = "encoder_decoder_bias"
            print("bias_name is set to encoder_decoder_bias since args.bias_name is not specified")
    # if model_args.tuning_mode == "fine_tuning":
    #     training_args.learning_rate = 1e-5
    #     print("lr is set to 1e-5 due to fine_tuning mode")

    if model_args.tuning_mode == "prefix_tuning":
        assert peft_args.prefix_len is not None, "prefix_len should be specified for prefix tuning mode"

    if training_args.per_device_test_batch_size is None:
        training_args.per_device_test_batch_size = training_args.per_device_eval_batch_size


    # extract suffix number from data_dir
    if data_args.data_dir is not None:
        import re
        result = re.findall(r'\d+', data_args.data_dir)
        if len(result) != 0:
            num_validation_tasks = int(result[-1])
    assert training_args.do_train or training_args.do_test or training_args.do_traditional_test, "At least one of `do_train` or `do_test` or `do_traditional_test` must be True."
    assert not (training_args.do_train and training_args.do_test), "do_train and do_test cannot be both True"


    
    if data_args.dataset_name == "ni":
        assert data_args.data_dir is not None, "data_dir is required for ni"
        assert data_args.task_dir is not None, "task_dir is required for ni"
        data_args.max_source_length = 1024
        data_args.max_target_length = 128
        print("max_source_length is set to 1024")
        print("max_target_length is set to 128")

    
    peft_config_name = build_peft_config_name(model_args, peft_args, training_args)

    if data_args.data_dir:
        data_folder_name = os.path.basename(data_args.data_dir)
    else:
        data_folder_name = "full_data"
    # output_dir:   xx/xx/xx
    # expr_dir/dataset/dataset_config/model/tuning_mode/model_config + training_config
    if data_args.data_dir:
        data_config_name = f"num_validation_tasks_{num_validation_tasks}"
    else:
        data_config_name = f"NoneConfig"
    random_seed_name = f"random_seed_{training_args.random_seed}"
    dev_folder = ""
    if training_args.dev_run:
        dev_folder = "dev_run"
    elif training_args.dev_train:
        dev_folder = "dev_train"
    
    output_dir = os.path.join(
            training_args.expr_dir,
            data_args.dataset_name,
            data_folder_name, 
            flatten(model_args.model_name_or_path, "/-", "_"),
            dev_folder,
            model_args.tuning_mode,
            "_".join([peft_config_name, data_config_name, random_seed_name]),
    )
    if training_args.dev_run:
        output_dir += "_dev_run"
    elif training_args.dev_train:
        output_dir += "_dev_train"
    if not training_args.output_dir:
        training_args.output_dir = output_dir

    if training_args.overwrite_output_dir:
        if os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir, ignore_errors=True)
            print(f"Removed output_dir: {training_args.output_dir}")
        if os.path.exists(training_args.logging_dir):
            shutil.rmtree(training_args.logging_dir, ignore_errors=True)
            print(f"Removed logging_dir: {training_args.logging_dir}")
        # --overwrite_output_dir in cluster should be used for only one time
        if training_args.is_cluster:
            exit()

    # run_name: xx-xx-xx
    training_args.run_name = flatten(training_args.run_name, "/", "-") # could pass in dir like run name like xx/xx/xx

    print("logging_dir: ", training_args.logging_dir)
    print("output_dir: ", training_args.output_dir)
    print("run_name: ", training_args.run_name)

    
    # either max_steps or num_train_epochs should be specified
    assert training_args.max_steps is not None or training_args.num_train_epochs is not None, "either max_steps or num_train_epochs should be specified"
    training_args.label_names = [training_args.label_names]
    trainer = PEFTTrainer(training_args, data_args, model_args, peft_args)
   

    if training_args.do_train:
        trainer.train() # train from scratch
        trainer.evaluate("test")
        logger.info(f"check the results in {training_args.output_dir}")
        logger.info("*** Training and Test finished ***")

    if training_args.do_test:
        trainer.evaluate("test")
        logger.info("*** Test finished ***")


    if training_args.do_traditional_test:
        trainer.evaluate("traditional_test")
        logger.info("*** Traditional Test finished ***")

    
if __name__ == "__main__":
    main()


