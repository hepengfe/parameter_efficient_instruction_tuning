from datasets import load_dataset
import numpy as np
import torch
from transformers import (
    AdamW,
    get_scheduler,
)
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from functools import partial
from transformers import (
    AutoTokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq
)
from transformers.optimization import AdamW
import transformers
from peft import get_peft_model, TaskType, PromptTuningConfig
from util.ni_dataset_collator import DataCollatorForNI
from copy import deepcopy
from utils import get_latest_checkpoint, remove_old_checkpoints, remove_files_and_folders_other_than, verify_complete_random_states, check_all_checkpoints_and_remove
import json
from accelerate import Accelerator
from tqdm.auto import tqdm
import shutil
from util.compute_metrics import compute_metrics, compute_grouped_metrics
from accelerate.utils import DistributedType
import time
from transformers.trainer_pt_utils import LabelSmoother
# modules use two pacakges 
ADAPTER_TRANSFORMERS_MODULES=["ia3"]
# ADAPTER_TRANSFORMERS_MODULES=[ "compactor", "prefix_tuning", "lora_adapter", "adapter_adapter","ia3"]
PEFT_MODULES=["prompt_tuning", "lora_peft", "bitfit", "adapter_peft", "prefix_tuning"]
CAUSAL_LM=["gpt", "llama", "opt"]


BEST_CP_FOLDER_NAME="best_checkpoint"
LATEST_CP_FOLDER_NAME="latest_checkpoint"
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import logging
from accelerate.logging import get_logger
import accelerate
import pandas as pd


logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)



class TrainingState:
    """
    Track current training state.
    """
    def __init__(self, training_args, global_step=0, loss=0, best_metric_val=0, eval_metric="rougeL"):
        
        for k in list(training_args.keys()):
            training_args[f"training_args/{k}"] = training_args.pop(k)
        self.training_args = training_args

        self.state_dict = {
            "epoch": 0,
            "step": 0,
            "global_step": global_step,
            "loss": loss,
            "best_metric_step":-1,
            "best_metric_val":best_metric_val,
            "eval_metric":eval_metric,
            "test_eval_finished": False,
            "traditional_test_eval_finished": False,
            "train_finished": False,
            "trainable_params": 0,
            "total_model_params": 0,
            "trainable_ratio": 0,
            "optimization_step": 0,
        }
        self.file_name = "training_state.json"

    def get(self, k):
        if k in self.state_dict:
            return self.state_dict[k]
        elif "train_finished" == k: # if it's not in k, then it's false
            return False
        elif "traditional_test_eval_finished" == k:
            return False
        elif k == "optimization_step":
            return 0
        else:
            if hasattr(self, k):
                return getattr(self,k)
            else:
                raise ValueError(
                    f"{k} cannot be found in train state"
                )

    def update(self, dict):
        self.state_dict.update(dict)

    def to_dict(self):
        return dict([(k, v) for k, v in self.__dict__.items() if not k.startswith("_")])

    def save_to_json(self, cp_path):
        if cp_path is None:
            return
        file_path = os.path.join(cp_path, self.file_name)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f)

    def load_from_json(self, cp_path):
        file_path = os.path.join(cp_path, self.file_name)
        with open(file_path, "r") as f:
            data = json.load(f)
        self.state_dict = data["state_dict"]
        self.training_args = data["training_args"]

    def __str__(self):
        return str(self.to_dict())



class PEFTTrainer:
    def __init__(self, training_args, data_args, model_args, peft_args):
        
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.peft_args = peft_args
        self.model_name_or_path = self.model_args.model_name_or_path
        self.potential_model_path =  os.path.join(
            self.training_args.saved_pretrained_model_path,
            self.model_name_or_path
        )
        
        self.model = None
        self.model_trainable_params = None
        self.recover_from = None
        
        # init
        self.best_metric_val = -1
        self.best_metric_step = -1
        
        
        self.start_epoch = 0 # start epoch from last checkpoint if exists, otherwise 0. when saving training state, it will be updated to latest epoch.
        
        self.start_step = 0 # start step from last checkpoint if exists
        self.epoch = 0 # current epoch
        self.step = 0 # current step at each epoch
        self.global_step = 0 # current global step
        self.total_optimization_step = 0 # total optimization step
        self.optimization_step = 0 # current optimization step


        self.train_finished = False
        self.test_eval_finished = False
        self.traditional_test_eval_finished = False
        
        self.model_lm_head_weight = None
        if self.model_args.model_arch != "decoder" and self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
            self.model_lm_head_weight = AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path).lm_head.weight

        
        
        self.accelerator = Accelerator(
                log_with="tensorboard",
                # logging_dir=self.training_args.logging_dir,
                project_dir=self.training_args.output_dir,
                gradient_accumulation_steps = self.training_args.gradient_accumulation_steps,
        )
        
        # deepspeed setting can be considered as distributed
        self.use_distributed = self.accelerator.use_distributed or self.accelerator.distributed_type == DistributedType.DEEPSPEED
        self.distributed_type = self.accelerator.distributed_type
        self.num_processes = self.accelerator.num_processes

        self.train_state = TrainingState(
            self.training_args.to_dict(), 
            eval_metric = self.training_args.eval_metric
        )
        self.accelerator.init_trackers(
                        self.training_args.run_name,
                        config=self.train_state.state_dict,
                        init_kwargs={"tensorboard": {"flush_secs": 60}},
                    )
        self.total_step = 1
        self.label_smoother = LabelSmoother(epsilon=self.training_args.label_smoothing_factor) if self.training_args.label_smoothing_factor > 0 else None
        self.load_tokenzier()
        self.build_dataloader()
        assert self.label_smoother is None
        # model needs to be loaded on all machines
        self.load_model_n_peft_module()
        
        # TODO: accelerator needs to load model and peft module first anyway
        # is there anyway to not load the original model? since if model is large then it will take a lot of time
        assert self.model is not None, "model should loaded"

        # also resize embedding here
        # self.load_tokenzier()
        assert self.tokenizer is not None, "tokenizer should loaded"
        # resize token embedding will set requires_grad back to True
        # we need to set it back to False
        
        if isinstance(self.model, PeftModel) and self.model_args.tuning_mode not in ["prompt_tuning", "prefix_tuning"]:
            # NOTE: for prompt tuning and prefix tuning, there is no model wrapper in peft package
            model = self.model.model
        else:
            model = self.model
        if self.model_args.tuning_mode != "fine_tuning":
            if "gpt2" in model_args.model_name_or_path:
                model.transformer.wte.weight.requires_grad = False
                model.transformer.wpe.weight.requires_grad = False
                model.lm_head.weight.requires_grad = False
            elif "llama" in model_args.model_name_or_path:
                model.lm_head.weight.requires_grad = False
                model.model.embed_tokens.weight.requires_grad = False
            elif "opt" in model_args.model_name_or_path:
                # check if it's type PeftModelForCausalLM
                model.model.decoder.embed_tokens.weight.requires_grad = False
                model.lm_head.weight.requires_grad = False

                
        trainable_params_percent = self.check_trainable_parameters()
        
        # self.total_step = -1
        # self.build_dataloader()
        # assert self.total_step > 0
        
        # some scheduler require num_training_steps which is depedent on len(dataset)
        self.load_optimizer_n_scheduler()
        
        if self.use_distributed:
            if self.distributed_type == DistributedType.DEEPSPEED:
                # model prepare should be called with dataloader prepare in deepspeed mode
                self.model, self.optimizer, self.scheduler, self.train_dataloader= self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_dataloader)
            elif self.distributed_type == DistributedType.MULTI_GPU:
                # model prepare should be called before optimizer prepare
                self.model, self.train_dataloader = self.accelerator.prepare(self.model, self.train_dataloader)
                self.optimizer, self.scheduler= self.accelerator.prepare(self.optimizer, self.scheduler)
            else:
                raise NotImplementedError(f"self.distributed_type {self.distributed_type} is not implemented")
            if self.data_args.dataset_name != "alpaca":
                self.eval_dataloader, self.test_dataloader, self.traditional_test_dataloader = self.accelerator.prepare(self.eval_dataloader, self.test_dataloader, self.traditional_test_dataloader)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)


    def load_model_n_peft_module(self):
        self.model = self.load_pretrained_model()
        self.configure_n_load_peft_module() # always load model from scratch for accelerate

    def load_optimizer_n_scheduler(self):
        if not self.distributed_type == DistributedType.DEEPSPEED:
            # DDP, keep parameters require_grad status
            # create AdamW optimizer
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.training_args.learning_rate,
                # eps=self.training_args.adam_epsilon,
                weight_decay=self.training_args.weight_decay,
            )
            # Create the learning rate scheduler.
            # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
            # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
            # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
            # number of updates in the end matches the num_training_steps here.
            # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
            # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.

            self.scheduler = get_scheduler(
                    name=self.training_args.scheduler_type,
                    optimizer=self.optimizer,
                    num_training_steps=self.num_training_steps_for_scheduler,
                    num_warmup_steps=self.warmup_steps_for_scheduler
            )

        else:
            # deepspeed
            # lora adapter and other adapter methods
            if self.model_args.tuning_mode not in ["fine_tuning"] + PEFT_MODULES:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for p in self.model.parameters() if p.requires_grad],
                        "lr": self.training_args.learning_rate,
                        "weight_decay": self.training_args.weight_decay,
                    },
                    {
                        "params": [p for p in self.model.parameters() if not p.requires_grad],
                        "lr": 0,
                        "weight_decay": 0.0
                    },
                ]

                for param in self.model.parameters():
                    param.requires_grad = True
                self.optimizer = accelerate.utils.DummyOptim(
                    optimizer_grouped_parameters,
                    lr=self.training_args.learning_rate   
                )
            else:
                # fine tuning, peft package methods
                self.optimizer = accelerate.utils.DummyOptim(
                    self.model.parameters(),
                    lr=self.training_args.learning_rate,
                    weight_decay=self.training_args.weight_decay,
                )

            assert self.optimizer.lr == self.training_args.learning_rate, "optimizer learning rate is not set successfully"
            self.print_log(f"Learning rate(lr) is set to {self.optimizer.lr}", )

            
            self.scheduler = accelerate.utils.DummyScheduler(
                self.optimizer,
                warmup_num_steps=self.warmup_steps_for_scheduler,
                total_num_steps=self.num_training_steps_for_scheduler
            )


        # some test for different peft setup to align original paper setup
        if "lora" in self.model_args.tuning_mode:
            assert self.training_args.scheduler_type == "linear"
            # assert self.training_args.warmup_steps == 500


    def load_tokenzier(self):

        if os.path.exists(self.potential_model_path):
            if "llama" in self.model_args.model_name_or_path.lower():
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.potential_model_path,
                    truncation_side = "left" # NOTE: this is important for causal lm data prepare in case </sep> is truncated
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.potential_model_path,
                    truncation_side = "left" # NOTE: this is important for causal lm data prepare in case </sep> is truncated
                    )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                # use_cache = self.arguments.use_cache,
                truncation=True,
                max_length=512,
                use_fast=True,
                return_tensors="pt"
            )
        
        if any([m in self.model_name_or_path for m in CAUSAL_LM]):
            # gpt2 model
            self.tokenizer.add_special_tokens({
                'pad_token': '</PAD>',
                'sep_token': '</SEP>',
            })
            self.model.resize_token_embeddings(len(self.tokenizer))

        
        
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        
        if "gpt2" in self.model_name_or_path or "llama" in self.model_name_or_path:
            print('gpt2/llama requires padding to max length')
            self.padding = "max_length"

    def load_pretrained_model(self, config=None):
        """
        1. Load model, tokenizer by model architecture and peft packages.
        2. load model from potential checkpoint/saved_pretrained model
        3. handles model parallel if needed.
        NOTE: it doesn't load peft module if it's not from checkpoint.
        """
        logging.info(f"Loading {self.model_args.model_name_or_path} (for large models, this might take a while)")
        logging.info(f"Files will be cached at: {self.training_args.cache_dir}")
        logging.info(f"Ensure this directory is persistent if you do not want to download model files again!")

        if "t5" in self.model_name_or_path or "bart" in self.model_name_or_path:
            if self.model_args.tuning_mode in ["fine_tuning", "prompt_tuning"]:
                
                if os.path.exists(self.potential_model_path):
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path, config = config)
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir, config = config)
            elif self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
                from transformers import AutoAdapterModel
                # adapter model + seq2seq lm head (replace lm head with original t5-lm head weights)
                if os.path.exists(self.potential_model_path):
                    model = AutoAdapterModel.from_pretrained(self.potential_model_path, config = config)
                else:
                    model = AutoAdapterModel.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir, config = config)

            elif self.model_args.tuning_mode in PEFT_MODULES:
                # NOTE: this is not compatible if loading for the first time as
                # for peft package, loading by AutoModelForSeq2SeqLM is good enough
                
                if os.path.exists(self.potential_model_path):
                    model =AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path, config = config)
                else:
                    model =AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path, cache_dir=self.training_args.cache_dir, config = config)
            else:
                raise NotImplementedError("Tuning mode not supported: " + self.model_args.tuning_mode)

        elif "llama" in self.model_name_or_path.lower():
            if self.model_args.tuning_mode in ["fine_tuning", "prompt_tuning", "adapter_peft", "lora_peft"] or self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
                model = LlamaForCausalLM.from_pretrained(self.potential_model_path, config = config)
            else:
                raise NotImplementedError("Tuning mode not supported: " + self.model_args.tuning_mode)
        elif "gpt2" in self.model_name_or_path or "bloom" in self.model_name_or_path or "opt" in self.model_name_or_path:
            from transformers import AutoModelForCausalLM
            if os.path.exists(self.potential_model_path):
                model = AutoModelForCausalLM.from_pretrained(self.potential_model_path, config = config)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    # from_tf=bool(".ckpt" in self.model_name_or_path),
                    # config=m_config,
                    cache_dir=self.training_args.cache_dir,
                    config = config
                )
        else:
            raise NotImplementedError("Model not supported: " + self.model_name_or_path)

        return model


    def build_dataloader(self):
        self.load_data_collator()
        self.load_dataset()

        min_eval_data_size_per_process = self.num_processes * self.training_args.per_device_eval_batch_size
        min_test_data_size_per_process = self.num_processes * self.training_args.per_device_test_batch_size
        # adjust dataset size based on distribution environment
        if self.data_args.dataset_name != "alpaca" and self.use_distributed:
            assert len(self.eval_dataset) >= min_eval_data_size_per_process, f"eval dataset size {len(self.eval_dataset)} must be greater than {min_eval_data_size_per_process} examples"

            assert len(self.test_dataset) >= min_test_data_size_per_process, f"test dataset size {len(self.test_dataset)} must be greater than {min_test_data_size_per_process} examples"

            if len(self.eval_dataset) % min_eval_data_size_per_process != 0:
                org_len = len(self.eval_dataset)
                new_size = len(self.eval_dataset)  - len(self.eval_dataset) % min_eval_data_size_per_process 

                self.eval_dataset = self.eval_dataset.select(range(new_size))
                new_len = len(self.eval_dataset)
                self.print_log(f"process {self.accelerator.process_index}: eval dataset size must be divisible by number of processes*eval_batch_size {self.num_processes}, truncating from {org_len} to {new_len} examples")

            if len(self.test_dataset) % min_test_data_size_per_process != 0:
                org_len = len(self.test_dataset)
                new_len = len(self.test_dataset)  - len(self.test_dataset) % min_test_data_size_per_process
                self.test_dataset = self.test_dataset.select(range(new_len))
                self.print_log(f"test dataset size must be divisible by number of processes*test_batch_size {min_test_data_size_per_process}, truncating from {org_len} to {new_len} examples")
            
            if len(self.traditional_test_dataset) % min_test_data_size_per_process != 0:
                org_len = len(self.traditional_test_dataset)
                new_len = len(self.traditional_test_dataset)  - len(self.traditional_test_dataset) % min_test_data_size_per_process
                self.traditional_test_dataset = self.traditional_test_dataset.select(range(new_len))
                self.print_log(f"traditional test dataset size must be divisible by number of processes*test_batch_size {min_test_data_size_per_process}, truncating from {org_len} to {new_len} examples")

            assert len(self.eval_dataset) % min_eval_data_size_per_process == 0, f"eval dataset size {len(self.eval_dataset)} must be divisible by number of processes*eval_batch_size {min_eval_data_size_per_process}"
            assert len(self.test_dataset) % min_test_data_size_per_process == 0, f"test dataset size {len(self.test_dataset)} must be divisible by number of processes*test_batch_size {min_test_data_size_per_process}"
            assert len(self.traditional_test_dataset) % min_test_data_size_per_process == 0, f"traditional test dataset size {len(self.traditional_test_dataset)} must be divisible by number of processes*test_batch_size {min_test_data_size_per_process}"
        self.load_dataloader()
        if self.training_args.early_exit:
            exit()

        train_bs_per_step = self.training_args.per_device_train_batch_size * self.num_processes
        # with gradient accumulation, per gradient update step is actually multiple steps
        self.total_step = self.training_args.num_train_epochs * len(self.train_dataset) // train_bs_per_step
        self.warmup_steps = self.total_step * self.training_args.warmup_ratio
        self.print_log(f"total_step: {self.total_step}, warmup_steps: {self.warmup_steps}", print_step=False)
        import math
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / (self.training_args.gradient_accumulation_steps * self.accelerator.num_processes * self.training_args.per_device_train_batch_size))
        self.optimization_step = 0
        self.total_optimization_step = self.training_args.num_train_epochs * num_update_steps_per_epoch
        self.print_log(f"total_optimization_step: {self.total_optimization_step}", print_step=False)
        self.training_args.eval_steps = self.total_optimization_step // self.training_args.eval_times
        self.training_args.save_steps = self.training_args.eval_steps//5
        self.print_log(f"eval_steps: {self.training_args.eval_steps}, save_steps: {self.training_args.save_steps}", print_step=False)

        if self.model_args.tuning_mode == "fine_tuning":
            assert self.warmup_steps == 0, f"constant lr for fine tuning, but got warmup steps {self.warmup_steps}"
        else:
            assert self.warmup_steps > 0, f"lr warmup steps should be larger than 0, but got {self.warmup_steps}"
        
        self.num_training_steps_for_scheduler = self.total_step * self.accelerator.num_processes
        self.warmup_steps_for_scheduler = self.num_training_steps_for_scheduler * self.training_args.warmup_ratio

        self.num_training_steps_for_scheduler = self.total_optimization_step * self.training_args.gradient_accumulation_steps * self.accelerator.num_processes
        self.warmup_steps_for_scheduler = self.num_training_steps_for_scheduler * self.training_args.warmup_ratio

        


    def load_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=self.data_collator
        )
        # no eval for alpaca dataset training
        if self.data_args.dataset_name != "alpaca":
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                shuffle=False,
                batch_size=self.training_args.per_device_eval_batch_size,
                # collate_fn=self.data_collator,
                collate_fn=partial(self.data_collator, eval_mode=True)
            )

            self.test_dataloader = DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.training_args.per_device_test_batch_size,
                # collate_fn=self.data_collator,
                collate_fn=partial(self.data_collator, eval_mode=True)
            )

            self.traditional_test_dataloader = DataLoader(
                self.traditional_test_dataset,
                shuffle=False,
                batch_size=self.training_args.per_device_test_batch_size,
                # collate_fn=self.data_collator,
                collate_fn=partial(self.data_collator, eval_mode=True)
            )


    def load_dataset(self):
        """
        dataset loading pipeline:
        1. load all dataset (train, eval, test)
        2. preprocess dataset
        3. dataloader with tokenizer inside, it requires tokenizer to provide padding token id
        4. return dataloader
        
        """
        if self.data_args.dataset_name == "ni":
            assert self.data_args.task_dir is not None, "task_dir is required for NaturalInstructions dataset"
            assert self.data_args.data_dir is not None, "data_dir is required for NaturalInstructions dataset"
            # Get the NaturalInstructions dataset
            raw_datasets = load_dataset(
                "util/ni_dataset.py",
                data_dir=self.data_args.data_dir,
                task_dir=self.data_args.task_dir,
                cache_dir=self.training_args.cache_dir,
                max_num_instances_per_task=self.data_args.max_num_instances_per_task,
                max_num_instances_per_eval_task=self.data_args.max_num_instances_per_eval_task,
                download_mode = "reuse_dataset_if_exists" if not self.data_args.overwrite_cache else "force_redownload",
                random_seed = 42, # it will affect the cache file name, so better fix it
            )

            if self.training_args.dev_run:
                raw_datasets["train"] = raw_datasets["train"].select(range(self.training_args.dev_run_data_size))
                raw_datasets["validation"] = raw_datasets["validation"].select(range(self.training_args.dev_run_data_size))
                raw_datasets["test"] = raw_datasets["test"].select(range(self.training_args.dev_run_data_size))

            elif self.training_args.dev_train:
                raw_datasets["train"] =  raw_datasets["train"].select(range(self.training_args.dev_train_data_size))
                # raw_datasets["validation"] = raw_datasets["train"]
                # raw_datasets["train"] =  raw_datasets["train"]

                # short train
                raw_datasets["validation"] = raw_datasets["train"].select(range(self.training_args.dev_train_data_size))
                raw_datasets["test"] = raw_datasets["test"].select(range(self.training_args.dev_train_data_size))
                
                # long train
                # select random 300 examples from validation and 500 examples from test
                # import random
                # random.seed(42)
                # raw_datasets["validation"] = raw_datasets["validation"].select(random.sample(range(len(raw_datasets["validation"])), 300))
                # raw_datasets["test"] = raw_datasets["test"].select(random.sample(range(len(raw_datasets["test"])), 500))
                # raw_datasets["trainditional_test"] = raw_datasets["traditional_test"].select(range(self.training_args.dev_train_data_size))
            elif self.training_args.dev_test:
                # test compute metrics are same for validation and test as
                # test evaluation load model from checkpoint and run on test dataset
                raw_datasets["train"] =  raw_datasets["train"].select(range(self.training_args.dev_test_data_size))
                raw_datasets["validation"] = raw_datasets["train"]
                raw_datasets["test"] = raw_datasets["train"]
                raw_datasets["traditional_test"] = raw_datasets["traditional_test"].select(range(self.training_args.dev_test_data_size))

            self.train_dataset = raw_datasets["train"]
            self.eval_dataset = raw_datasets["validation"]
            self.test_dataset = raw_datasets["test"]
            self.traditional_test_dataset = raw_datasets["traditional_test"]
        elif self.data_args.dataset_name == "alpaca":
            from utils import encode_with_messages_format
            data_files = {}
            dataset_args = {}
            data_dir="data/processed/stanford_alpaca"
            data_files["train"] = os.path.join(data_dir, "stanford_alpaca_data.jsonl")
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=data_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            encode_function = partial(
                encode_with_messages_format,
                tokenizer=self.tokenizer,
                max_seq_length=self.data_args.max_source_length, # self.data_args.max_seq_length,
            )
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=1, # data_args.preprocessing_num_workers,
                remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
                load_from_cache_file=True, # not data_args.overwrite_cache,
                desc="Tokenizing and reformatting instruction data",
            )
            lm_datasets.set_format(type="pt")
            lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

            self.train_dataset = lm_datasets["train"]
            if self.training_args.dev_test:
                self.train_dataset = lm_datasets["train"].select(range(self.training_args.dev_test_data_size))

        else:
            raise NotImplementedError("New implementation no train,valid,test.   Dataset not supported: " + self.data_args.dataset_name)


                
    def load_data_collator(self):
        if self.data_args.dataset_name == "ni":
            dataset_dependent_data_collator = DataCollatorForNI(
                self.tokenizer,
                model=self.model,
                model_arch=self.model_args.model_arch,
                padding="max_length" if self.data_args.pad_to_max_length else "longest",
                max_source_length=self.data_args.max_source_length,
                max_target_length=self.data_args.max_target_length,
                label_pad_token_id=self.tokenizer.pad_token_id,
                pad_to_multiple_of=8 if self.training_args.bf16 else None,
                add_task_name=self.data_args.add_task_name,
                add_task_definition=self.data_args.add_task_definition,
                num_pos_examples=self.data_args.num_pos_examples,
                num_neg_examples=self.data_args.num_neg_examples,
                add_explanation=self.data_args.add_explanation,
                tk_instruct=self.data_args.tk_instruct
            )
            self.training_args.remove_unused_columns = False
        elif self.data_args.dataset_name == "alpaca":
            dataset_dependent_data_collator = DataCollatorForSeq2Seq(
                                                tokenizer=self.tokenizer,
                                                model=self.model,
                                                padding="longest",
                                                # batch_size=self.training_args.per_device_train_batch_size,
            )
        
        else:
            dataset_dependent_data_collator = default_data_collator
        self.data_collator = dataset_dependent_data_collator


    def load_peft_module(self, peft_config=None, reset_peft=False):
        """
        1. prepare peft model
        2. set up trainer

        Args:
            peft_config (_type_): _description_
        """
        adapter_name = self.model_args.tuning_mode
        if self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES: # prefix_tuning
            
            
            # add and activate adapter
            self.model.add_adapter(adapter_name, config = peft_config, overwrite_ok=reset_peft)
            self.model.train_adapter(adapter_name)

            
            lm_head_adapter_name = f"lm_head-{adapter_name}"
            # trainer.model
            if self.model_args.model_arch == "encoder":
                self.model.add_classification_head(lm_head_adapter_name, num_labels=2, overwrite_ok=reset_peft)
            elif self.model_args.model_arch == "encoder-decoder":
                self.model.add_seq2seq_lm_head(lm_head_adapter_name, overwrite_ok=reset_peft)
                self.model.heads[lm_head_adapter_name][0].weight = self.model_lm_head_weight
                self.model.heads[lm_head_adapter_name][0].weight.requires_grad = False
                del self.model_lm_head_weight
                import gc
                gc.collect()
            elif self.model_args.model_arch == "decoder":
                pass
                # since we don't fine tune causal lm head and inherit
                # llama causal model directly, we don't need to add lm head
            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.model_args.model_arch}"
                )

            self.model.set_active_adapters(adapter_name)
            # self.model.freeze_model(True)
            if self.model.active_adapters is None:
                raise ValueError(
                    "Expected a model with an active adapter setup."
                    "If you want to fully finetune the model use the Trainer class."
                )

            
        elif self.model_args.tuning_mode == "bitfit":
            for param in self.model.parameters():
                param.requires_grad = False
            for name, module in self.model.named_modules():
                if hasattr(module, "bias"):
                    if module.bias is None:
                        module.bias = torch.nn.Parameter(torch.zeros(module.out_features))
                    # pytorch Parameter by default requires grad
                    if not module.bias.requires_grad:
                        module.bias.requires_grad = True
        else:
            # NOTE: prompt tuning
            # general peft converting based on different peft config
            assert peft_config is not None, "peft config should be provided for non-adapter peft method"
            
            if reset_peft:
                # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir)
                self.model = deepcopy(self.model_cache)
            # add tokens in models and tokenizers + freeze model
            self.model.enable_input_require_grads()
            
            self.model = get_peft_model(self.model, peft_config)



    def check_trainable_parameters(self, print_params_required_grad = False):
        # total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # print_params_required_grad = True
        if print_params_required_grad:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    print(n,p.data.shape)
        # translate trainable_params to human readable format
        def human_readable_format(num, precision=3, suffixes=['', 'K', 'M', 'G', 'T', 'P']):
            m = sum([abs(num/1000.0**x) >= 1 for x in range(1, len(suffixes))])
            return f'{num/1000.0**m:.{precision}f}{suffixes[m]}'
        self.model_trainable_params = sum(p.numel() for p in self.model.parameters())
        if self.model_trainable_params > 0: 
            trainable_ratio = trainable_params/self.model_trainable_params
        else:
            trainable_ratio = 0
        trainable_params = human_readable_format(trainable_params)
        
        trainable_state = {
            "trainable_params": trainable_params,
            "total_model_params": self.model_trainable_params,
            "trainable_ratio":trainable_ratio
        }
        self.train_state.update(
            trainable_state
        )
        self.print_log(trainable_state, print_step=False)
        return trainable_state

    def train(self):
        """
        0. load pretrained model, dataset, optimizer and scheduler
        1. set up self.accelerator
        2. load previous checkpoint if resume training
        3. load training components such as data_collator, dataset and optimizer.
        4. start training.
        5. save the best model during evaluation
        5. evaluate the best model on test set
        
        Plus,
        - support resume training
        """

        # steps/epoches 
        assert self.training_args.num_train_epochs is not None, "num_train_epochs is not set"
        assert self.training_args.max_steps == -1, "max_steps is not supported yet, but got {}".format(self.training_args.max_steps)

        train_bs_per_step = self.training_args.per_device_train_batch_size * self.num_processes
        expected_num_train_step_per_epoch = len(self.train_dataset) // train_bs_per_step
        assert abs(expected_num_train_step_per_epoch -len(self.train_dataloader)) <= 1 , f"expected_num_train_step_per_epoch {expected_num_train_step_per_epoch} != len(self.train_dataloader) {len(self.train_dataloader)}"
        

    

        loss = 0
        # handle early stopping separately
        self.load_last_train_state() # load train state in case test is finished
        if self.test_eval_finished or self.train_finished:
            if self.test_eval_finished:
                self.print_log("test evaluation is already finished,  exit training...")
            if self.train_finished:
                self.print_log("training is already finished, exit training...")
            return
        self.load_last_run_multi_proc()


        self.print_log(f"Per step batch size (no grad acc): {train_bs_per_step}")
        # NOTE: only loss computation will be affected by gradient accumulation

        train_bs = self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps * self.num_processes
        self.print_log(f"Training batch size (considering grad acc): {train_bs}")

        # TODO: add expected train bs assertion or automatic adjusting

        if self.use_distributed:
            self.accelerator.log(self.training_args.to_dict())

            progress_bar = tqdm(
                # range(0, self.total_step),
                range(0, self.total_optimization_step),
                disable=not self.accelerator.is_local_main_process or self.training_args.is_cluster,
                initial = self.optimization_step,
                miniters=self.training_args.logging_steps,
            )
            if self.global_step > 0:
                self.print_log(f"Resume training from epoch {self.start_epoch}, step {self.start_step}, global_step {self.global_step}")

                self.accelerator.skip_first_batches(self.train_dataloader,  self.start_step)
                self.print_log(f"skip first {self.start_step} steps in train_dataloader", print_step=False)
        else:
            progress_bar = tqdm(
                range(0, self.total_optimization_step),
                initial = self.optimization_step,
                miniters=self.training_args.logging_steps,
                disable=self.training_args.is_cluster
            )

        self.print_log(f"***** Running training *****")
        self.print_log(f"  Num examples = {len(self.train_dataset)}")
        self.print_log(f"  Num Epochs = {self.training_args.num_train_epochs}")
        self.print_log(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size}")
        self.print_log(f"  Total train batch size (w. parallel, distributed & accumulation) = {train_bs}")
        self.print_log(f"  Gradient Accumulation steps = {self.training_args.gradient_accumulation_steps}")
        self.print_log(f"  Total optimization steps = {self.total_optimization_step}")

        # start step -> start optimization step
        # 


        self.model.train()
        logging_loss = 0

        for self.epoch in range(self.start_epoch, self.training_args.num_train_epochs):
            # it can show the processes to reach here
            self.print_log(f"------------{self.accelerator.device}: new epoch: {self.epoch} global_step: {self.global_step}")
            for self.step, inputs in enumerate(self.train_dataloader, start=self.start_step): # start count step from self.start_step 
                self.train_state.update(
                            {
                                "epoch": self.epoch,
                                "step": self.step,
                                "global_step": self.global_step,
                                "optimization_step": self.optimization_step,
                            }
                        )
                if self.use_distributed:
                    # per progress bar step is actually gradient_accumulation_steps
                    with self.accelerator.accumulate(self.model):
                        
                        try:
                            if self.label_smoother is None:
                                outputs = self.model(**inputs)
                                loss = outputs["loss"]
                            else:
                                labels = inputs.pop("labels")
                                outputs = self.model(**inputs)
                                loss = self.label_smoother(outputs, labels)
                        except RuntimeError as e:
                            if self.accelerator.is_local_main_process:
                                shutil.rmtree(self.training_args.output_dir)
                                # shutil.rmtree(self.training_args.logging_dir)
                            print(f"this expr's output dir and logging dir have been removed due to error \n {e}")
                            raise e
                        # log before backward
                        self.accelerator.backward(loss) # it does gradient acc internally
                        # under accelerator.accumulate context
                        # it steps until gradient_accumulation_steps
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        self.optimization_step += 1
                else:
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.device)
    
                    outputs = self.model(**inputs)
                    loss = outputs["loss"]
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                # self.save_and_eval(self.global_step)
                self.save_and_eval(self.optimization_step)
                if self.training_args.is_cluster:
                    import hfai
                    # cluster pre-interrupt saving
                    if hfai.distributed.get_rank() == 0 and self.accelerator.is_local_main_process: # 获取当前节点序号。在0号节点的0号进程上接收集群调度信息
                        if hfai.client.receive_suspend_command(): 
                            self.print_log(f"Received suspend command, saving model at {self.global_step} steps")
                            self.save(self.global_step)
                            self.accelerator.wait_for_everyone()
                            self.print_log(f"Model checkpoint at {self.global_step} steps is saved. Going suspend...")
                            
                            hfai.client.go_suspend()
                            


                # log each backward step (not grad acc step)
                self.global_step += 1
                # progress_bar.update(1)
                logging_loss += loss.item()
                
                # logging
                if self.global_step != 0 and self.global_step % self.training_args.logging_steps == 0:
                    try:
                        last_lr = self.scheduler.get_last_lr()[0]
                    except AssertionError:
                        last_lr = None
                        self.print_log("No latest lr found in scheduler...")
                    self.log({
                            "train/loss": logging_loss/self.training_args.logging_steps,
                            "train/lr": last_lr,
                            })
                    self.print_log(f"train/loss: {logging_loss/self.training_args.logging_steps}")
                    self.print_log(f"train/lr: {last_lr}")
                    logging_loss = 0

                # NOTE: code version updated to optimization step
                # the code below ensures backward compatibility
                if self.global_step >= self.total_step:
                    self.save_and_eval(self.global_step, force=True)
                    self.end_training()
                    return
            self.start_step = 0
            self.print_log(f"epoch {self.epoch} finished, evaluating...")
            # To be compatible with low data size, eval per epoch as well
            if not (self.training_args.dev_train or self.training_args.dev_run or self.training_args.dev_test):
                # self.save_and_eval(self.global_step, force=True)
                self.save_and_eval(self.optimization_step, force=True)
                
            self.print_log(f"epoch {self.epoch} finished, best_metric_step: {self.best_metric_step}, best_metric_val {self.best_metric_val}")
            self.print_log(f"steps per epoch: {self.global_step/(self.epoch+1)}")
        
        # log best metric val at final step for easy comparison
        self.end_training()

    def end_training(self):
        """
        end training for cluster
        """
        self.print_log(f"training is already finished, {self.global_step} steps are already done")
        self.print_log(f"best_metric_step: {self.best_metric_step}, best_metric_val {self.best_metric_val}")
        self.print_log("Ending training...")
        self.log(
        {
            "best_metric_val": self.best_metric_val,
            "best_metric_step": self.best_metric_step,
            "train_finished": True,
        }
        )
        self.accelerator.end_training()



    def log(self, d):
        """
        log to tensorboard/train state.
        but it doesn't save train state as train state could be saved to diff dirs.
        """
        self.accelerator.log(d,
                            step=self.global_step
        )
        self.train_state.update(d)
    
    def print_by_rank(self, s):
        print(f"{self.accelerator.device}: {s}")
    
    def print_log(self, s, print_step=True):
        """
        print log under different training system.
        """
        if self.optimization_step > 0 and print_step:
            s = f"global_optimization_step {self.optimization_step}/{self.total_optimization_step}  ({self.optimization_step/self.total_optimization_step}): {s}"
        if self.training_args.is_cluster:
            import hfai
            if hfai.distributed.get_rank() == 0:
                print(s)
        elif self.accelerator.is_main_process:
            logger.info(s)

    def evaluate(self, mode="eval", during_training=True):
        """
        There are two cases calling evaluate function:
        1. During training, it's called after each epoch or each eval step
        2. During test evaluation, it's called after loading the best checkpoint (dry train).
        """
        if mode == "eval":
            dataset2eval = self.eval_dataset
            dataloader2eval = self.eval_dataloader
            ni_eval_results =  self.evaluate_dataset(dataset2eval, dataloader2eval, mode=mode)
            self.log(ni_eval_results)
            # NOTE: there is a bug in accelerator that finishing evaluation
            # on evaluation dataloader
            # will set accelerator end_of_dataloader to True thus further 
            # affect gradient accumulation
            if during_training:
                self.accelerator.gradient_state.end_of_dataloader = False
            return ni_eval_results


        # Generally, in test mode, trainer.model is pretrained model and we load best checkpoint model.
        # 1. find the best checkpoint folder
        elif mode in ["test", "traditional_test"] and self.training_args.load_best_checkpoint:
            torch.cuda.empty_cache()
            # NOTE: test evaluation is done, finish
            if self.test_eval_finished and mode == "test":
                self.print_log("test evaluation is done, finish...")
                return
            if self.traditional_test_eval_finished and mode == "traditional_test":
                self.print_log("traditional test evaluation is done, finish...")
                exit()
            best_cp_dir = None
            try:
                # during test mode, self.model is pretrained model. after loading state, it's the best checkpoint model
                best_cp_dir = get_latest_checkpoint(os.path.join(self.training_args.output_dir, "best_checkpoint"))
            except FileNotFoundError as e:
                print("During test evaluation, found the error that will be raised later...")
                print(f"Removing the experiment folder {self.training_args.output_dir} ...")
                if os.path.exists(self.training_args.output_dir) and self.accelerator.is_local_main_process:
                    # shutil.rmtree(self.training_args.output_dir)
                    # print("output folder removed, please rerun the script to restart training")
                    print(f"output folder not found: {self.training_args.output_dir}")
                if os.path.exists(self.training_args.logging_dir) and self.accelerator.is_local_main_process:
                    # shutil.rmtree(self.training_args.logging_dir)
                    print("logging folder removed, please rerun the script to restart training")
                raise e

            self.print_by_rank(f"load from existing state: {best_cp_dir}")
            assert best_cp_dir is not None, "It's expected to have dir for self.accelerator to load state"

            # 2. verify complete random states. If there is missing random states, remove the best checkpoint folder. (accelerator will hang with incomplete random states).
            if self.training_args.is_cluster and not verify_complete_random_states(best_cp_dir):
                shutil.rmtree(best_cp_dir)
                check_all_checkpoints_and_remove(self.training_args.output_dir)
                exit(f"Not found complete random states in {best_cp_dir} remove the best checkpoint folder: {best_cp_dir}, exiting...")
            
            # 3. load best checkpoint for test evaluation
            # since we have loaded pretrained model in gpu and want to load trained model weights into the model within constrained gpu memory
            # 3.1 first move loaded pretrained model to cpu
            # 3.2 remove optimizer and scheduler as we no longer need training
            # 3.3 load trained model weights into the model
            # 3.4 we move model back to gpu
            if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
                self.model = self.model.to("cpu")
                del self.optimizer
                del self.scheduler
                self.accelerator._optimizers = []
                self.accelerator._schedulers = []
                torch.cuda.empty_cache()
                self.accelerator.wait_for_everyone()
                # load state will load optimizer and scheduler so we load everything into cpu
                # and then move model back to gpu
                self.accelerator.load_state(best_cp_dir)
                self.train_state.load_from_json(best_cp_dir)
                self.accelerator.wait_for_everyone()
                self.model = self.model.to(self.accelerator.device)
            else:
                self.accelerator.load_state(best_cp_dir)
                self.train_state.load_from_json(best_cp_dir)
            if not hasattr(self.accelerator, "trackers"):
                self.accelerator.init_trackers(
                        self.training_args.run_name,
                        config=self.train_state.state_dict,
                        init_kwargs={"tensorboard": {"flush_secs": 60}},
                    )
                if during_training:
                    print("warning: tracker is expected to exist during training")
        

            # free memory in test mode
            if mode == "test":
                dataset2eval = self.test_dataset
                dataloader2eval = self.test_dataloader
            elif mode == "traditional_test":
                dataset2eval = self.traditional_test_dataset
                dataloader2eval = self.traditional_test_dataloader
            ni_eval_results =  self.evaluate_dataset(dataset2eval, dataloader2eval, mode=mode)
            self.print_log("Finished test dataset evaluation...")
            return ni_eval_results
        else:
            raise NotImplementedError("Not implemented for evaluation mode other than eval, test and traditional_test")



    def evaluate_dataset(self, dataset2eval, dataloader2eval, mode="eval"):
        """
        eval mode: evaluate use loaded current model
        test mode: evaluate best model loaded from output_dir
        
        parallel generation, gather results on all processes, compute metrics on main process, return results on main process and None on other processes.

        log automatically on main process, return True if it outperform previous best checkpoint, False otherwise.
        """
        self.model.eval()
        model = accelerate.utils.extract_model_from_parallel(self.model)
        if mode == "eval":
            dataset2eval = self.eval_dataset
            dataloader2eval = self.eval_dataloader
        elif mode in ["test", "traditional_test"]:
            torch.cuda.empty_cache()
            # NOTE: test evaluation is done, finish
            if self.test_eval_finished and mode == "test":
                self.print_log("test evaluation is done, finish...")
                return
            if self.traditional_test_eval_finished and mode == "traditional_test":
                self.print_log("traditional test evaluation is done, finish...")
                exit()
            # free memory in test mode 
            if mode == "test":
                dataset2eval = self.test_dataset
                dataloader2eval = self.test_dataloader
            elif mode == "traditional_test":
                dataset2eval = self.traditional_test_dataset
                dataloader2eval = self.traditional_test_dataloader


        input_host = []
        output_host = []
        label_host = []
        from tqdm import tqdm
        self.print_log(f"***** Running evaluation {mode} *****")
        # wait for everyone to finish prepare dataset
        with torch.no_grad():
            progress_bar = tqdm(dataloader2eval,
                                miniters=0 if not self.training_args.is_cluster else 500,
                                disable=self.training_args.is_cluster)
            for inputs in progress_bar:
                
                if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.accelerator.device)
                labels = inputs.pop("labels")
                # if distrubted data parallel object 

                if self.model_args.tuning_mode in ["lora_peft", "prompt_tuning", "adapter_peft", "prefix_tuning"]: # temp PEFT lora implementation
                    input_ids = inputs.pop("input_ids")
                    attention_mask = inputs.pop("attention_mask")
                    # generation_inputs is removed from later package
                    # adapter_peft generate requires generation_inputs
                    outputs = model.generate(generation_inputs = input_ids, attention_mask=attention_mask, **inputs,
                                        max_new_tokens = self.data_args.max_target_length,
                                        synced_gpus = True if self.use_distributed else False,
                                        pad_token_id=self.tokenizer.eos_token_id if self.model_args.model_arch == "decoder" else self.tokenizer.pad_token_id,
                                        # synced_gpus = False,
                                        # pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = model.generate(**inputs,
                                            max_new_tokens = self.data_args.max_target_length,
                                            synced_gpus = True if self.use_distributed else False,
                                            # synced_gpus = False,
                                            pad_token_id=self.tokenizer.eos_token_id if self.model_args.model_arch == "decoder" else self.tokenizer.pad_token_id,
                                            # synced_gpus = False,
                                            # pad_token_id=self.tokenizer.pad_token_id,
                    )
                outputs = self.accelerator.pad_across_processes(outputs, pad_index=self.tokenizer.pad_token_id, dim=1)
                outputs = self.accelerator.gather(outputs)

                def remove_special_token(t):
                    t = t.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
                    return t

                outputs_text =  self.tokenizer.batch_decode(outputs)
                outputs_text = [remove_special_token(t) for t in outputs_text]
                output_host += outputs_text
                
                # labels[labels == self.tokenizer.mask_token_id] = self.tokenizer.pad_token_id
                # labels[labels == -100] = self.tokenizer.pad_token_id
                
                label_host += self.tokenizer.batch_decode(labels)

                cnt = len(label_host)
                if cnt % 100 == 0:
                    self.print_log(f"evaluated {cnt} {mode} data")
                # if cnt < 100:
                #     # check if there are duplicated labels
                #     # if there are, it means dataloader is not prepared for multi-gpu setup
                #     print(f"{self.accelerator.device}: {label_host}")
                self.accelerator.wait_for_everyone()

        results = self.compute_metrics(
                (output_host, label_host),
                dataset2eval
            )
        results_with_mode = {}
        for k in results:
            results_with_mode[f"{mode}/{k}"] = results[k]
        self.model.train()

        self.log(results_with_mode)
        if mode == "test":
            print("update train state test_eval_finished to True")
            self.train_state.update({"test_eval_finished": True})
        elif mode == "traditional_test":
            print("update train state traditional_test_eval_finished to True")
            self.train_state.update({"traditional_test_eval_finished": True})
        self.save_train_state(get_latest_checkpoint(self.training_args.output_dir))
        metric="rougeL"
        self.print_log(f"{mode}/{metric}: {results_with_mode[f'{mode}/{metric}']}")
        
        return results_with_mode


    def compute_metrics(self, eval_preds, eval_dataset, is_pred_logits = False, model_idx = 0, metrics = {}):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        def postprocess_text(preds, labels):
            preds = [pred.strip().lower() for pred in preds]
            labels = [[label.strip().lower()] for label in labels]
            return preds, labels
        result = metrics

        if self.data_args.dataset_name == "ni":
            save_prefix = None
            decoded_preds = preds
            
            
            references = [e["Instance"]["output"] for e in eval_dataset]
            # references = labels
            
            result = compute_metrics(predictions=decoded_preds, references=references)
            result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=eval_dataset["Task"])
            result.update(result_per_task)
            categories = ["_".join(it[0].lower().split()) for it in eval_dataset["Categories"]]

            result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories) # by category only, evaluate the category on all dev and test

            categories_split = ["_".join(it[0].lower().split()) for it in eval_dataset["Categories_split"]]
            result_per_category_split = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories_split) # by category + split, evaluate the category on dev and test separately

            result.update(result_per_category)
            result.update(result_per_category_split)
            
            
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(self.training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result
        else:
            raise NotImplementedError("compute_metrics is not implemented for dataset other than ni")


    def configure_n_load_peft_module(self):
        # model loading procedure:
        # 1. load model from model_name_or_path    (self.load_pretrained_model())
        # 2. not satisfied with peft, load model from self.model_cache and convert again. self.model = deepcopy(self.model_cache)
        task_type = TaskType.SEQ_2_SEQ_LM
        if self.model_args.tuning_mode == "prompt_tuning":
            config = PromptTuningConfig(
                task_type=task_type,
                num_virtual_tokens=self.peft_args.prompt_len,
                inference_mode=False,
            )
            self.load_peft_module(config)

        elif self.model_args.tuning_mode == "prefix_tuning":
            from peft import PrefixTuningConfig
            # prefix reparameterization enabled
            if self.peft_args.bottleneck_size > 0:
                # projection: embedding -> encoder_hidden_size -> layer hidden size
                config = PrefixTuningConfig(
                    task_type=task_type,
                    inference_mode=False,
                    num_virtual_tokens=self.peft_args.prefix_len,
                    prefix_projection = True,
                    encoder_hidden_size=self.peft_args.bottleneck_size,
                )
            else:
                config = PrefixTuningConfig(
                    task_type=task_type,
                    inference_mode=False,
                    num_virtual_tokens=self.peft_args.prefix_len,
                )

            self.load_peft_module(config)

        
        elif self.model_args.tuning_mode == "lora_adapter":
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=self.peft_args.lora_r ,
                                alpha=self.peft_args.lora_alpha,
                                attn_matrices=self.peft_args.lora_modules.split(",") if self.peft_args.lora_modules else ["q", "v"],
                                # mlp_lora=True,
                                dropout=self.peft_args.dropout_rate,
                                )
            self.load_peft_module(config)

        elif self.model_args.tuning_mode == "lora_peft":
            from peft import LoraConfig
            config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=self.peft_args.lora_r ,
                lora_alpha=self.peft_args.lora_alpha,
                lora_dropout=self.peft_args.dropout_rate,
                # lora_modules
                target_modules= self.peft_args.lora_modules.split(",") if self.peft_args.lora_modules else ["q", "v"],
            )
            self.load_peft_module(config)


        elif self.model_args.tuning_mode == "ia3":
            from transformers.adapters import IA3Config
            config = IA3Config(
                dropout=self.peft_args.dropout_rate,
            )
            self.load_peft_module(config)

        elif self.model_args.tuning_mode == "adapter_peft":
            from peft import AdapterConfig
            if "t5" in self.model_name_or_path:
                peft_config = AdapterConfig(
                    task_type = TaskType.SEQ_2_SEQ_LM,
                    adapter_size = self.peft_args.adapter_size,
                    target_modules = ["encoder.block", "decoder.block"],
                    model_config = self.model.config.to_dict(),
                    inference_mode=False
                )
            elif "opt" in self.model_name_or_path:
                peft_config = AdapterConfig(
                    task_type = TaskType.CAUSAL_LM,
                    adapter_size = self.peft_args.adapter_size,
                    target_modules =["model.decoder.layers"],
                    model_config = self.model.config.to_dict(),
                    inference_mode=False
                )
            elif "llama" in self.model_name_or_path:
                peft_config = AdapterConfig(
                    adapter_size = self.peft_args.adapter_size,
                    task_type = TaskType.CAUSAL_LM,
                    target_modules = ["model.layers"],
                    model_config = self.model.config.to_dict(),
                    inference_mode=False
                )
            else:
                raise NotImplementedError(
                    f"Adapter is not implemented for {self.self.model_name_or_path}"
                )
            self.load_peft_module(peft_config)
        
        elif self.model_args.tuning_mode in ["adapter_adapter", "compactor"]:
            cur_reduction_factor = 64 if self.peft_args.reduction_factor is  None else self.peft_args.reduction_factor
 
            from transformers.adapters import AdapterConfig, HoulsbyConfig, CompacterConfig
            # check existing adapter and remove them
            # config = AdapterConfig()
            if self.model_args.tuning_mode == "adapter_adapter":
                # config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
                config = HoulsbyConfig(adapter_size=self.peft_args.adapter_size,)
            else:
                config = CompacterConfig(reduction_factor=cur_reduction_factor,
                                            phm_dim=self.peft_args.phm_dimension )
            self.load_peft_module(config)
        elif self.model_args.tuning_mode == "parallel_adapter":
            from transformers.adapters import ParallelConfig
            config = ParallelConfig(reduction_factor= self.peft_args.reduction_factor)
            self.load_peft_module(config)
        elif self.model_args.tuning_mode == "bitfit":
            self.load_peft_module()
        elif self.model_args.tuning_mode == "fine_tuning":
            # no converting needed
            pass
        else:
            raise NotImplementedError(f"mode {self.model_args.tuning_mode} is not implemented")

    def save_and_eval(self, step, force=False):
        """
        Two step conditions to save and evaluate:
        1. optimization step matches save_steps/eval_steps
        2. global step % grad acc steps == 0. The reason is that optimization step update every gradient accumulation steps.

        force is used in end of training or after each epoch.
        """
        force = False if self.training_args.dev_run or self.training_args.dev_train or self.training_args.dev_test else force

        if force or (self.optimization_step > 0 and self.global_step % self.accelerator.gradient_accumulation_steps == 0 and self.optimization_step  % self.training_args.save_steps == 0):
            self.save(self.optimization_step)

        if force or (self.optimization_step > 0 and self.global_step % self.accelerator.gradient_accumulation_steps == 0 and self.optimization_step  % self.training_args.eval_steps == 0):
            results = self.evaluate(mode="eval", during_training=True)
            if results is None:
                return
            eval_metric_name = "eval/"+self.training_args.eval_metric
            cur_metric_val = results[eval_metric_name]
            self.print_log(f"current metric_val: {cur_metric_val}")
            if cur_metric_val > self.best_metric_val:
                self.best_metric_val = cur_metric_val
                self.best_metric_step = step
                # log before save
                self.log(
                    {
                        "best_metric_val": self.best_metric_val,
                        "best_metric_step": self.best_metric_step,
                    }
                )
                self.print_log("saving a new best checkpoint...")
                # save model and state
                self.save(step, save_best_checkpoint=True)
                self.print_log(f"new best_metric_step...")
            self.print_log(f"best_metric_val: {self.best_metric_val}, best_metric_step: {self.best_metric_step}")



    def save(self, global_step, remove_old_cp = True, save_best_checkpoint = False):
        """
        cp_folder_path could be best checkpoint folder path or checkpoint folder path.
        
        if save_best_checkpoint is True, then save to best checkpoint folder path.
        """
        self.accelerator.wait_for_everyone()
        start_time = time.time()
        self.log(
            {"global_step": global_step,
             "start_epoch": self.epoch,
             "start_step": self.step,}
        )
        cp_folder_name = f"checkpoint-{global_step}"
        # remove old checkpoint if eval and has a better checkpoint
        if save_best_checkpoint:
            checkpoint_dir_path = os.path.join(self.training_args.output_dir, "best_checkpoint")
            checkpoint_folder_path_to_save = os.path.join(checkpoint_dir_path, cp_folder_name)
        else:
            # in case we don't want checkpoints in separate sub-folders
            # checkpoint_dir_path = self.training_args.output_dir
            # checkpoint_folder_path_to_save = self.training_args.output_dir
            checkpoint_dir_path = os.path.join(self.training_args.output_dir)
            checkpoint_folder_path_to_save = os.path.join(checkpoint_dir_path, cp_folder_name)
        self.accelerator.wait_for_everyone()
        if self.training_args.is_cluster:
            save_counter = 0
            while not verify_complete_random_states(checkpoint_folder_path_to_save):
                self.accelerator.wait_for_everyone()
                self.accelerator.save_state(checkpoint_folder_path_to_save)
                # make sure two savings are async
                self.accelerator.wait_for_everyone()
                save_counter +=1
                if save_counter > 10:
                    raise ValueError(f"{self.accelerator.device}: tried save accelerator state to {checkpoint_folder_path_to_save} after {save_counter} times but failed. please check the folder state {checkpoint_folder_path_to_save}")

            self.print_by_rank(f"save accelerator state to {checkpoint_folder_path_to_save} after {save_counter} times")
        else:
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(checkpoint_folder_path_to_save)
            # make sure two savings are async
            self.accelerator.wait_for_everyone()
        # save new train state no matter if it's best checkpoint
        # save train state earlier than model
        if self.accelerator.is_main_process:
            self.train_state.save_to_json(checkpoint_folder_path_to_save)
            if remove_old_cp:
                remove_old_checkpoints(checkpoint_dir_path, self.training_args.checkpoint_save_total_limit)
        self.print_log(f"save accelerator state to {checkpoint_folder_path_to_save}")
        # save sharded checkpoint into another model
        sharded_model_path = os.path.join(checkpoint_folder_path_to_save, "save_pretrained")
        if save_best_checkpoint:
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(sharded_model_path)
            if self.model_args.tuning_mode == "fine_tuning":
                unwrapped_model.save_pretrained(sharded_model_path,
                                                is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save, state_dict=self.accelerator.get_state_dict(unwrapped_model))
            else:
                # PEFT
                if self.accelerator.is_main_process:
                    state_dict = self.accelerator.get_state_dict(self.model)
                    unwrapped_model.save_pretrained(sharded_model_path, state_dict=state_dict)
                    # adapter pacakge
                    if hasattr(unwrapped_model, "save_all_adapters"):
                        unwrapped_model.save_all_adapters(sharded_model_path)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("Model saving time in seconds:", time.time() - start_time)
        if self.accelerator.is_main_process and remove_old_cp:
            remove_old_checkpoints(checkpoint_dir_path, self.training_args.checkpoint_save_total_limit)

    def load_last_run_multi_proc(self):
        # load last state
        if self.accelerator.is_local_main_process:
            # first make sure load latest uncorrupted state
            self.load_last_accelerator_state()
            # second load train state for the uncorrupted state
            self.load_last_train_state()
            
        
        # train state and model checkpoint are loaded in main process
        latest_cp = get_latest_checkpoint(self.training_args.output_dir)
        

        if latest_cp and not self.accelerator.is_local_main_process:
            # OOM issue: only load state in main process is needed?
            # self.print_by_rank(f"loading last accelerator state")
            # self.accelerator.load_state(latest_cp)
            self.print_by_rank(f"loading last train state")
            self.load_last_train_state()
            # deepspeed requires load state in non-main processes as well
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                self.print_by_rank(f"loading last accelerator state")
                self.accelerator.load_state(latest_cp)

        self.accelerator.wait_for_everyone()
        self.accelerator.init_trackers(
                self.training_args.run_name,
                config=self.train_state.state_dict,
                init_kwargs={"tensorboard": {"flush_secs": 60}},
            )

        # some extra post-loading check
        if self.global_step >= self.total_step:
            self.print_log(f"training is already finished, {self.start_epoch} epochs and {self.start_step} steps are already done")
            self.print_log("Only train state is loaded...")
            self.train_finished = True
            self.train_state.update({"train_finished": True})
            self.save_train_state(latest_cp)
        
        # some helpful info for debugging
        self.print_log(f"start_epoch: {self.start_epoch}, start_step: {self.start_step}, global_step: {self.global_step}, total_step: {self.total_step}")
        self.print_log(f"best_metric_step: {self.best_metric_step}, best_metric_val: {self.best_metric_val}")


    def load_last_train_state(self):
        """
        load training state from checkpoint folder.
        It only tries to load once and if failed, then remove the checkpoint folder and raise error.
        """
        latest_cp = get_latest_checkpoint(self.training_args.output_dir)
        
        if latest_cp:
            try:
                self.train_state.load_from_json(latest_cp)
                self.start_epoch = self.train_state.get("epoch")
                
                self.start_step = self.train_state.get("step")
                self.optimization_step = self.train_state.get("optimization_step")
                self.global_step = int(latest_cp.split("-")[-1])
                # train/test status
                self.train_finished = self.train_state.get("train_finished")
                self.test_eval_finished = self.train_state.get("test_eval_finished")
                self.traditional_test_eval_finished = self.train_state.get("traditional_test_eval_finished")
                self.best_metric_step = self.train_state.get("best_metric_step")
                self.best_metric_val = self.train_state.get("best_metric_val")
                if self.test_eval_finished:
                    self.print_log("test evaluation is already finished,  exit...")
                    return
            except Exception as e:
                # posssibly train state is not saved correctly
                if self.accelerator.is_local_main_process:
                    print(f"train state is not saved correctly, remove {latest_cp} and restart training")
                    shutil.rmtree(latest_cp)
                raise e
        else:
            self.print_log(f"latest checkpoint not found in {self.training_args.output_dir}")


    def load_last_accelerator_state(self):
        """
        load previous training state and model checkpoint from checkpoint folder, if failed, then init a new run.
        It tries to run and probably fails and delete checkpoint in a single process. So it's expected to run this function in a single process and then loads other states in other processes.
        - if a checkpoint folder doesn't contain all random states, remove it and reload the previous checkpoint.
        - Note: variables such as self.start_step is loaded in first process only.
        """
        loaded = False
        trial_times = 0 # record how many times we have tried to load from existing state, if there are more than 5 times of failure, then remove the output dir
        while not loaded:
            if os.path.exists(self.training_args.output_dir):
                self.print_log("load from existing state", print_step=False)
                latest_cp = get_latest_checkpoint(self.training_args.output_dir)
                # if latest cp is not None and is in cluster environment, then check if it has complete random states
                if self.training_args.is_cluster and latest_cp is not None and  not verify_complete_random_states(latest_cp):
                    check_all_checkpoints_and_remove(self.training_args.output_dir)
                    latest_cp = None

                # try to load the latest checkpoint
                try:
                    # check it earlier
                    if self.training_args.is_cluster and not verify_complete_random_states(latest_cp):
                        raise ValueError(f"Not found complete random states in {latest_cp} for testing.")
                    self.print_by_rank(f"loading from accelerator state")
                    self.accelerator.load_state(latest_cp)
                    # time.sleep(60)
                    loaded = True
                except Exception as e:
                    trial_times += 1
                    error_message = str(e)
                    
                    # it can be state dict file corrupted or model checkpoint corrupted
                    # in any case, remove the checkpoint folder and reload the previous one
                    # remove the latest checkpoint
                    # output_dir might already has been removed
                    if trial_times > 10:
                        print("Tried to load more than 10 times, remove the output dir and re-init the tracker")
                        shutil.rmtree(self.training_args.output_dir)
                        self.start_epoch = 0
                        self.start_step = 0
                        self.global_step = 0
                    elif "state_dict" in error_message:
                        print("state dict file corrupted, remove the latest checkpoint and reload the previous one")
                        shutil.rmtree(latest_cp)
                    elif "random states" in error_message:
                        print(f"Not found complete random states in {latest_cp} for testing. Removing the folder...")
                        shutil.rmtree(latest_cp)
                    elif latest_cp is not None and self.accelerator.is_local_main_process:
                        self.print_log(f"remove the latest checkpoint {latest_cp} due to\n\n {e}")
                        self.print_log(f"latest checkpoint is not None but has the loading issue")
                        self.print_log(f"output dir is {self.training_args.output_dir}, consider remove it")
                        shutil.rmtree(latest_cp)
                        # exit(f"latest checkpoint is not None and has the loading issue {e}. So it's removed.")
                    elif latest_cp is None and self.accelerator.is_local_main_process:
                        if os.path.exists(self.training_args.output_dir):
                            if "best_checkpoint" in os.listdir(self.training_args.output_dir):
                                # if no latest checkpoint but has best checkpoint, then copy best checkpoint to output dir
                                best_cp_dir = os.path.join(self.training_args.output_dir, "best_checkpoint")
                                
                                best_cp = get_latest_checkpoint(best_cp_dir)
                                if best_cp is None: # not latest checkpoint and no best checkpoint to move neither
                                    shutil.rmtree(self.training_args.output_dir)
                                    print(f"latest checkpoint is None and no best checkpoint to move neither, remove the folder {self.training_args.output_dir} and restart the experiment")
                                    self.start_epoch = 0
                                    self.start_step = 0
                                    self.global_step = 0
                                else:
                                    print(f"latest checkpoint is None, copy best checkpoint {best_cp} to output dir {self.training_args.output_dir}")
                                    cp_name = os.path.basename(best_cp)
                                    shutil.copytree(best_cp, os.path.join(self.training_args.output_dir, cp_name))
                            else:
                                # if no latest checkpoint and no best checkpoint, then remove the folder
                                shutil.rmtree(self.training_args.output_dir)
                                logger.warn(f"no checkpoint found, remove the folder {self.training_args.output_dir} and re-init the tracker")
                                self.start_epoch = 0
                                self.start_step = 0
                                self.global_step = 0
                    else:
                        print("Not sure the situation raise the error instead")
                        print(f"latest checkpoint is {latest_cp}")
                        print(f"output dir is {self.training_args.output_dir}, consider remove it")
                        raise e
                        # raise e
                    # else:
                    #     # TODO: progressive delete the folder if latest cp is not found
                    #     self.print_log(f"latest checkpoint is None, no removing")
                    #     self.print_log(f"output dir is {self.training_args.output_dir}, consider remove it")
                    #     exit(f"latest checkpoint is None, no removing")
                    #     # raise e


                    # NOTE: it should be if there is no checkpoint then remove 
                    # for safer remove, we remove the whole folder when no file at all in the folder
                    # possibly remove the output dir if it's empty
                    # if os.path.exists(self.training_args.output_dir) and not not os.listdir(self.training_args.output_dir):
                    #     shutil.rmtree(self.training_args.output_dir)
                    #     logger.warn(f"no checkpoint found, remove the folder {self.training_args.output_dir} and re-init the tracker")
                    # still not loaded
                    continue
            else:
                self.print_log(f"no previous run found, create a new one at {self.training_args.output_dir}")
                if self.accelerator.is_main_process and not os.path.exists(self.training_args.output_dir):
                    os.makedirs(self.training_args.output_dir)
                self.accelerator.init_trackers(
                        self.training_args.run_name,
                        config=self.train_state.state_dict,
                        init_kwargs={"tensorboard": {"flush_secs": 60}},
                )
                loaded = True
        return loaded

    def save_train_state(self, path):
        if self.accelerator.is_main_process:
            self.train_state.save_to_json(path)