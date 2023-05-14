from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Model, T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModelForSequenceClassification, Seq2SeqAdapterTrainer, TrainerState

from datasets import load_dataset, load_metric, concatenate_datasets
import numpy as np
from torch.nn import CrossEntropyLoss
import torch
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    AdamW,
    Adafactor,
    get_scheduler,
    EarlyStoppingCallback
)
from transformers import AutoModelForSeq2SeqLM, AutoAdapterModel

from torch.utils.data import DataLoader
import numpy as np
import os
import time
from functools import partial
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
    create_optimizer,
    get_constant_schedule
)
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
import transformers
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptTuningConfig,PrefixTuningConfig
from util.ni_dataset_collator import DataCollatorForNI
from copy import deepcopy
import datetime
from peft import PeftModelForSeq2SeqLM
from utils import get_latest_checkpoint, remove_old_checkpoints, remove_files_and_folders_other_than
import json
import math
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import shutil
from util.compute_metrics import compute_metrics, compute_grouped_metrics
from accelerate.utils import DistributedType
import time
from transformers.trainer_pt_utils import LabelSmoother
# modules use two pacakges 
ADAPTER_TRANSFORMERS_MODULES=["adapter", "compactor", "prefix_tuning", "ia3",  "parallel_adapter", "lora_adapter"]
PEFT_MODULES=["prompt_tuning", "lora_peft", "bitfit"]
CAUSAL_LM=["gpt", "llama", "opt"]


BEST_CP_FOLDER_NAME="best_checkpoint"
LATEST_CP_FOLDER_NAME="latest_checkpoint"
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

import logging
from accelerate.logging import get_logger
import accelerate


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
            "trainable_params": 0,
            "total_model_params": 0,
            "trainable_ratio": 0,
        }
        self.file_name = "training_state.json"


    def get(self, k):
        if k in self.state_dict:
            return self.state_dict[k]
        else:
            if hasattr(self, k):
                return getattr(self,k)
            else:
                raise ValueError(
                    f"{k} cannot be found in train state"
                )
            

    def update(self, dict):
        self.state_dict.update(dict)
        # for k, v in dict.items():
        #     if hasattr(self, k):
        #         setattr(self, k, v)
        #     else:
        #         logger.warning(f"key {k} is not in TrainingState")

    def to_dict(self):
        return dict([(k, v) for k, v in self.__dict__.items() if not k.startswith("_")])


    def save_to_json(self, cp_path):
        file_path = os.path.join(cp_path, self.file_name)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f)


    def load_from_json(self, cp_path):
        file_path = os.path.join(cp_path, self.file_name)
        with open(file_path, "r") as f:
            data = json.load(f)
        # if self.accelerator.is_main_process:
        #     import pdb; pdb.set_trace()
        #     print('check keys')
        #     # it updates d like  {"state_dict": ___}
        #     # so eval_save doesn't log the step
        #     # log is implemented wrong
        #     # it's right in file
        #     # so it's wrong in loading
        # self.accelerator.wait_for_everyone()
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
        # self.use_distributed = use_accelerator replaced
        
        # self.arguments = arguments
        self.model_name_or_path = self.model_args.model_name_or_path
        self.potential_model_path =  os.path.join(
            self.training_args.saved_pretrained_model_path,
            self.model_name_or_path
        )
        self.verbalizers = self.get_dataset_verbalizers(self.data_args.dataset_name if self.data_args.dataset_name != "super_glue" else self.data_args.dataset_config_name)
        
        self.model = None
        self.model_trainable_params = None
        self.recover_from = None
        
        # init
        self.best_metric_val = -1
        self.best_metric_step = -1
        self.warmup_steps = -1
        
        self.start_epoch = 0
        self.start_step = 0
        self.global_step = 0
        self.test_eval_finished = False
        
        if self.model_args.model_arch != "decoder":
            self.model_lm_head_weight = AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path).lm_head.weight
        
        if training_args.do_search_hyperparams:
            return
        
        
        self.accelerator = Accelerator(
                log_with="tensorboard",
                logging_dir=self.training_args.logging_dir,
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

        self.label_smoother = LabelSmoother(epsilon=self.training_args.label_smoothing_factor) if self.training_args.label_smoothing_factor > 0 else None
        assert self.label_smoother is None
        # model needs to be loaded on all machines
        self.load_model_n_peft_module()
        
        
        # TODO: accelerator needs to load model and peft module first anyway
        # is there anyway to not load the original model? since if model is large then it will take a lot of time
        assert self.model is not None, "model should loaded"

        # also resize embedding here
        self.load_tokenzier()
        assert self.tokenizer is not None, "tokenizer should loaded"
        # resize token embedding will set requires_grad back to True
        # we need to set it back to False
        if self.model_args.tuning_mode != "fine_tuning":
            if "gpt2" in model_args.model_name_or_path:
                self.model.transformer.wte.weight.requires_grad = False
                self.model.transformer.wpe.weight.requires_grad = False
                self.model.lm_head.weight.requires_grad = False
            elif "llama" in model_args.model_name_or_path:
                self.model.lm_head.weight.requires_grad = False
                self.model.model.embed_tokens.weight.requires_grad = False
            elif "opt" in model_args.model_name_or_path:
                self.model.model.decoder.embed_tokens.weight.requires_grad = False
                self.model.lm_head.weight.requires_grad = False
                
        trainable_params_percent = self.check_trainable_parameters()

        self.total_step = -1
        self.build_dataloader()
        assert self.total_step > 0
        assert self.warmup_steps > 0
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
            self.eval_dataloader, self.test_dataloader = self.accelerator.prepare(self.eval_dataloader, self.test_dataloader)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)

        """
        Model is loaded, now we need to set up the trainer
        1. prepare peft model
        2. set up trainer
        """
        task_type = TaskType.SEQ_2_SEQ_LM
        init_from_text = False
        if self.data_args.dataset_name == "sst2" and self.model_args.model_arch == "encoder":
            task_type = TaskType.SEQ_CLS
            prompt_tuning_init = None # NOTE: it decides whether prompt init is used
            prompt_tuning_init_text = None
            init_text_tokenizer_name_or_path = None
            prompt_tuning_init = "TEXT"
            print("task type is seq_cls")
        prompt_tuning_init_text=" ".join(self.verbalizers)
        init_text_tokenizer_name_or_path = self.model_name_or_path

        if self.peft_args.trainable_params_percentage and not self.model_args.tuning_mode in ["compactor","fine_tuning"]:
            # not check compactor
            assert abs(self.peft_args.trainable_params_percentage - cur_trainable_params_percentage) < 0.002, f"trainable_params_percentage {self.peft_args.trainable_params_percentage} is not matched with cur_trainable_params_percentage {cur_trainable_params_percentage}"



    def load_model_n_peft_module(self):
        self.model = self.load_pretrained_model()
        self.model_trainable_params = sum(p.numel() for p in self.model.parameters())
        assert self.model_trainable_params > 0, "Model has no trainable parameters"
        self.configure_n_load_peft_module() # always load model from scratch for accelerate



    def load_optimizer_n_scheduler(self):
        

        # lr should be scaled linearly with the number of processes
        # if self.use_distributed or self.distributed_type == DistributedType.DEEPSPEED:
        #     # self.training_args.learning_rate = self.training_args.learning_rate * self.num_processes
        #     # lr  = self.training_args.learning_rate / self.num_processes
        #     self.print_log(f"Scale up learning rate from to {self.training_args.learning_rate} due to number of processes({self.num_processes})")

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
                num_training_steps=self.total_step,
                num_warmup_steps=self.warmup_steps,
                total_num_steps=self.total_step
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
        
        # if self.tokenizer.pad_token is None:
            # assert "gpt2" in self.model_name_or_path or "llama" in self.model_name_or_path, "Only gpt2 or llama is expected not having pad tokens for now"
            # add <sep> token
        
            
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
                
                # if os.path.exists(self.potential_model_path):
                #     model =AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path, config = config)
                # else:
                #     model =AutoModelForSeq2SeqLM.from_pretrained(self.potential_model_path, cache_dir=self.training_args.cache_dir, config = config)
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
            if self.model_args.tuning_mode in ["fine_tuning", "prompt_tuning"] or self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
                model = LlamaForCausalLM.from_pretrained(self.potential_model_path, config = config)
            else:
                raise NotImplementedError("Tuning mode not supported: " + self.model_args.tuning_mode)
        elif "roberta" in self.model_name_or_path:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        elif "gpt2" in self.model_name_or_path or "bloom" in self.model_name_or_path or "opt" in self.model_name_or_path:
            from transformers import AutoModelForCausalLM
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

    def preprocess(self, examples, class_ids = [0,1], evaluation=False):
        prefix_prompt = ""
        if self.peft_args.num_soft_tokens ==0:
            assert prefix_prompt == ""
        if self.model_args.model_arch == "encoder":
            add_prompt_for_gen = False
        else:
            add_prompt_for_gen = True
        if self.data_args.dataset_name == "sst2":
            prompt_for_gen = "Sentiment:"
            inputs =["Sentence: " + sent for sent, label_id in zip(examples["sentence"], examples["label"]) if label_id in class_ids]
            
            # verbalize the sentiment id to tokens
            # it's not used for t5 evaluation though (label id is used instead)
            verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
        elif self.data_args.dataset_name == "yelp_review_full":
            prompt_for_gen = "give the review score from 1-5 stars:"
            inputs =["Sentence: " + sent for sent, label_id in zip(examples["text"], examples["label"]) if label_id in class_ids]
            verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            
        elif self.data_args.dataset_name == "super_glue":
            prompt_for_gen = "Answer:"
            if self.data_args.dataset_config_name in['axb', 'axg'] :
                raise NotImplementedError("axb and axg are not implemented yet")
                inputs = ["Premise: " + premise + " Hypothesis: " + hypothesis + "Given the premise, is the hypothesis correct? Answer: " for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])]
                verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            elif self.data_args.dataset_config_name in ["boolq", "multirc"]:
                if self.data_args.dataset_config_name == "boolq":
                    inputs = ["Question: " + question + "Passage: " + passage for question, passage, label_id in zip(examples["question"], examples["passage"], examples["label"]) if label_id in class_ids]
                elif self.data_args.dataset_config_name == "multirc":
                    inputs = ["Question: " + question + "Paragraph: " + paragraph  for question, paragraph, label_id in zip(examples["question"], examples["paragraph"], examples["label"]) if label_id in class_ids]
                    # inputs = ["" for question, paragraph in zip(examples["question"], examples["paragraph"])]
                # inputs = ["Passage: " + passage + " Question: " + question for question, passage in zip(examples["question"], examples["passage"])]
                verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            elif self.data_args.dataset_config_name in ["wic"]:
                prompt_for_gen = "" # NOTE: wic has a special format
                inputs = ["Sentence1: " + sentence1 + " Sentence2: " + sentence2 + f" Question: does the word '{word}' have the same meaning in the two sentences? Answer: " for sentence1, sentence2, word, label_id in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["label"])  if label_id in class_ids]
                verbal_targets = [self.verbalizers[l] for l, label_id in zip(examples["label"], examples["label"])  if label_id in class_ids]
        elif self.data_args.dataset_name in ["trec"]:
            prompt_for_gen = " What's the type of the question? "
            inputs = ["Question: " + t for t, label_id in zip(examples["text"],examples["coarse_label"]) if label_id in class_ids]
            verbal_targets = [self.verbalizers[l] for l, label_id in zip(examples["coarse_label"], examples["coarse_label"])  if label_id in class_ids]
        else:
            
            
            raise NotImplementedError("Dataset not supported: " + self.data_args.dataset_name)
        if add_prompt_for_gen:
                inputs = [inp + " " + prompt_for_gen for inp in inputs]
        formatted_inputs, verbal_targets =\
                self._format_prompts(prefix_prompt,
                                    inputs,
                                    # [self.prefix_prompt]*len(inputs),
                                    verbal_targets,
        )
        print("Sample input: ", formatted_inputs[0])

        
        model_inputs = {}
        tokenizer = self.tokenizer

        model_inputs = tokenizer(
                formatted_inputs,
                max_length=self.data_args.max_source_length,
                padding=self.padding,
                truncation=True,
                return_tensors='pt'
        )
        # build label input ids
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                verbal_targets,
                return_tensors='pt',
                padding="max_length", 
                max_length=self.data_args.max_target_length,
                truncation=True,
            )
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            pad_token_id = tokenizer.pad_token_id
            labels["input_ids"][labels["input_ids"]==pad_token_id] = -100
            
            
            # if encoder only model
            if self.model_args.model_arch == "encoder":
                # for SequenceClassificationModel
                # model_inputs["label"] = [l for l in examples["label"] if l in class_ids]
                model_inputs["labels"] = [l for l in examples["label"] if l in class_ids]
            else:
                # model_inputs["label"] = labels["input_ids"]
                model_inputs["labels"] = labels["input_ids"]

        if evaluation:
            label_name = "label" if self.data_args.dataset_name not in ["trec"] else "coarse_label"
            label_class_ids = [l for l in examples[label_name] if l in class_ids]
            model_inputs["class_ids"] = torch.tensor(label_class_ids)
            # model_inputs = add_idx_to_inputs_keys(model_inputs, model_idx)

        return model_inputs


    def build_dataloader(self):
        self.load_data_collator()
        self.load_dataset()

        min_eval_data_size_per_process = self.num_processes * self.training_args.per_device_eval_batch_size
        min_test_data_size_per_process = self.num_processes * self.training_args.per_device_test_batch_size
        if self.use_distributed:
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
            
            assert len(self.eval_dataset) % min_eval_data_size_per_process == 0, f"eval dataset size {len(self.eval_dataset)} must be divisible by number of processes*eval_batch_size {min_eval_data_size_per_process}"
            assert len(self.test_dataset) % min_test_data_size_per_process == 0, f"test dataset size {len(self.test_dataset)} must be divisible by number of processes*test_batch_size {min_test_data_size_per_process}"
        self.load_dataloader()


        train_bs_per_step = self.training_args.per_device_train_batch_size * self.num_processes
        # with gradient accumulation, per gradient update step is actually multiple steps
        self.total_step = self.training_args.num_train_epochs * len(self.train_dataset) // train_bs_per_step
        self.warmup_steps = self.total_step * self.training_args.warmup_ratio
        self.print_log(f"total_step: {self.total_step}, warmup_steps: {self.warmup_steps}", print_step=False)
        
        self.num_training_steps_for_scheduler = self.total_step * self.accelerator.num_processes
        self.warmup_steps_for_scheduler = self.num_training_steps_for_scheduler * self.training_args.warmup_ratio
        


    
    
    def load_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=self.data_collator
        )
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
            )

            if self.training_args.dev_run:
                raw_datasets["train"] = raw_datasets["train"].select(range(self.training_args.dev_run_data_size))
                raw_datasets["validation"] = raw_datasets["validation"].select(range(self.training_args.dev_run_data_size))
                raw_datasets["test"] = raw_datasets["test"].select(range(self.training_args.dev_run_data_size))

            elif self.training_args.dev_train:
                raw_datasets["train"] =  raw_datasets["train"].select(range(self.training_args.dev_train_data_size))
                raw_datasets["validation"] = raw_datasets["train"]
                raw_datasets["test"] = raw_datasets["test"].select(range(self.training_args.dev_train_data_size))
            elif self.training_args.dev_test:
                # test compute metrics are same for validation and test as
                # test evaluation load model from checkpoint and run on test dataset
                raw_datasets["train"] =  raw_datasets["train"].select(range(self.training_args.dev_test_data_size))
                raw_datasets["validation"] = raw_datasets["train"]
                raw_datasets["test"] = raw_datasets["train"]

            self.train_dataset = raw_datasets["train"]
            self.eval_dataset = raw_datasets["validation"]
            self.test_dataset = raw_datasets["test"]
        else:
            raise NotImplementedError("New implementation no train,valid,test.   Dataset not supported: " + self.data_args.dataset_name)
            raw_datasets = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name)
            column_names = raw_datasets["train"].column_names

            if train:
                self.train_dataset = raw_datasets["train"].map(
                    self.preprocess,
                    batched=True,
                    remove_columns= column_names,
                    num_proc=1,
                    # load_from_cache_file=self.arguments.dataset_cache,
                    fn_kwargs = {"evaluation": False},
                    # fn_kwargs = {"evaluation": True},
                    desc="Running tokenizer on train dataset",
                )
                
                
                # sample_input = np.array(self.train_dataset[0]["input_ids_0"])
                sample_input = np.array(self.train_dataset[0]["input_ids"])
                # sample_label = np.array(self.train_dataset[0]["labels_0"])
                sample_label = np.array(self.train_dataset[0]["labels"])
                sample_input[sample_input==-100] = 0
                sample_label[sample_label==-100] = 0

                
                print("train dataset input sample", sample_input, "\n", self.tokenizer.decode(sample_input))
                print("train dataset label sample", sample_label,"\n", self.tokenizer.decode(sample_label))
                self.train_dataset.set_format(type="torch")

            if valid:
                if self.data_args.dataset_name in ["yelp_review_full", "ag_news", "trec"]:
                    valid_split_name = "test"
                else:
                    valid_split_name = "validation"
                
                if self.training_args.dev:

                    true_val_dataset = raw_datasets[valid_split_name].map(
                        self.preprocess,
                        batched=True,
                        remove_columns=column_names,
                        num_proc=1,
                        # load_from_cache_file=self.arguments.dataset_cache,
                        # fn_kwargs = {"evaluation": False},  # hf internal validation
                        fn_kwargs = {"evaluation": True, "class_ids":[0]},
                        desc="Running tokenizer on validation dataset",
                    )

                    false_val_dataset = raw_datasets[valid_split_name].map(
                        self.preprocess,
                        batched=True,
                        remove_columns=column_names,
                        num_proc=1,
                        # load_from_cache_file=self.arguments.dataset_cache,
                        # fn_kwargs = {"evaluation": False},  # hf internal validation
                        fn_kwargs = {"evaluation": True, "class_ids":[1]},
                        desc="Running tokenizer on validation dataset",
                    )

                    # select 100 data from each class and merge them
                    self.eval_dataset = concatenate_datasets([true_val_dataset.select(range(100)),false_val_dataset.select(range(100))])
                else:
                    self.eval_dataset = raw_datasets[valid_split_name].map(
                        self.preprocess,
                        batched=True,
                        remove_columns=column_names,
                        num_proc=1,
                        # load_from_cache_file=self.arguments.dataset_cache,
                        # fn_kwargs = {"evaluation": False},  # hf internal validation
                        fn_kwargs = {"evaluation": True},
                        desc="Running tokenizer on validation dataset",
                    )
                self.eval_dataset.set_format(type="torch")

                
    def load_data_collator(self):
        if self.data_args.dataset_name == "ni":
            dataset_dependent_data_collator = DataCollatorForNI(
                self.tokenizer,
                model=self.model,
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
            layers = []
            if self.peft_args.bias_name == "encoder_bias":
                modules = self.model.encoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif self.peft_args.bias_name == "decoder_bias":
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif  self.peft_args.bias_name == "encoder_decoder_bias":
                modules = self.model.encoder.block
                for m in modules:
                    layers.append(m.layer[0])
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            else:
                raise ValueError("bias name not supported: ", arguments.bias_name)

            for l in layers:
                for name, module in l.named_modules():

                    # first check if bias is settable
                    if hasattr(module, "bias") and type(module) == transformers.adapters.lora.Linear:
                        if module.bias is None:
                            print("found none bias, init bias for ", name)
                            module.bias = torch.nn.Parameter(torch.randn(module.out_features))
                        if not module.bias.requires_grad:
                            print("activate gradient for ", name)
                            module.bias.requires_grad = True
        else:
            # NOTE: prompt tuning
            # general peft converting based on different peft config
            assert peft_config is not None, "peft config should be provided for non-adapter peft method"
            # convert text to token ids
            verbalizer_ids = [self.tokenizer.encode(v) for v in self.verbalizers]
            
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
        trainable_ratio = trainable_params/self.model_trainable_params
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

        # it has past run but might not have model checkpoint and wandb file
        # we should first guarantee that it has the model checkpoint and it's correctly loaded, otherwise, we re-init the tracker
        self.load_previous_run()

        
        # NOTE: gradient accumulation step is not unrelated to the computation below

        
        if self.global_step >= self.total_step:
            self.print_log(f"training is already finished, {self.start_epoch} epochs and {self.start_step} steps are already done")
            self.print_log("Ending training...")
            return


        self.print_log(f"Per step batch size (no grad acc): {train_bs_per_step}")
        # NOTE: only loss computation will be affected by gradient accumulation

        train_bs = self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps * self.num_processes
        self.print_log(f"Training batch size (considering grad acc): {train_bs}")

        # TODO: add expected train bs assertion or automatic adjusting

        if self.use_distributed:
            self.accelerator.log(self.training_args.to_dict())

            progress_bar = tqdm(
                range(self.global_step, self.total_step),
                disable=not self.accelerator.is_local_main_process or self.training_args.is_cluster,
                initial=self.global_step,
                # miniters=0 if not self.training_args.is_cluster else self.training_args.logging_steps
                miniters=self.training_args.logging_steps,
            )
            if self.global_step > 0:
                self.print_log(f"Resume training from epoch {self.start_epoch}, step {self.start_step}, global_step {self.global_step}")

                self.accelerator.skip_first_batches(self.train_dataloader,  self.start_step)
                self.print_log(f"skip first {self.start_step} steps in train_dataloader", print_step=False)
        else:
            progress_bar = tqdm(
                range(self.global_step, self.total_step),
                initial=self.global_step,
                # miniters=0 if not self.training_args.is_cluster else self.training_args.logging_steps,
                miniters=self.training_args.logging_steps,
                disable=self.training_args.is_cluster
            )

        self.model.train()
        logging_loss = 0
        

        for epoch in range(self.start_epoch, self.training_args.num_train_epochs):
            if self.accelerator.is_local_main_process:
                self.print_log(f"------------new epoch: {epoch} global_step: {self.global_step}")
            for step, inputs in enumerate(self.train_dataloader, start=self.start_step):
                self.accelerator.wait_for_everyone() # wait for all processes to finish the previous step
                self.train_state.update(
                            {
                                "epoch": epoch,
                                "step": step,
                                "global_step": self.global_step,
                            }
                        )
                if self.use_distributed:
                    # per progress bar step is actually gradient_accumulation_steps
                    with self.accelerator.accumulate(self.model):
                        if self.label_smoother is None:
                            outputs = self.model(**inputs)
                            loss = outputs["loss"]
                        else:
                            labels = inputs.pop("labels")
                            outputs = self.model(**inputs)
                            loss = self.label_smoother(outputs, labels)

                        # log before backward
                        self.accelerator.backward(loss) # it does gradient acc internally
                        
                        # under accelerator.accumulate context
                        # it steps until gradient_accumulation_steps
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.save_and_eval(self.global_step)

                else:
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.device)
                    outputs = self.model(**inputs)
                    loss = outputs["loss"]
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.save_and_eval(self.global_step)

                if self.training_args.is_cluster:
                    import hfai
                    # cluster pre-interrupt saving
                    if hfai.distributed.get_rank() == 0 and self.accelerator.is_local_main_process: # 获取当前节点序号。在0号节点的0号进程上接收集群调度信息
                        if hfai.client.receive_suspend_command(): 
                            self.print_log(f"Received suspend command, saving model at {self.global_step} steps")
                            self.save(self.global_step)
                            self.print_log(f"Model checkpoint at {self.global_step} steps is saved. Going suspend...")
                            hfai.client.go_suspend()
                            


                # log each backward step (not grad acc step)
                self.global_step += 1
                progress_bar.update(1)
                logging_loss += loss.item()
                
                # logging
                if self.global_step != 0 and self.global_step % self.training_args.logging_steps == 0:
                    self.log({
                            "train/loss": logging_loss/self.training_args.logging_steps,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            })
                    self.print_log(f"train/loss: {logging_loss/self.training_args.logging_steps}")
                    self.print_log(f"train/lr: {self.scheduler.get_last_lr()[0]}")
                    logging_loss = 0

    


            self.print_log(f"epoch {epoch} finished, evaluating...")
            # eval and save per epoch as well
            # guarantee to have a best checkpoint folder at the end of training
            self.save_and_eval(self.global_step, force=True if not ( self.training_args.dev_train or self.training_args.dev_run or self.training_args.dev_test) else False)
            self.print_log(f"epoch {epoch} finished, best_metric_step: {self.best_metric_step}, best_metric_val {self.best_metric_val}")
            self.print_log(f"steps per epoch: {self.global_step/(epoch+1)}")
        
        # log best metric val at final step for easy comparison
        self.log(
        {
            "best_metric_val": self.best_metric_val
        }
        )
        self.accelerator.end_training()
        # remove all step checkpoints after training is finished
        if self.accelerator.is_main_process:
            # only keep lastest checkpoint's train_state.json file
            remove_old_checkpoints(self.training_args.output_dir, num_to_keep=1)
            latest_cp = get_latest_checkpoint(self.training_args.output_dir)
            remove_files_and_folders_other_than(latest_cp, self.train_state.file_name)
        # wait for last eval saving and old checkpoint removal
        self.accelerator.wait_for_everyone()


    def log(self, d):
        """
        log to tensorboard/train state.
        but it doesn't save train state as train state could be saved to diff dirs.
        """
        self.accelerator.log(d,
                            step=self.global_step
        )
        self.train_state.update(d)
    
    def print_log(self, s, print_step=True):
        """
        print log under different training system.
        """
        if print_step:
            s = f"global_step {self.global_step}: {s}"
        if self.training_args.is_cluster:
            import hfai
            if hfai.distributed.get_rank() == 0:
                print(s)
        elif self.accelerator.is_main_process:
            logger.info(s)

    def evaluate(self, mode="eval", step=None):
        """
        eval mode: evaluate use loaded current model
        test mode: evaluate best model loaded from output_dir
        
        parallel generation, gather results on all processes, compute metrics on main process, return results on main process and None on other processes.

        log automatically on main process, return True if it outperform previous best checkpoint, False otherwise.
        """
        if mode=="test":
            assert step is None
        elif mode=="eval":
            assert step is not None
        else:
            raise NotImplementedError(
                "mode must be either eval or test mode"
            )
        self.model.eval()
        model = accelerate.utils.extract_model_from_parallel(self.model)
        if mode == "eval":
            dataset2eval = self.eval_dataset
            dataloader2eval = self.eval_dataloader
        elif mode == "test":
            torch.cuda.empty_cache()
            # NOTE: test evaluation is done, finish
            if self.test_eval_finished:
                self.print_log("test evaluation is done, finish...")
                return
            
            # free memory in test mode 
            dataset2eval = self.test_dataset
            dataloader2eval = self.test_dataloader
            best_cp_dir = None
            # during test mode, self.model is pretrained model. after loading state, it's the best checkpoint model
            try:
                best_cp_dir = get_latest_checkpoint(os.path.join(self.training_args.output_dir, "best_checkpoint"))
                print("load from existing state: ", best_cp_dir)
                assert best_cp_dir is not None, "It's expected to have dir for self.accelerator to load state"
                # only in this case we need to first move loaded pretrained mdoel to cpu
                # and load trained model weights into the model
                # then we move model back to gpu
                self.model = self.model.to("cpu")
                self.accelerator.load_state(best_cp_dir, map_location="cpu")
                self.model = self.model.to(self.accelerator.device)
                
            except Exception as e:
                self.accelerator.clear()
                # This might be buggy and needs to be improved.
                # sleep 5 seconds
                time.sleep(5) # wait for oom processes to be killed in case of oom
                print(f"load state failed, try to load from model_path due to \n {e}")
                if best_cp_dir is None:
                    print("No best checkpoint found, exiting")
                    exit()
                sharded_model_path = os.path.join(best_cp_dir, "save_pretrained")
                config = AutoConfig.from_pretrained(sharded_model_path)
                # reload tokenizer in this case
                if "llama" in self.model_args.model_name_or_path.lower():
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        self.potential_model_path,
                        truncation_side = "left"
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        sharded_model_path,
                        truncation_side = "left"
                    )

                model = self.load_pretrained_model(config)
                if hasattr(self.model, "load_adapter"):
                    adapter_path = os.path.join(sharded_model_path, self.model_args.tuning_mode)
                    model.load_adapter(adapter_path)
                
                # model = AutoModelForSeq2SeqLM.from_pretrained(self.model_args.model_name_or_path)
                # prepare model with dataloader (dataloader is already prepared tho)
                # move model to device and possibly ddp or deepspeed
                model, dataloader2eval = self.accelerator.prepare(model, dataloader2eval)

        input_host = []
        output_host = []
        label_host = []
        from tqdm import tqdm
        self.print_log(f"***** Running evaluation {mode} *****")
        # it handles deepspeed and DDP
        
        # print("extract model")
        from copy import deepcopy
        if self.accelerator.is_main_process and self.model_args.tuning_mode == "lora_adapter":
            pass
            # logger.debug('check active adapter')
            # model.model.layers
            # logger.debug("lora_B weight (should change): " + str(model.transformer.encoder.block[0].layer[0].SelfAttention.q.loras.lora_adapter.lora_B))
            # logger.debug("query weight(should not change): " + str(model.transformer.encoder.block[0].layer[0].SelfAttention.q.weight))
            

        with torch.no_grad():
            progress_bar = tqdm(dataloader2eval,
                                miniters=0 if not self.training_args.is_cluster else 500,
                                disable=self.training_args.is_cluster)
            for inputs in progress_bar:
                if not self.use_distributed:
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.device)
                labels = inputs.pop("labels")
                # if distrubted data parallel object 

                if self.model_args.tuning_mode == "lora_peft" or self.model_args.tuning_mode == "prompt_tuning": # temp PEFT lora implementation
                    generation_inputs = inputs.pop("input_ids")
                    outputs = model.generate(generation_inputs, **inputs,
                                        max_new_tokens = self.data_args.max_target_length,
                                        synced_gpus = True if self.use_distributed else False,
                                        pad_token_id=self.tokenizer.eos_token_id if self.model_args.model_arch == "decoder" else self.tokenizer.pad_token_id,
                                        # synced_gpus = False,
                                        # pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    # self.tokenizer.batch_decode(labels)
                    # self.tokenizer.batch_decode(inputs.input_ids)
                    outputs = model.generate(**inputs,
                                            max_new_tokens = self.data_args.max_target_length,
                                            # synced_gpus = True if self.use_distributed else False,
                                            synced_gpus = False,
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

        results = self.compute_metrics(
                (output_host, label_host),
                dataset2eval
            )
        results_with_mode = {}
        for k in results:
            results_with_mode[f"{mode}/{k}"] = results[k]
        self.model.train()
        self.log(results_with_mode)
        self.train_state.save_to_json(get_latest_checkpoint(self.training_args.output_dir))
        metric="rougeL"
        self.print_log(f"{mode}/{metric}: {results_with_mode[f'{mode}/{metric}']}")
        if mode == "test":
            self.train_state.update({"test_eval_finished": True})
            latest_cp = get_latest_checkpoint(self.training_args.output_dir)
            self.train_state.save_to_json(latest_cp)
            self.print_log("Finished test dataset evaluation...")
            
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
            
            # decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
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

        raise NotImplementedError("compute_metrics is not implemented for dataset other than ni")
        
        if not is_pred_logits:
            # based on predicted tokens to compute metrics
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels,
                                self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True)
            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )
            metric = load_metric("sacrebleu")
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            result = {f"bleu_{model_idx}": result["score"]}
            prediction_lens = [
                np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
            ]
            result[f"gen_len_{model_idx}"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            correct =0 
            missed_pred = [] # maybe empty or some other wrong predictions
            true_cnt = 0
            false_cnt = 0
            for idx, (pred, label)in enumerate(zip(decoded_preds, decoded_labels)):
                if idx < 5:
                    print("pred: ", pred, " label: ", label)
                if self.verbalizers[0] in pred and self.verbalizers[1] in pred:
                    print("both true and false in pred: ", pred)
                    continue
                # check if pred is correct
                if (self.verbalizers[0] in pred and self.verbalizers[0] in label):
                    correct += 1
                    true_cnt += 1
                elif (self.verbalizers[1] in pred and self.verbalizers[1] in label):
                    correct += 1
                    false_cnt += 1
                else:
                    print("missed pred: ", pred, " label: ", label)
                    missed_pred.append(pred)            
            result[f"acc_{model_idx}"] = correct / len(eval_dataset)
            result[f"true_ratio_{model_idx}"] = true_cnt / len(eval_dataset)
            result[f"false_ratio_{model_idx}"] = false_cnt / len(eval_dataset)
        else:
            model = self.trainer.model
            
            if self.model_args.model_arch == "encoder":
                import evaluate
                accuracy = evaluate.load("accuracy")
                print("preds: ", preds)
                preds = np.argmax(preds, axis=1)
                
                return accuracy.compute(predictions=preds, references=labels)
            
            encoder = model.get_encoder()

            dataloader = DataLoader(eval_dataset,
                                    # collate_fn=collate_fn,
                                    batch_size=self.training_args.per_device_eval_batch_size)
            
            correct = 0
            for idx, data in enumerate(dataloader):
                model_verbalizer_logits = [None, None]
                # batch_size = data["input_ids_0"].size(0)
                # input_ids = data["input_ids_0"]
                batch_size = data["input_ids"].size(0)
                input_ids = data["input_ids"]
                
                # attention_mask = data["attention_mask_0"]
                # labels = data["labels_0"]
                attention_mask = data["attention_mask"]
                # labels = data["label"]
                labels = data["labels"]
                class_ids = data["class_ids"]

                for j in range(len(self.verbalizers)):
                    v_ids = torch.tensor(self.tokenizer.encode(self.verbalizers[j]))
                    v_ids = v_ids[:1] # ONLY USE THE FIRST TOKEN
                    v_logits =  torch.zeros(batch_size)
                    decoder_input_ids = torch.zeros(input_ids.size(0),1).long().cuda()
                    # get out with encoder outputs
                    # decoder input ids are just pad tokens as <bos>
                    out = model(input_ids = input_ids.cuda(), attention_mask=attention_mask.cuda(), decoder_input_ids=decoder_input_ids)
                    # out = model(
                    #     input_ids = input_ids.cuda(), 
                    #     attention_mask=attention_mask.cuda(), 
                    #     # labels=labels.cuda()
                    # )
                    logits = out.logits.cpu()
                    
                    for seq_step, vid in enumerate(v_ids):
                        v_logits += logits[:, -1, vid].detach().cpu()    # logits for the batch

                    # model_verbalizer_logits: bs x 1
                    model_verbalizer_logits[j] = v_logits  # set jth verbalizer logits
                
                # compute probability using softmax
                model_verbalizer_logits = torch.stack(model_verbalizer_logits, dim=1)
                probs = torch.softmax(model_verbalizer_logits, dim=-1)
                
                predict_label = torch.argmax(probs, dim=1)
                correct += (predict_label == class_ids).sum()
            print(f"probs {model_idx}: ", probs) # just print the last probs
            # import pdb; pdb.set_trace()
            # print('double check ac checkpoint 2')
            
            result[f"acc_{model_idx}"] = correct / len(eval_dataset)

        return result

    def get_dataset_verbalizers(self, dataset):
        if dataset in ['sst-2', 'sst2', 
                       'yelp-2', 'mr', 'cr']:
            # verbalizers = ['\u0120terrible', '\u0120great'] # num_classes
            verbalizers = ['terrible', 'great']
        elif dataset == 'agnews': 
            verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
        elif dataset in ['sst-5', 'yelp-5']:
            # verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', 
            #             '\u0120good', '\u0120great'] # num_classes
            verbalizers = ['terrible', 'bad', 'okay', 'good', 'great']
        elif dataset == 'subj':
            # verbalizers = ['\u0120subjective', '\u0120objective']
            verbalizers = ['subjective', 'objective']
        elif dataset == 'trec':
            verbalizers = ['Description', 'Entity',
                        'Expression', 'Human',
                        'Location', 'Number']
            verbalizers = ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number']
        elif dataset == 'yahoo':
            verbalizers = ['culture', 'science',
                        'health', 'education',
                        'computer', 'sports',
                        'business', 'music',
                        'family', 'politics']
        elif dataset == 'dbpedia':
            verbalizers = ['\u0120Company', '\u0120Education',
                        '\u0120Artist', '\u0120Sports',
                        '\u0120Office', '\u0120Transportation',
                        '\u0120Building', '\u0120Natural',
                        '\u0120Village', '\u0120Animal',
                        '\u0120Plant', '\u0120Album',
                        '\u0120Film', '\u0120Written']
            verbalizers = ['Company', 'Education', 'Artist', 'Sports', 'Office', 'Transportation', 'Building', 'Natural', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written']
        elif dataset in ['axb', 'axg']:
            verbalizers = ["No", "Yes"]
        elif dataset in ['cb', "rtx"]:
            verbalizers = ["Yes", "No"]
        elif dataset in ['copa', ]:
            verbalizers = ["choice1", "choice2"]
        elif dataset in ['boolq','multirc', 'wic', 'wsc', 'wsc_fixed']:
            if dataset == 'boolq':
                # only boolq
                verbalizers = ["True", "False"]
            else:
                verbalizers = ["False", "True"]
        elif dataset == 'record':
            verbalizers = [None] # answer is text
        elif dataset == 'yelp_review_full':
            verbalizers = ['1', '2', '3', '4', '5']
        elif dataset == 'ag_news':
            verbalizers = ['World', 'Sports', 'Business', 'Sci/Tech']
        elif dataset == 'ni':
            verbalizers = []
        else:
            raise NotImplementedError("Dataset not supported: " + dataset)
        return verbalizers
    
    def _deactivate_relevant_gradients(self, trainable_components):
        """
        https://github.com/benzakenelad/BitFit/blob/7ead19a8350a01d5701f9e2df896a1c5b42c3723/glue_evaluator.py#L612
        """
        for param in self.model.parameters():
            param.requires_grad = False
        if trainable_components:
            trainable_components = trainable_components + ['pooler.dense.bias']
        trainable_components = trainable_components + ['classifier']
        # it iterates exsiting parameters only
        # bias init must be done before this
        for name, param in self.model.named_parameters():
            for component in trainable_components:
                # print(f"check component {name}")
                # if component in name:
                #     print(f"activate {name}")
                #     param.requires_grad = True
                #     break
                
                # brute force bias activation
                if "bias" in name:
                    print(f"activate {name}")
                    param.requires_grad = True
                    break


    def _format_prompts(self, prefix, source_strs, verbal_targets, include_labels_in_input = False):
        prompts = [""] * len(source_strs)
        prompts = [""]* len(source_strs)
        formatted_inputs = [f"{prefix} {s_1} {prompt} " for s_1, prompt in zip(source_strs, prompts)]
        formatted_inputs = [f"{input} " for input in formatted_inputs]

        if include_labels_in_input:
            labels = ",".join(verbal_targets)
            formatted_inputs = [f"{input} Decide the label in {self.verbalizers}." for input in formatted_inputs]    

        return formatted_inputs, verbal_targets

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
                device= self.peft_args.module_device,
                # prompt_tuning_init="TEXT",
                # prompt_tuning_init_text=prompt_tuning_init_text,
                # tokenizer_name_or_path=init_text_tokenizer_name_or_path,
            )
            self.load_peft_module(config)

        elif self.model_args.tuning_mode == "prefix_tuning":
            from transformers.adapters import PrefixTuningConfig
            # from peft import PrefixTuningConfig
            config = PrefixTuningConfig(
                        prefix_length=self.peft_args.prefix_len,        
                        bottleneck_size=self.peft_args.bottleneck_size,
                        encoder_prefix=True,
                        cross_prefix=True,
                        dropout=self.peft_args.dropout_rate,
            )
            self.load_peft_module(config)

        
        elif self.model_args.tuning_mode == "lora_adapter":
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=self.peft_args.lora_r ,
                                alpha=self.peft_args.lora_alpha,
                                attn_matrices=list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"],
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
                target_modules= list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"],
            )
            get_peft_model(self.model, config)

        elif self.model_args.tuning_mode == "ia3":
            from transformers.adapters import IA3Config

            config = IA3Config(
                dropout=self.peft_args.dropout_rate,
            )
            self.load_peft_module(config)

        elif self.model_args.tuning_mode in ["adapter", "compactor"]:
            cur_reduction_factor = 64 if self.peft_args.reduction_factor is  None else self.peft_args.reduction_factor
 
            from transformers.adapters import AdapterConfig, HoulsbyConfig, CompacterConfig
            # check existing adapter and remove them
            # config = AdapterConfig()
            if self.model_args.tuning_mode == "adapter":
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

        elif self.model_args.tuning_mode == "embedding_tuning":
            self.convert_to_embedding_tuning()
            if self.peft_args.num_soft_tokens > 0:
                self.peft_args.num_soft_tokens = 0
                print("num_soft_tokens is set to 0 for embedding tuning mode")
        elif self.model_args.tuning_mode == "bitfit":
            self.load_peft_module()
        elif self.model_args.tuning_mode == "fine_tuning":
            # no converting needed
            pass
        else:
            raise NotImplementedError(f"mode {self.model_args.tuning_mode} is not implemented")

    def save_and_eval(self, global_step, force=False):
        """
        if global step satisfies condition, save and/or evaluate.

        force is used in end of training or after each epoch.
        """
        if force or ((global_step != 0 or self.training_args.dev_run) and global_step % self.training_args.save_steps == 0):
            self.save(global_step)

        
        if force or ((global_step != 0 or self.training_args.dev_run) and global_step % self.training_args.eval_steps == 0):
            results = self.evaluate(step=global_step)
            eval_metric_name = "eval/"+self.training_args.eval_metric
            cur_metric_val = results[eval_metric_name]
            self.print_log(f"current metric_val: {cur_metric_val}")
            if cur_metric_val > self.best_metric_val:
                self.best_metric_val = cur_metric_val
                self.best_metric_step = global_step
                # log before save
                self.log(
                    {
                        "best_metric_val": self.best_metric_val,
                        "best_metric_step": global_step,
                    }
                )
                self.print_log(f"best_metric_val: {self.best_metric_val}, new best_metric_step: {global_step}")
                self.print_log("saving a new best checkpoint...")
                # save model and state
                self.save(global_step, save_best_checkpoint=True)



    def save(self, global_step, remove_old_cp = True, save_best_checkpoint = False):
        """
        cp_folder_path could be best checkpoint folder path or checkpoint folder path.
        
        if save_best_checkpoint is True, then save to best checkpoint folder path.
        """
        start_time = time.time()
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
        self.accelerator.save_state(checkpoint_folder_path_to_save)
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
        # save new train state only if it is best checkpoint     
        if self.accelerator.is_main_process:
            self.train_state.save_to_json(checkpoint_folder_path_to_save)
            print("Model saving time in seconds:", time.time() - start_time)
            if remove_old_cp:
                remove_old_checkpoints(checkpoint_dir_path, self.training_args.checkpoint_save_total_limit)

    def load_previous_run(self):
        """
        load previous run from checkpoint folder, if failed, then init a new run.
        """
        loaded = False
        while not loaded:
            self.accelerator.wait_for_everyone()
            if os.path.exists(self.training_args.output_dir):
                self.print_log("load from existing state", print_step=False)
                latest_cp = get_latest_checkpoint(self.training_args.output_dir)
                

                # if no checkpoint, then remove the folder and re-init the tracker
                if latest_cp is None and self.accelerator.is_local_main_process:
                    # if os.path.exists(self.training_args.output_dir):
                    shutil.rmtree(self.training_args.output_dir)
                    logger.warn(f"no checkpoint found, remove the folder {self.training_args.output_dir} and re-init the tracker")
                    time.sleep(3)

                # try to load the latest checkpoint
                try:
                    self.train_state.load_from_json(latest_cp)
                    self.accelerator.init_trackers(
                        self.training_args.run_name,
                        config=self.train_state.state_dict,
                        init_kwargs={"tensorboard": {"flush_secs": 60}},
                    )
                    self.start_epoch = self.train_state.get("epoch")
                    self.start_step = self.train_state.get("step")
                    self.global_step =  self.train_state.get("global_step")
                    self.test_eval_finished = self.train_state.get("test_eval_finished")
                    self.best_metric_step = self.train_state.get("best_metric_step")
                    # if training is finished, then only load the state
                    # because the model checkpoint is already removed
                    if self.global_step >= self.total_step:
                        self.print_log(f"training is already finished, {self.start_epoch} epochs and {self.start_step} steps are already done")
                        self.print_log("Only train state is loaded...")
                        return True
                    
                    # only train state exists in latest checkpoint
                    # so no further state loading
                    if self.test_eval_finished:
                        self.print_log("test evaluation is already finished,  exit...")
                        exit()
                    
                    self.accelerator.load_state(latest_cp)
                    # TODO: is it better to store wandb separately under every checkpoint folder? Since in offline mode, it cannot be resuemd anyway. But upload to wandb might be tricky as it requires some script to extract.
                    

                    
                    loaded = True
                except Exception as e:
                    # it can be state dict file corrupted or model checkpoint corrupted
                    # in any case, remove the checkpoint folder and reload the previous one
                    # remove the latest checkpoint
                    # output_dir might already has been removed
                    if latest_cp is not None and self.accelerator.is_local_main_process:
                        self.print_log(f"remove the latest checkpoint {latest_cp} due to\n\n {e}")
                        shutil.rmtree(latest_cp)
                    # still not loaded
                    continue
            else:
                self.print_log(f"no previous run found, create a new one at {self.training_args.output_dir}")
                os.makedirs(self.training_args.output_dir, exist_ok = True)
                self.accelerator.init_trackers(
                        self.training_args.run_name,
                        config=self.train_state.state_dict,
                        init_kwargs={"tensorboard": {"flush_secs": 60}},
                )
                loaded = True
        return loaded

    
    def search_for_hyperperameters(self):
        # assert  self.use_distributed is False, "search for hyperparameters doesn't need distributed training"

        self.model = self.load_pretrained_model()

        
        # trainable parameters percent list
        # for loop linearly for each hyperparameter for the peft method
        # print out corresponding hyperparameter
        # exit()
        # parallel to configure_n_load_peft_module
        
        # in init method, model and peft method has
        # we reload all of peft modules
        
        self.model_cache = deepcopy(self.model)
        from peft import LoraConfig
        # peft package
        cur_lora_r = 1 if self.peft_args.lora_r is None else self.peft_args.lora_r
        # adapter
        cur_reduction_factor = 64 if self.peft_args.reduction_factor is  None else self.peft_args.reduction_factor
        # prompt tuning
        cur_prompt_len = 1 if self.peft_args.prompt_len is None else self.peft_args.prompt_len
        # prefix tuning
        cur_prefix_len = 1 if self.peft_args.prefix_len is None else self.peft_args.prefix_len
        bottleneck_size = 10 # original paper, 512 table to text, 800 for summarization
        
        
        trainable_params_percentage = self.training_args.trainable_params_percentage
        assert trainable_params_percentage is not None and len(trainable_params_percentage) > 0
        cur_trainable_params_percent= 0
        percent_iterator = iter(trainable_params_percentage)
        cur_percent_threshold = next(percent_iterator)
        task_type = TaskType.SEQ_2_SEQ_LM
        while trainable_params_percentage and cur_trainable_params_percent < cur_percent_threshold:
            if self.model_args.tuning_mode == "lora_peft":
                task_type = TaskType.SEQ_2_SEQ_LM
                config = LoraConfig(
                    task_type=task_type,
                    inference_mode=False,
                    r=cur_lora_r,
                    lora_alpha=32,
                    lora_dropout=0,
                    # lora_modules
                    target_modules= list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"],
                )
                # trainable_params_status = self.load_peft_module(config, reset_peft=True)
                # cur_trainable_params_percent = trainable_params_status["trainable_ratio"]
                if cur_trainable_params_percent > cur_percent_threshold:
                    print(f"model: {self.model_args.model_name_or_path}, lora_r: {cur_lora_r}, lora_alpha: {config.lora_alpha}, lora_dropout: {config.lora_dropout}, lora_modules: {config.target_modules}, trainable_params_percent: {cur_trainable_params_percent} for percent threshold: {cur_percent_threshold}")
                    try:
                        cur_percent_threshold = next(percent_iterator)
                    except StopIteration:
                        break
                        
                cur_lora_r += 1
        
            elif self.model_args.tuning_mode in ["adapter", "compactor"]:
                from transformers.adapters import AdapterConfig, HoulsbyConfig, CompacterConfig

                if self.model_args.tuning_mode == "adapter":
                    config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
                else:
                    config = CompacterConfig(reduction_factor=cur_reduction_factor,
                                            phm_dim=self.peft_args.phm_dimension)
                
                
                # reload peft module with new config
                trainable_params_status = self.load_peft_module(config, reset_peft=True)
                cur_trainable_params_percent = trainable_params_status["trainable_ratio"]
                
                if cur_trainable_params_percent > cur_percent_threshold:
                    print(f"model: {self.model_args.model_name_or_path}, reduction_factor: {cur_reduction_factor}, phm_dim: {config.phm_dim}, trainable_params_percent: {cur_trainable_params_percent} for percent threshold: {cur_percent_threshold}")
                    try:
                        cur_percent_threshold = next(percent_iterator)
                    except StopIteration:
                        break
                cur_reduction_factor /=1.01
            elif self.model_args.tuning_mode == "prompt_tuning":
                config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=cur_prompt_len,
                    inference_mode=False,
                    device= self.peft_args.module_device,
                    # prompt_tuning_init="TEXT",
                    # prompt_tuning_init_text=prompt_tuning_init_text,
                    # tokenizer_name_or_path=init_text_tokenizer_name_or_path,
                )
                trainable_params_status = self.load_peft_module(config, reset_peft=True)
                cur_trainable_params_percent = trainable_params_status["trainable_ratio"]
                if cur_trainable_params_percent > cur_percent_threshold:
                    print(f"model: {self.model_args.model_name_or_path}, num_virtual_tokens: {cur_prompt_len}, trainable_params_percent: {cur_trainable_params_percent} for percent threshold: {cur_percent_threshold}")
                    try:
                        cur_percent_threshold = next(percent_iterator)
                    except StopIteration:
                        break
                cur_prompt_len += 1
            elif self.model_args.tuning_mode == "prefix_tuning":
                from transformers.adapters import PrefixTuningConfig

                config = PrefixTuningConfig(prefix_length=cur_prefix_len, bottleneck_size=bottleneck_size,
                                            encoder_prefix=True,
                                            cross_prefix=True)
                trainable_params_status = self.load_peft_module(config, reset_peft=True)
                cur_trainable_params_percent = trainable_params_status["trainable_ratio"]
                if cur_trainable_params_percent > cur_percent_threshold:
                    print(f"model: {self.model_args.model_name_or_path}, prefix_length: {cur_prefix_len}, bottleneck_size: {bottleneck_size}, trainable_params_percent: {cur_trainable_params_percent} for percent threshold: {cur_percent_threshold}")
                    try:
                        cur_percent_threshold = next(percent_iterator)
                    except StopIteration:
                        break
                cur_prefix_len += 1
