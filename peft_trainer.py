from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Model, T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModelForSequenceClassification, Seq2SeqAdapterTrainer, TrainerState

from datasets import load_dataset, load_metric, concatenate_datasets
import numpy as np
from torch.nn import CrossEntropyLoss
import torch
from transformers import (
    GPT2TokenizerFast,
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
import accelerator
from peft import PeftModelForSeq2SeqLM


# modules use two pacakges 
ADAPTER_TRANSFORMERS_MODULES=["adapter", "compactor", "prefix_tuning", "ia3", "lora", "parallel_adapter"]
PEFT_MODULES=["prompt_tuning"]

class PEFTTrainer:
    def __init__(self, training_args, data_args, model_args, peft_args):
        
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.peft_args = peft_args
        
        # self.arguments = arguments
        self.model_name_or_path = self.model_args.model_name_or_path
        self.verbalizers = self.get_dataset_verbalizers(self.data_args.dataset_name if self.data_args.dataset_name != "super_glue" else self.data_args.dataset_config_name)
        
        self.model = None
        self.model_trainable_params = None
        self.recover_from = None
        # if self.training_args.do_train:
        self.load_model_n_peft_module()
        assert self.model is not None, "model should loaded"
        
        assert self.model_trainable_params is not None, "model_trainable_params should be counted"
            
        self.load_tokenzier()
        assert self.tokenizer is not None, "tokenizer should loaded"
        self.load_data_collator()
        


        
        # recover from will be set in load_model() method
        # 1. None. start training from scratch.
        # 2. checkpoint
        # 3. trainer state
        # self.recover_from = None
        # self.load_pretrained_model_for_peft()
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



    def load_model_n_peft_module(self, device="cuda"):
        self.load_model()
        
        
        self.configure_n_load_peft_module() # always load model from scratch for accelerate
        # handle situation loading model from checkpoint
        # if self.recover_from is None:
        #     # loaded pretrained model without adapter, needs peft module
        #     self.configure_n_load_peft_module()
        # else:
        #     # loaded model with peft module, needs set up
        #     adapter_name = self.model_args.tuning_mode
        #     self.model.set_active_adapters(adapter_name)
        #     self.model.train_adapter(adapter_name, freeze_model=True)
        #     self.model.heads[f"lm_head-{adapter_name}"][0].weight.requires_grad = False

        #     # loaded peft module might not at the correct device, load them to
        #     # model's device
        #     for param in self.model.parameters():
        #         model_device = param.device
        #         break
        #     self.model = self.model.to(model_device)
        # self.model = self.model.to(device)

        
    
    def configure_n_load_peft_module(self):
        # model loading procedure:
        # 1. load model from model_name_or_path    (self.load_pretrained_model_for_peft())
        # 2. not satisfied with peft, load model from self.model_cache and convert again. self.model = deepcopy(self.model_cache)
        if self.model_args.tuning_mode == "prompt_tuning":
            cur_prompt_len = 1
            assert self.peft_args.trainable_params_percentage is not None or self.peft_args.num_soft_tokens > 0, "either prompt_len or trainable_params_percentage should be set"
            config = PromptTuningConfig(
                task_type=task_type,
                num_virtual_tokens=cur_prompt_len,
                inference_mode=False,
                device= self.peft_args.module_device,
                # prompt_tuning_init="TEXT",
                # prompt_tuning_init_text=prompt_tuning_init_text,
                # tokenizer_name_or_path=init_text_tokenizer_name_or_path,
            )
            cur_trainable_params_percentage = self.load_peft_module(config)
            while self.peft_args.trainable_params_percentage and cur_trainable_params_percentage < self.peft_args.trainable_params_percentage:
                    
                config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=cur_prompt_len,
                    inference_mode=False,
                    device= self.peft_args.module_device,
                    
                    
                    # prompt_tuning_init="TEXT",
                    # prompt_tuning_init_text=prompt_tuning_init_text,
                    # tokenizer_name_or_path=init_text_tokenizer_name_or_path,
                )
                cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            
                print("prompt length is {}".format(cur_prompt_len))
                print("trainable params percentage is {}".format(cur_trainable_params_percentage))
                cur_prompt_len += 1

        elif self.model_args.tuning_mode == "prefix_tuning":
            from transformers.adapters import PrefixTuningConfig
            # from peft import PrefixTuningConfig
            
            cur_prefix_len = 1 if self.peft_args.prefix_len is None else self.peft_args.prefix_len
            bottleneck_size = 576
            assert self.peft_args.trainable_params_percentage is not None or self.peft_args.prefix_len > 0, "either prefix_len or trainable_params_percentage should be set"

            config = PrefixTuningConfig(prefix_length=cur_prefix_len,       bottleneck_size=bottleneck_size,
                                        encoder_prefix=True,
                                        cross_prefix=True)
            cur_trainable_params_percentage = self.load_peft_module(config)
            while self.peft_args.trainable_params_percentage and cur_trainable_params_percentage  < self.peft_args.trainable_params_percentage:
                cur_prefix_len += 1
                # config = PrefixTuningConfig(prefix_length=cur_prefix_len, flat=True)
                config = PrefixTuningConfig(prefix_length=cur_prefix_len, bottleneck_size=bottleneck_size,
                                            encoder_prefix=True,
                                            cross_prefix=True)
                cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
                print("prefix length is {}".format(cur_prefix_len))
                
            
        
        elif self.model_args.tuning_mode == "lora":
            # peft package
            cur_lora_r = 15 if self.peft_args.lora_r is None else self.peft_args.lora_r
            assert self.peft_args.trainable_params_percentage is not None or self.peft_args.lora_r > 0, "either lora_r or trainable_params_percentage should be set"

            # config = LoraConfig(
            #     task_type=task_type,
            #     inference_mode=False,
            #     r=cur_lora_r,
            #     lora_alpha=32,
            #     lora_dropout=0.1,
            #     # lora_modules
            #     target_modules= list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"],
            # )
            # cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            # print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)
            # while self.peft_args.trainable_params_percentage and cur_trainable_params_percentage < self.peft_args.trainable_params_percentage:
            #     cur_lora_r += 1
            #     config = LoraConfig(
            #         task_type=task_type,
            #         inference_mode=False,
            #         r=cur_lora_r,
            #         lora_alpha=32,
            #         lora_dropout=0.1,
            #         target_modules=list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"],
            #     )
            #     cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            #     print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)

            
            from transformers.adapters import LoRAConfig
            config = LoRAConfig(r=cur_lora_r,
                                alpha=16,
                                attn_matrices=list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"]
                                )
            # self.model.add_adapter("lora_adapter", config=config)
            cur_trainable_params_percentage = self.load_peft_module(config)
            while self.peft_args.trainable_params_percentage and cur_trainable_params_percentage < self.peft_args.trainable_params_percentage:
                cur_lora_r += 1
                config = LoRAConfig(
                                r=cur_lora_r,
                                alpha=16,
                                attn_matrices=list(self.peft_args.lora_modules) if self.peft_args.lora_modules else ["q", "v"]
                                )
                cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
                print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)
            
            
            
            # cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            
            
                
            self.peft_args.lora_r = cur_lora_r
        elif self.model_args.tuning_mode == "ia3":
            from transformers.adapters import IA3Config
            cur_lora_r = 15 if self.peft_args.lora_r is None else self.peft_args.lora_r
            assert self.peft_args.trainable_params_percentage is not None or self.peft_args.lora_r > 0, "either lora_r or trainable_params_percentage should be set"

            config = IA3Config(
                r = cur_lora_r,
            )
            cur_trainable_params_percentage = self.load_peft_module(config)
            print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)
            while self.peft_args.trainable_params_percentage and cur_trainable_params_percentage < self.peft_args.trainable_params_percentage:
                cur_lora_r += 1
                config = IA3Config(
                    r = cur_lora_r,
                )
                cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
                print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)

        elif self.model_args.tuning_mode in ["adapter", "compactor"]:
            cur_reduction_factor = 64 if self.peft_args.reduction_factor is  None else self.peft_args.reduction_factor
            assert self.peft_args.trainable_params_percentage is not None or self.peft_args.reduction_factor > 0, "either reduction_factor or trainable_params_percentage should be set"

            from transformers.adapters import AdapterConfig, HoulsbyConfig, CompacterConfig
            # check existing adapter and remove them
            # config = AdapterConfig()
            if self.model_args.tuning_mode == "adapter":
                config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
            else:
                config = CompacterConfig(reduction_factor=cur_reduction_factor,
                                            phm_dim=self.peft_args.phm_dimension )

            cur_trainable_params_percentage = self.load_peft_module(config)
            while self.peft_args.trainable_params_percentage and  cur_trainable_params_percentage < self.peft_args.trainable_params_percentage:
                cur_reduction_factor /=1.01
                if self.model_args.tuning_mode == "adapter":
                    config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
                else:
                    config = CompacterConfig(reduction_factor=cur_reduction_factor,
                                             phm_dim=self.peft_args.phm_dimension)
                cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
                print(f"cur_trainable_params_percentage: {cur_trainable_params_percentage}, cur_reduction_factor: {cur_reduction_factor}")
        elif self.model_args.tuning_mode == "parallel_adapter":
            from transformers.adapters import ParallelConfig
            config = ParallelConfig(reduction_factor= self.peft_args.reduction_factor)
            cur_trainable_params_percentage = self.load_peft_module(config)
            print(f"cur_trainable_params_percentage: {cur_trainable_params_percentage}")
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
        elif self.model_args.tuning_mode == "lm_head_tuning":
            for param in self.model.parameters():
                param.requires_grad = False
            # lm head takes almost 10% paramaters
            for name, module in self.model.named_modules():
                if "lm_head" in name:
                    module.weight.requires_grad = True
        elif self.model_args.tuning_mode == "layer_tuning":
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            layers = []
            # NOTE: we only fine-tune attention weights for now
            if self.peft_args.layer_name == "first_encoder_layer":
                layers.append(self.model.encoder.block[0].layer[0])
            elif self.peft_args.layer_name == "last_encoder_layer":
                layers.append(self.model.encoder.block[-1].layer[0])
            elif self.peft_args.layer_name == "first_decoder_layer":
                layers.append(self.model.decoder.block[0].layer[0])
            elif self.peft_args.layer_name == "last_decoder_layer":
                layers.append(self.model.decoder.block[-1].layer[0])
            elif self.peft_args.layer_name == "custom":
                # all decoder layer
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif self.peft_args.layer_name == "custom2":
                # all decoder layer
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            else:
                raise NotImplementedError(f"layer_name {self.peft_args.layer_name} is not implemented")
            
            for l in layers:
                for name, module in l.named_modules():
                    # if "selfattention" in name.lower():
                    if hasattr(module, "weight"):
                        module.weight.requires_grad = True
                        print("activate gradient for ", name)
        elif self.model_args.tuning_mode == "lora+adapter":
            from transformers.adapters import AdapterConfig, HoulsbyConfig, CompacterConfig
            
            
            # model.base_model.model_frozen = True
            # print('cur_trainable_params_percentage', self.check_trainable_parameters())
            # # peft package
            # cur_lora_r = 15 if self.peft_args.lora_r is None else self.peft_args.lora_r
            # assert self.peft_args.trainable_params_percentage is not None or self.peft_args.lora_r > 0, "either lora_r or trainable_params_percentage should be set"
            # task_type = TaskType.SEQ_2_SEQ_LM
            # config = LoraConfig(
            #     task_type=task_type,
            #     inference_mode=False,
            #     r=cur_lora_r,
            #     lora_alpha=32,
            #     lora_dropout=0.1
            # )
            # cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            
            
            
            
            # lora 2
            cur_lora_r = 40 if self.peft_args.lora_r is None else self.peft_args.lora_r
            freeze_model = True
            from transformers.adapters import LoRAConfig
            config = LoRAConfig(r=cur_lora_r, alpha=16)
            self.model.add_adapter("lora_adapter", config=config)
            # cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            self.check_trainable_parameters()
            print("lora_adapter is added, trainable parameters: ", self.check_trainable_parameters())

            # adapter
            cur_reduction_factor=18
            peft_config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
            reset_peft = False
            # cur_trainable_params_percentage = self.load_peft_module(config, reset_peft=True)
            self.model.add_adapter(self.model_args.tuning_mode,
                                   config = peft_config, overwrite_ok=reset_peft)
            self.model.train_adapter(self.model_args.tuning_mode)
            print('cur_reduction_factor', cur_reduction_factor, 'cur_trainable_params_percentage', self.check_trainable_parameters())
            # do not freeze model as model itself is already frozen
            # we don't want to freeze the lora module
            
            # self.model.train_adapter(["sst-2","lora_adapter"] , freeze_model=freeze_model)
            self.model.train_adapter(["sst-2", "lora_adapter"], freeze_model=freeze_model)
            
            if self.model_args.model_arch == "encoder":
                self.model.add_classification_head("lm_head-sst-2", num_labels=2, overwrite_ok=reset_peft)
            elif self.model_args.model_arch == "encoder-decoder":
                self.model.add_seq2seq_lm_head("lm_head-sst-2", overwrite_ok=reset_peft)
                # reset weight self.model.heads["lm_head-sst-2"][0].weight
                # self.model.heads["lm_head-sst-2"][0].weight = self.model_lm_head_weight

                self.model.heads["lm_head-sst-2"][0].weight.requires_grad = False
                
                
            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.model_args.model_arch}"
                )
            
            self.model.set_active_adapters(["sst-2", "lora_adapter"])
            
            
            print('cur_reduction_factor', cur_reduction_factor, 'cur_trainable_params_percentage', self.check_trainable_parameters())

            self.peft_args.lora_r = cur_lora_r

            
            # invalid
        elif self.model_args.tuning_mode == "unipelt":
            from transformers.adapters import UniPELTConfig, PrefixTuningConfig, PfeifferConfig, LoRAConfig, HoulsbyConfig
            gating = False
            reset_peft=False
            # peft_config = UniPELTConfig(
            #     PrefixTuningConfig(prefix_length=1, use_gating=self.peft_args.use_pelt_gate),
            #     PfeifferConfig(reduction_factor=500, use_gating=self.peft_args.use_pelt_gate),
            #     LoRAConfig(r=self.peft_args.lora_r, use_gating=self.peft_args.use_pelt_gate),
            #     )
            peft_config = UniPELTConfig(
                PrefixTuningConfig(prefix_length=1, use_gating=self.peft_args.use_pelt_gate),
                HoulsbyConfig(reduction_factor=500, use_gating=self.peft_args.use_pelt_gate),
                LoRAConfig(r=self.peft_args.lora_r, use_gating=self.peft_args.use_pelt_gate),
                )
            self.model.add_adapter(adapter_name, config = peft_config)
            self.model.train_adapter(adapter_name)
            
            
            self.model.add_seq2seq_lm_head(f"lm_head-{adapter_name}", overwrite_ok=reset_peft)
            self.model.set_active_adapters(adapter_name)
            # reset weight self.model.heads["lm_head-sst-2"][0].weight
            # self.model.heads[f"lm_head-{adapter_name}"][0].weight = self.model_lm_head_weight

            self.model.heads[f"lm_head-{adapter_name}"][0].weight.requires_grad = False

            print('cur_trainable_params_percentage', self.check_trainable_parameters())
        
        else:
            raise NotImplementedError(f"mode {self.model_args.tuning_mode} is not implemented")
    
    def load_optimizer_n_scheduler(self):
        self.optimizer = Adafactor(
            self.model.parameters(),
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            lr= self.training_args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=1e-5,
            # fixed learning rate
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        self.scheduler = get_constant_schedule(self.optimizer)

    def load_model(self):
        """
        General load model method.
        
        train mode:
        - load model from scratch
        - not load model (resume from trainer state)

        test mode:
        - load model from checkpoint
        - load model from trainer state

        Generally,
        1. load self.model
        2. compute trainable params stats, record it into self.model_trainable_params
        3. load into trainer.model
        4. Also, determine self.recover_from
        """
        if self.training_args.do_train:
            pass
            # we don't consider a situation that we load a specific checkpoint for training
            # check under ouput_dir if there is checkpoint folder
            # from utils import get_latest_checkpoint
            # if os.path.exists(self.training_args.output_dir):
            #     # resume checkpoint from trainer state, assume it has checkpoint for now
            #     # latest_checkpoint = get_latest_checkpoint(self.training_args.output_dir)
            #     # assert latest_checkpoint is not None, "output directory has no checkpoint, consider --override_outputdir or remove the folder"
            #     # # while latest_checkpoint is not None:
            #     # while self.recover_from is None:
            #     #     try:
            #     #         trainer_state_path = os.path.join(latest_checkpoint, 'trainer_state.json')
            #     #         # Get the path to the best checkpoint
            #     #         model_path = TrainerState.load_from_json(trainer_state_path).best_model_checkpoint
            #     #         # no need for conversion
            #     #         self.load_pretrained_model_for_peft(model_path)
            #     #         self.recover_from = "trainer_state"
            #     #     except:
            #     #         print(f"Failed to load checkpoint {latest_checkpoint}, try previous checkpoint")
            #     #         import shutil
            #     #         # remove the corrupted checkpoint
            #     #         shutil.rmtree(latest_checkpoint)
            #     #         # re-check latest checkpoint
            #     #         latest_checkpoint = get_latest_checkpoint(self.training_args.output_dir)
            #     #         continue
            #     latest_checkpoint = os.path.join(self.training_args.output_dir, "latest_checkpoint")

            #     try:
            #         trainer_state_path = os.path.join(latest_checkpoint, 'trainer_state.json')
            #         # Get the path to the best checkpoint
            #         model_path = TrainerState.load_from_json(trainer_state_path).best_model_checkpoint
            #         # no need for conversion
            #         self.load_pretrained_model_for_peft(model_path)
            #         self.recover_from = "trainer_state"
            #     except Exception as e:
            #         print(f"Failed to load checkpoint {latest_checkpoint}, removing ")
            #         import shutil
            #         # remove the corrupted checkpoint
            #         # shutil.rmtree(latest_checkpoint)
            #         raise e
            # else:
            #     # start training
            #     self.load_pretrained_model_for_peft()
            #     # self.load_peft_module() # load peft module for training start
        
        self.load_pretrained_model_for_peft()
        
        # init trainer requires model or model_init
        # then trainer create output directory
        # trainer will call model_init again

        # 1. it's better to take model_init out of model, to better controll break point
        # 2. let model load pretrained model first 
        
        
        if self.training_args.do_test:
            pass
        #     if "checkpoint" in self.model_args.model_name_or_path:
        #         print("Found checkpoint in model_name_or_path, so load model from checkpoint: ", self.model_args.model_name_or_path)
        #         model_path = self.model_args.model_name_or_path
        #         self.recover_from = "checkpoint"
        #     elif os.path.exists(self.training_args.output_dir):
        #         # in test mode, state needs to be used to load the best checkpoint explicitly
        #         print("Found trainer output path, so load best model checkpoint from trainer state: ", self.training_args.output_dir)
        #         state = TrainerState.load_from_json(self.training_args.output_dir)
        #         # Get the path to the best checkpoint
        #         model_path = state.best_model_checkpoint
        #         self.recover_from = "trainer_state"
        #     else:
        #         raise ValueError("trainer output path is not specified or not exists: ", self.training_args.output_dir)
        
        #     if self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
        #         self.model = AutoAdapterForSeq2SeqLM.from_pretrained(
        #             model_path,
        #             cache_dir = self.training_args.cache_dir,
        #         )
        #     elif self.model_args.tuning_mode in PEFT_MODULES:
        #         raise NotImplementedError("prompt tuning model loading is not supported yet for testing")
        #     else:
        #         self.model = AutoModelForSeq2SeqLM.from_pretrained(
        #             model_path,
        #             cache_dir = self.training_args.cache_dir,
        #         )

        # assert self.model is not None, "It should already loaded model"
        
        return self.model
            
    def load_tokenzier(self, potential_model_path=None):
        potential_model_path = os.path.join(
            self.training_args.saved_pretrained_model_path,
            self.model_name_or_path
        )  if potential_model_path is None else potential_model_path
        if os.path.exists(potential_model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(potential_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                # use_cache = self.arguments.use_cache,
                use_fast=True,
                return_tensors="pt"
            )
        if self.tokenizer.pad_token is None:
            assert self.model_name_or_path == "gpt2", "Only gpt2 is expected not having pad tokens for now"
            # gpt2 model
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            m.resize_token_embeddings(len(self.tokenizer))
        
        
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        print('gpt2 requires padding to max length')
        if "gpt2" in self.model_name_or_path:
            self.padding = "max_length"


    def load_pretrained_model_for_peft(self, potential_model_path=None):
        """
        1. Load model, tokenizer by model architecture and peft packages.
        2. load model from potential checkpoint/saved_pretrained model
        3. handles model parallel if needed.
        NOTE: it doesn't load peft module if it's not from checkpoint.
        """
        print(
            "Loading",
            self.model_args.model_name_or_path,
            "(for large models, this might take a while)",
        )
        print("Files will be cached at:", self.training_args.cache_dir)
        print(
            "Ensure this directory is persistent if you do not want to download model files again!"
        )

        
        potential_model_path = os.path.join(
            self.training_args.saved_pretrained_model_path,
            self.model_name_or_path
        ) if potential_model_path is None else potential_model_path
        if "t5" in self.model_name_or_path or "bart" in self.model_name_or_path:
            if self.model_args.tuning_mode in ["fine_tuning", "prompt_tuning"]:
                # trainer not compactiable with AdapterTrainer yet due to forward function not returning loss
                if os.path.exists(potential_model_path):
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(potential_model_path)
                else:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir,)
            elif self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES:
                if os.path.exists(potential_model_path):
                    self.model = AutoAdapterModel.from_pretrained(potential_model_path)
                else:
                    self.model = AutoAdapterModel.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir)
                
                    
            elif self.model_args.tuning_mode in PEFT_MODULES:
                # NOTE: this is not compatible if loading for the first time as
                # for peft package, loading by AutoModelForSeq2SeqLM is good enough
                if os.path.exists(potential_model_path):
                    self.model = PeftModelForSeq2Seq.from_pretrained(potential_model_path)
                else:
                    self.model = PeftModelForSeq2Seq.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir)
            else:
                raise NotImplementedError("Tuning mode not supported: " + self.model_args.tuning_mode)



            
        elif "roberta" in self.model_name_or_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        elif "gpt2" in self.model_name_or_path or "bloom" in self.model_name_or_path or "opt" in self.model_name_or_path:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                # from_tf=bool(".ckpt" in self.model_name_or_path),
                # config=m_config,
                cache_dir=self.training_args.cache_dir,
            )
        else:
            raise NotImplementedError("Model not supported: " + self.model_name_or_path)

        # if self.model_args.tuning_mode in ADAPTER_TRANSFORMERS_MODULES: # "prefix_tuning"
        #     self.model = AutoAdapterModel.from_pretrained(self.model_name_or_path, cache_dir=self.training_args.cache_dir,)


            
        if self.training_args.model_parallel_gpus > 1 and torch.cuda.device_count() > 1:
            if torch.cuda.device_count() != self.training_args.model_parallel_gpus:
                print(f"WARNING: model parallel is enabled but the number of GPUs does not match the number of GPUs specified in the model_parallel_gpus argument. Using all available GPUs. ({torch.cuda.device_count()} GPUs found)")
            if hasattr(self.model, "parallelize"):
                self.model.parallelize()
            else:
                print(f"Model {self.model_name_or_path} cannot be parallelized")

        self.model_trainable_params = sum(p.numel() for p in self.model.parameters())


        


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
        
        
        # multi_model_inputs_l = []# initalize as list of dict, finally merge dict into one dict
        # multi_model_inputs = {}
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
                return_tensors='pt', padding="max_length", 
                max_length=self.data_args.max_target_length,
                # padding=self.padding,
                truncation=True,
            )
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"][labels["input_ids"]==0] = -100
                # labels["input_ids"] = [
                #     [(l if l != self.tokenizer.pad_token_id else -100)
                #     for l in label]
                #     for label in labels["input_ids"]
                # ]
            
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


    def load_dataset(self, train, valid, test):
        """
        dataset loading pipeline:
        1. load dataset from huggingface datasets
        2. preprocess dataset
        3. tokenize dataset and have multi-model inputs like (input_ids_0, input_ids_1, input_ids_2, labels), also padding and convert to tensor
        4. dataloader with tokenizer inside, it requires tokenizer to provide padding token id
        
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

            if self.training_args.dev:
                raw_datasets["validation"] = raw_datasets["validation"].select(range(200))

            if train:
                self.train_dataset = raw_datasets["train"]
                self.eval_dataset = raw_datasets["validation"]
                self.test_dataset = raw_datasets["test"]
            else:
                self.test_dataset = raw_datasets["test"]
        else:
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
                # model=self.model,
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
            
            # trainer.model
            if self.model_args.model_arch == "encoder":
                self.model.add_classification_head(f"lm_head-{adapter_name}", num_labels=2, overwrite_ok=reset_peft)
            elif self.model_args.model_arch == "encoder-decoder":
                self.model.add_seq2seq_lm_head(f"lm_head-{adapter_name}", overwrite_ok=reset_peft)
                self.model.heads[f"lm_head-{adapter_name}"][0].weight.requires_grad = False

            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.model_args.model_arch}"
                )
            self.model.set_active_adapters(self.model_args.tuning_mode)
            
            # self.model.freeze_model(True)
    
        elif self.model_args.tuning_mode == "bitfit":
            # if self.model_args.model_arch == "encoder":
            #     # deactivate gradients except for bias terms
            #     BIAS_TERMS_DICT = {
            #         'intermediate': 'intermediate.dense.bias',
            #         'key': 'attention.self.key.bias',
            #         'query': 'attention.self.query.bias',
            #         'value': 'attention.self.value.bias',
            #         'output': 'output.dense.bias',
            #         'output_layernorm': 'output.LayerNorm.bias',
            #         'attention_layernorm': 'attention.output.LayerNorm.bias',
            #         'all': 'bias',
            #     }
            # elif self.model_args.model_arch == "encoder-decoder":
            #     BIAS_TERMS_DICT = {
            #         'intermediate': 'intermediate.dense.bias',
            #         'key': 'attention.self.key.bias',
            #         'query': 'attention.self.query.bias',
            #         'value': 'attention.self.value.bias',
            #         'output': 'output.dense.bias',
            #         'output_layernorm': 'output.LayerNorm.bias',
            #         'attention_layernorm': 'attention.output.LayerNorm.bias',
            #         'all': 'bias',
            #     }
            # def convert_to_actual_components(components):
            #     return [BIAS_TERMS_DICT[component] for component in components]
            # is_bias_init = False
            # for name, module in self.model.named_modules():
            #     if hasattr(module, "bias") and "lm_head" not in name:
            #         if module.bias is None:
            #             print("found none bias, init bias for ", name)
            #             module.bias = torch.nn.Parameter(torch.randn(module.out_features))
            #             is_bias_init = True
            #         if not module.bias.requires_grad:
            #             module.bias.requires_grad = True
                        
            # assert is_bias_init == True, "bias should be initialized"
            
            # raise NotImplementedError("bitfit is not computed for trainable paramters yet")
            # components = ["intermediate", "key", "query", "value", "output", "output_layernorm", "attention_layernorm", "all"]
            # trainable_components = convert_to_actual_components(components)
            # self._deactivate_relevant_gradients(trainable_components)
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
            # NOTE: test enable bias 
            # for name, module in self.model.named_modules():
            #     if  "lm_head" in name:
            #         module.bias.requires_grad = True
            for l in layers:
                for name, module in l.named_modules():
                    # if "selfattention" in name.lower():
                    # if hasattr(module, "bias") and type(module) == transformers.adapters.lora.Linear:
                    #     module.bias.requires_grad = True
                    #     print("activate gradient for ", name)
                    #     # print("name: ", name, " type: ", type(module))
                        
                    # first check if bias is settable
                    if hasattr(module, "bias") and type(module) == transformers.adapters.lora.Linear:
                        if module.bias is None:
                            print("found none bias, init bias for ", name)
                            module.bias = torch.nn.Parameter(torch.randn(module.out_features))
                            is_bias_init = True
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
        
        
        return self.check_trainable_parameters()



    def check_trainable_parameters(self, print_params_required_grad = False):
        # total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print_params_required_grad = True
        if print_params_required_grad:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    print(n,p.data.shape)
        print(f"Model Params: {self.model_trainable_params}, Trainable Params: {trainable_params}, Trainable Ratio: {trainable_params/self.model_trainable_params}")

        return trainable_params/self.model_trainable_params




    def train(self):
        """
        0. load pretrained model or from checkpoint
        1. load hf trainer and move model to device
        2. load peft module and move it to model device
        3. load training components such as data_collator, dataset and optimizer.
        4. start training.
        5. save the best model.
        5. evaluate the best model on test set
        
        Plus,
        - support resume training
        """

        self.load_dataset(train = True, valid = True, test=True)
        # 4. load optimizer
        self.load_optimizer_n_scheduler()
        
        from accelerate import Accelerator
        from accelerate.tracking import WandBTracker, GeneralTracker
        accelerator = Accelerator(
            log_with="wandb",
            logging_dir=self.training_args.output_dir,
            project_dir="./accelerate",
            gradient_accumulation_steps = self.training_args.gradient_accumulation_steps,
            )
        lr = self.scheduler.get_last_lr()[0]
        config = {
            "num_iterations": 5,
            "learning_rate": lr,
        }
        accelerator.init_trackers("huggingface",config=config)

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=16, collate_fn=self.data_collator)

        # if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
        #    self._load_rng_state(resume_from_checkpoint)
        self.model, self.optimizer, self.train_dataloader, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.scheduler)
        device = accelerator.device
        # get scheduler lr
        # wandb_tracker = accelerator.get_tracker("wandb")
        i = 0
        
        # save training arguments
        self.training_args.save_to_json_file(os.path.join(self.training_args.output_dir, "training_args.json"))
        
        
        # steps/epoches 
        assert self.training_args.num_train_epochs is not None, "num_train_epochs is not set"
        assert self.training_args.max_steps == -1, "max_steps is not supported yet, but got {}".format(self.training_args.max_steps)
        import math
        len_dataloader = len(self.train_dataloader)
        num_update_steps_per_epoch = len_dataloader // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(self.training_args.num_train_epochs)
        num_train_samples = len(self.train_dataset) * self.training_args.num_train_epochs
        tmp_dir = self.training_args.output_dir + "_test"
        # accelerator.save_state(self.training_args.output_dir)
        if os.path.exists(tmp_dir):
            print("load from existing state")
            accelerator.load_state(tmp_dir)

        # epochs_trained = self.state.global_step // num_update_steps_per_epoch
        for epoch in range(epochs_trained, num_train_epochs):
            for step, inputs in enumerate(self.train_dataloader):
                with accelerator.accumulate(self.model):
                    i += 1
                    if i > 5:
                        break
                    
                    # move inputs to device
                    # inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    loss = outputs["loss"]
                    accelerator.backward(loss) # it does gradient acc internally
                    
                    accelerator.log({"training_loss": loss.item()}, step=step)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # wandb_tracker.log({"training_loss": loss}, step=step)
                    print("step: ", step)
            accelerator.save_state(tmp_dir)
        accelerator.end_training()
    
    def evaluate(self):
        """
        currently only supports evaluation on test set.
        """
        
        
        self.load_dataset(train = False, valid = False, test=True)
        from accelerate import Accelerator
        accelerator = Accelerator(
            log_with="wandb",
            logging_dir=self.training_args.output_dir,
            project_dir="./accelerate",
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, shuffle=True,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=self.data_collator
        )
        self.model, self.test_dataloader = accelerator.prepare(self.model, self.test_dataloader)
        self.model_cache = deepcopy(self.model)
        
        tmp_dir = self.training_args.output_dir + "_test"
        # accelerator.save_state(self.training_args.output_dir)
        assert os.path.exists(tmp_dir), "It's expected to have dir for accelerator to load state"
        print("load from existing state: ", tmp_dir)
        accelerator.load_state(tmp_dir)
        self.model.eval()
        i = 0

        stop = 5
        output_host = []
        with torch.no_grad():
            for step, inputs in enumerate(self.test_dataloader):
                with accelerator.accumulate(self.model):
                    i += 1
                    if i > stop:
                        break
                    # move inputs to device
                    # inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    labels = inputs.pop("labels")
                    outputs = self.model.generate(**inputs)
                    output_host.append(outputs)

        output_host = torch.cat(output_host, dim=0).to("cpu")
        labels = None
        results = self.compute_metrics(
            (output_host, labels),
            self.test_dataset.select(
                range(stop*self.training_args.per_device_eval_batch_size)
            )
        )
        return results
    
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
            from util.compute_metrics import compute_metrics, compute_grouped_metrics
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            references = [e["Instance"]["output"] for e in eval_dataset]
            for pred, ref in zip(decoded_preds[:5], references[:5]):
                print("pred: ", pred, "ref: ", ref)

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

        # import pdb; pdb.set_trace()
        # print('break point for bias activiation')
        
            
    def _format_prompts(self, prefix, source_strs, verbal_targets, include_labels_in_input = False):
        prompts = [""] * len(source_strs)
        prompts = [""]* len(source_strs)
        formatted_inputs = [f"{prefix} {s_1} {prompt} " for s_1, prompt in zip(source_strs, prompts)]
        formatted_inputs = [f"{input} " for input in formatted_inputs]

        if include_labels_in_input:
            labels = ",".join(verbal_targets)
            formatted_inputs = [f"{input} Decide the label in {self.verbalizers}." for input in formatted_inputs]    

        return formatted_inputs, verbal_targets
