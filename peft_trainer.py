from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Model, T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModelForSequenceClassification, Seq2SeqAdapterTrainer

from arguments import TrainerArguments
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
from utils import get_embedding_layer, get_soft_prompt_token_list, get_all_params, round_up
import transformers
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptTuningConfig,PrefixTuningConfig
from util.ni_dataset_collator import DataCollatorForNI
from copy import deepcopy
import datetime

class PEFTTrainer:
    def __init__(self, arguments):
        self.model_names_or_path = arguments.model_names_or_path
        
        self.arguments = arguments
        
        self.verbalizers = self.get_dataset_verbalizers(self.arguments.dataset_name if self.arguments.dataset_name != "super_glue" else self.arguments.dataset_config_name)
        
        self.models = []
        # self.tokenizers = []
        # self.configs = []
        self.num_soft_tokens = arguments.num_soft_tokens
        self.load_model()
        self.org_vocab_size = self.tokenizer.vocab_size
        # assert len(self.models) == len(self.tokenizers) == len(self.configs)

        self.new_vocab_sizes = [self.org_vocab_size]
        # temp set
        # self.model = self.models[0]
        self.model_cache = deepcopy(self.model)
        self.default_optimizer_n_scheduler = self.arguments.default_optimizer_n_scheduler
        """
        Model is loaded, now we need to set up the trainer
        1. prepare peft model
        2. set up trainer
        """
        task_type = TaskType.SEQ_2_SEQ_LM
        init_from_text = False
        if self.arguments.dataset_name == "sst2" and self.arguments.model_arch == "encoder":
            task_type = TaskType.SEQ_CLS
            prompt_tuning_init = None # NOTE: it decides whether prompt init is used
            prompt_tuning_init_text = None
            init_text_tokenizer_name_or_path = None
            prompt_tuning_init = "TEXT"
            print("task type is seq_cls")
        prompt_tuning_init_text=" ".join(self.verbalizers)
        init_text_tokenizer_name_or_path = self.model_names_or_path

        if arguments.mode == "prompt_tuning":
            cur_prompt_len = self.num_soft_tokens
            assert self.arguments.trainable_params_percentage is not None or self.arguments.prompt_len > 0, "either prompt_len or trainable_params_percentage should be set"
            if self.arguments.trainable_params_percentage is not None:
                config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=cur_prompt_len,
                    inference_mode=False,
                    device= self.arguments.device,
                    
                    
                    
                    # prompt_tuning_init="TEXT",
                    # prompt_tuning_init_text=prompt_tuning_init_text,
                    # tokenizer_name_or_path=init_text_tokenizer_name_or_path,

                )
                cur_trainable_params_percentage = self.convert_to_peft(config)
            while abs(cur_trainable_params_percentage - self.arguments.trainable_params_percentage) > 0.00001:
                if cur_trainable_params_percentage > self.arguments.trainable_params_percentage:
                    cur_prompt_len -= 1
                else:
                    cur_prompt_len += 1
                config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=cur_prompt_len,
                    inference_mode=False,
                    device= self.arguments.device,
                    
                    
                    # prompt_tuning_init="TEXT",
                    # prompt_tuning_init_text=prompt_tuning_init_text,
                    # tokenizer_name_or_path=init_text_tokenizer_name_or_path,
                )
                cur_trainable_params_percentage = self.convert_to_peft(config, reset_peft=True)
            
                print("prompt length is {}".format(cur_prompt_len))
                print("trainable params percentage is {}".format(cur_trainable_params_percentage))
            self.arguments.run_name += "_prompt_len_{}".format(cur_prompt_len)
        elif arguments.mode == "prefix_tuning":
            from transformers.adapters import PrefixTuningConfig
            from peft import PrefixTuningConfig
            
            cur_prefix_len = 100 if self.arguments.prefix_len is None else self.arguments.prefix_len
            assert self.arguments.trainable_params_percentage is not None or self.arguments.prefix_len > 0, "either prefix_len or trainable_params_percentage should be set"
            if self.arguments.trainable_params_percentage is not None:
                config = PrefixTuningConfig(prefix_length=cur_prefix_len, bottleneck_size=512)
                cur_trainable_params_percentage = self.convert_to_peft(config)
            while abs(cur_trainable_params_percentage - self.arguments.trainable_params_percentage) > 0.00005:
                if cur_trainable_params_percentage > self.arguments.trainable_params_percentage:
                    cur_prefix_len -= 1
                else:
                    cur_prefix_len += 1
                config = PrefixTuningConfig(prefix_length=cur_prefix_len, bottleneck_size=512)
                cur_trainable_params_percentage = self.convert_to_peft(config, reset_peft=True)
                print("prefix length is {}".format(cur_prefix_len))
            self.arguments.run_name += "_prefix_len_{}".format(cur_prefix_len)
        
        elif arguments.mode == "lora":
            cur_lora_r = 32 if self.arguments.lora_r is None else self.arguments.lora_r
            assert self.arguments.trainable_params_percentage is not None or self.arguments.lora_r > 0, "either lora_r or trainable_params_percentage should be set"
            if self.arguments.trainable_params_percentage is not None:
                config = LoraConfig(
                    task_type=task_type,
                    inference_mode=False,
                    r=cur_lora_r,
                    lora_alpha=32,
                    lora_dropout=0.1
                )
                cur_trainable_params_percentage = self.convert_to_peft(config, reset_peft=True)

            while abs(cur_trainable_params_percentage - self.arguments.trainable_params_percentage) > 0.0001:
                if cur_trainable_params_percentage > self.arguments.trainable_params_percentage:
                    cur_lora_r -= 1
                else:
                    cur_lora_r += 1
                config = LoraConfig(
                    task_type=task_type,
                    inference_mode=False,
                    r=cur_lora_r,
                    lora_alpha=32,
                    lora_dropout=0.1
                )
                cur_trainable_params_percentage = self.convert_to_peft(config, reset_peft=True)
                print("cur_lora_r", cur_lora_r, "cur_trainable_params_percentage", cur_trainable_params_percentage)
            self.arguments.lora_r = cur_lora_r
            self.arguments.run_name += "_lora_r_" + str(cur_lora_r)
        elif arguments.mode in ["adapter", "compactor"]:
            cur_reduction_factor = self.arguments.reduction_factor if self.arguments.reduction_factor is not None else 32
            assert self.arguments.trainable_params_percentage is not None or self.arguments.reduction_factor > 0, "either reduction_factor or trainable_params_percentage should be set"
            
            if self.arguments.trainable_params_percentage is not None:
                from transformers.adapters import AdapterConfig, HoulsbyConfig
                # check existing adapter and remove them
                
                # config = AdapterConfig()
                config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
                cur_trainable_params_percentage = self.convert_to_peft(config)
            while abs(cur_trainable_params_percentage - self.arguments.trainable_params_percentage) > 0.00002:
                if cur_trainable_params_percentage > self.arguments.trainable_params_percentage:
                    cur_reduction_factor += 1
                else:
                    cur_reduction_factor -= 1
                config = HoulsbyConfig(reduction_factor=cur_reduction_factor)
                cur_trainable_params_percentage = self.convert_to_peft(config, reset_peft=True)
                print(f"cur_trainable_params_percentage: {cur_trainable_params_percentage}, cur_reduction_factor: {cur_reduction_factor}")
            self.arguments.run_name += f"_reduction_factor_{cur_reduction_factor}"
        elif arguments.mode == "embedding_tuning":
            self.convert_to_embedding_tuning()
            if self.num_soft_tokens > 0:
                self.num_soft_tokens = 0
                print("num_soft_tokens is set to 0 for embedding tuning mode")
        elif arguments.mode == "bitfit":
            self.convert_to_peft()
        elif arguments.mode == "fine_tuning":
            pass
        elif arguments.mode == "layer_tuning":
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            layers = []
            # NOTE: we only fine-tune attention weights for now
            if arguments.layer_name == "first_encoder_layer":
                layers.append(self.model.encoder.block[0].layer[0])
            elif arguments.layer_name == "last_encoder_layer":
                layers.append(self.model.encoder.block[-1].layer[0])
            elif arguments.layer_name == "first_decoder_layer":
                layers.append(self.model.decoder.block[0].layer[0])
            elif arguments.layer_name == "last_decoder_layer":
                layers.append(self.model.decoder.block[-1].layer[0])
            elif arguments.layer_name == "custom":
                # all decoder layer
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif arguments.layer_name == "custom2":
                # all decoder layer
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            else:
                raise NotImplementedError(f"layer_name {arguments.layer_name} is not implemented")
            
            for l in layers:
                for name, module in l.named_modules():
                    # if "selfattention" in name.lower():
                    if hasattr(module, "weight"):
                        module.weight.requires_grad = True
                        print("activate gradient for ", name)
            
        else:
            raise NotImplementedError(f"mode {arguments.mode} is not implemented")
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model = self.model_cache
        self.arguments.run_name += f"_{time}"
        self.set_up_hf_trainer()
        self.tokenizer = self.tokenizer
        

    def set_up_hf_trainer(self):
        del self.model_cache
        if self.arguments.model_parallel_gpus > 1 and torch.cuda.device_count() > 1:
            if torch.cuda.device_count() != self.arguments.model_parallel_gpus:
                print(f"WARNING: model parallel is enabled but the number of GPUs does not match the number of GPUs specified in the model_parallel_gpus argument. Using all available GPUs. ({torch.cuda.device_count()} GPUs found)")
            if hasattr(m, "parallelize"):
                self.model_cache.parallelize()
            else:
                print(f"Model {self.model_names_or_path} cannot be parallelized")
        optimizer = Adafactor(
            self.model.parameters(),
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            lr= self.arguments.learning_rate,
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

        lr_scheduler = get_constant_schedule(optimizer)

        if self.arguments.dataset_name == "ni":
            dataset_dependent_data_collator = DataCollatorForNI(
                self.tokenizer,
                model=self.model,
                padding="max_length" if self.arguments.pad_to_max_length else "longest",
                max_source_length=self.arguments.max_source_length,
                max_target_length=self.arguments.max_target_length,
                label_pad_token_id=self.tokenizer.pad_token_id,
                pad_to_multiple_of=8 if self.arguments.fp16 else None,
                add_task_name=self.arguments.add_task_name,
                add_task_definition=self.arguments.add_task_definition,
                num_pos_examples=self.arguments.num_pos_examples,
                num_neg_examples=self.arguments.num_neg_examples,
                add_explanation=self.arguments.add_explanation,
                tk_instruct=self.arguments.tk_instruct
            )
            self.arguments.remove_unused_columns = False
        else:
            dataset_dependent_data_collator = default_data_collator

        if self.arguments.mode in ["adapter",  "compactor"]: # "prefix_tuning",
            self.trainer = Seq2SeqAdapterTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = None,
                eval_dataset = None,
                args = self.arguments,
                optimizers=[optimizer, lr_scheduler] if not self.default_optimizer_n_scheduler else [None, None],
                compute_metrics=partial(self.compute_metrics, is_pred_logits = not self.arguments.predict_with_generate),
                data_collator=dataset_dependent_data_collator,
            )
        else:

            self.trainer = Seq2SeqTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = None,
                eval_dataset = None,
                args = self.arguments,
                optimizers=[optimizer, lr_scheduler] if not self.default_optimizer_n_scheduler else [None, None],
                compute_metrics=partial(self.compute_metrics, is_pred_logits = not self.arguments.predict_with_generate),
                data_collator=dataset_dependent_data_collator,
            )
        self.trainer.model = self.model
        
    def _format_prompts(self, prefix, source_strs, verbal_targets, include_labels_in_input = False):
        prompts = [""] * len(source_strs)
        prompts = [""]* len(source_strs)
        formatted_inputs = [f"{prefix} {s_1} {prompt} " for s_1, prompt in zip(source_strs, prompts)]
        formatted_inputs = [f"{input} " for input in formatted_inputs]

        if include_labels_in_input:
            labels = ",".join(verbal_targets)
            formatted_inputs = [f"{input} Decide the label in {self.verbalizers}." for input in formatted_inputs]    

        return formatted_inputs, verbal_targets

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

    
    def load_model(self):
        print(
            "Loading",
            self.arguments.model_names_or_path,
            "(for large models, this might take a while)",
        )
        print("Files will be cached at:", self.arguments.cache_dir)
        print(
            "Ensure this directory is persistent if you do not want to download model files again!"
        )
        
        # for self.model_names_or_path in self.model_names_or_path:
            
        # self.config = AutoConfig.from_pretrained(
        #     self.model_names_or_path,
        #     cache_dir=self.arguments.cache_dir,
        #     gradient_checkpointing=self.arguments.gradient_checkpointing,
        #     use_cache=not self.arguments.gradient_checkpointing,
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_names_or_path,
            cache_dir=self.arguments.cache_dir,
            # use_cache = self.arguments.use_cache,
            use_fast=True,
            return_tensors="pt"
        )
        # m = T5ForConditionalGeneration.from_pretrained(
        #     self.model_names_or_path,
        # )
        if "t5" in self.model_names_or_path or "bart" in self.model_names_or_path:
            
            # import vanilla T5 model
            
            # m_config = T5Config.from_pretrained(self.model_names_or_path)
            # import pdb; pdb.set_trace()
            # print('check config')
            
            # m =T5Model.from_pretrained("t5-small")
            # torch.zeros(linear_layer.out_features)
            
            
            
            # m.encoder.block[0].layer[0].SelfAttention.q.weight.bias
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_names_or_path, cache_dir=self.arguments.cache_dir,)

            # m = T5PT.from_pretrained(
            #     self.model_names_or_path,
            #     # from_tf=bool(".ckpt" in self.model_names_or_path),
            #     # config=m_config,
            #     cache_dir=self.arguments.cache_dir,
            # )
            
        elif "roberta" in self.model_names_or_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_names_or_path)
        elif "gpt2" in self.model_names_or_path or "bloom" in self.model_names_or_path or "opt" in self.model_names_or_path:
            from transformers import AutoModelForCausalLM
            self.model_cache = AutoModelForCausalLM.from_pretrained(
                self.model_names_or_path,
                # from_tf=bool(".ckpt" in self.model_names_or_path),
                # config=m_config,
                cache_dir=self.arguments.cache_dir,
            )
        else:
            raise NotImplementedError("Model not supported: " + self.model_names_or_path)
        # Wrap model in adapter package
        # NOTE: temp implementation
        if self.arguments.mode in ["adapter", "compactor"] : # "prefix_tuning", 
            self.model = AutoAdapterModel.from_pretrained(self.model_names_or_path, cache_dir=self.arguments.cache_dir,)

        if self.tokenizer.pad_token is None:
            assert self.model_names_or_path == "gpt2", "Only gpt2 is expected not having pad tokens for now"
            # gpt2 model
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            m.resize_token_embeddings(len(self.tokenizer))
        
        
        self.padding = "max_length" if self.arguments.pad_to_max_length else False
        print('gpt2 requires padding to max length')
        if "gpt2" in self.model_names_or_path:
            self.padding = "max_length"
        # self.config = m_config
        # self.tokenizer = m_tokenizer
            # self.models.append(m)
            # self.configs.append(m_config)
            # self.tokenizers.append(m_tokenizer)
            


    def preprocess(self, examples, class_ids = [0,1], evaluation=False):
        # disable prefix prompt
        # prefix_prompt = "".join(get_soft_prompt_token_list(self.num_soft_tokens))
        prefix_prompt = ""
        if self.num_soft_tokens ==0:
            assert prefix_prompt == ""
        if self.arguments.model_arch == "encoder":
            add_prompt_for_gen = False
        else:
            add_prompt_for_gen = True
        if self.arguments.dataset_name == "sst2":
            prompt_for_gen = "Sentiment:"
            inputs =["Sentence: " + sent for sent, label_id in zip(examples["sentence"], examples["label"]) if label_id in class_ids]
            
            # verbalize the sentiment id to tokens
            # it's not used for t5 evaluation though (label id is used instead)
            verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
        elif self.arguments.dataset_name == "yelp_review_full":
            prompt_for_gen = "give the review score from 1-5 stars:"
            inputs =["Sentence: " + sent for sent, label_id in zip(examples["text"], examples["label"]) if label_id in class_ids]
            verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            
        elif self.arguments.dataset_name == "super_glue":
            prompt_for_gen = "Answer:"
            if self.arguments.dataset_config_name in['axb', 'axg'] :
                raise NotImplementedError("axb and axg are not implemented yet")
                inputs = ["Premise: " + premise + " Hypothesis: " + hypothesis + "Given the premise, is the hypothesis correct? Answer: " for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])]
                verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            elif self.arguments.dataset_config_name in ["boolq", "multirc"]:
                if self.arguments.dataset_config_name == "boolq":
                    inputs = ["Question: " + question + "Passage: " + passage for question, passage, label_id in zip(examples["question"], examples["passage"], examples["label"]) if label_id in class_ids]
                elif self.arguments.dataset_config_name == "multirc":
                    inputs = ["Question: " + question + "Paragraph: " + paragraph  for question, paragraph, label_id in zip(examples["question"], examples["paragraph"], examples["label"]) if label_id in class_ids]
                    # inputs = ["" for question, paragraph in zip(examples["question"], examples["paragraph"])]
                # inputs = ["Passage: " + passage + " Question: " + question for question, passage in zip(examples["question"], examples["passage"])]
                verbal_targets = [self.verbalizers[l]
                    for l in examples["label"] if l in class_ids]
            elif self.arguments.dataset_config_name in ["wic"]:
                prompt_for_gen = "" # NOTE: wic has a special format
                inputs = ["Sentence1: " + sentence1 + " Sentence2: " + sentence2 + f" Question: does the word '{word}' have the same meaning in the two sentences? Answer: " for sentence1, sentence2, word, label_id in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["label"])  if label_id in class_ids]
                verbal_targets = [self.verbalizers[l] for l, label_id in zip(examples["label"], examples["label"])  if label_id in class_ids]
        elif self.arguments.dataset_name in ["trec"]:
            prompt_for_gen = " What's the type of the question? "
            inputs = ["Question: " + t for t, label_id in zip(examples["text"],examples["coarse_label"]) if label_id in class_ids]
            verbal_targets = [self.verbalizers[l] for l, label_id in zip(examples["coarse_label"], examples["coarse_label"])  if label_id in class_ids]
        else:
            
            
            raise NotImplementedError("Dataset not supported: " + self.arguments.dataset_name)
        if add_prompt_for_gen:
                inputs = [inp + " " + prompt_for_gen for inp in inputs]
        formatted_inputs, verbal_targets =\
                self._format_prompts(prefix_prompt,
                                    inputs,
                                    # [self.prefix_prompt]*len(inputs),
                                    verbal_targets,
                                    self.arguments.include_labels_in_input,
        )
        print("Sample input: ", formatted_inputs[0])

                        
        def add_idx_to_inputs_keys(model_inputs, model_idx):
            # add model_idx to the keys of model_inputs
            # to avoid key conflict when merging model_inputs into one dict
            model_inputs = {f"{key}_{model_idx}": value for key, value in model_inputs.items() if key != "class_ids"}
            return model_inputs

        multi_model_inputs_l = []# initalize as list of dict, finally merge dict into one dict
        multi_model_inputs = {}

        for model_idx in range(len(self.models)):
            tokenizer = self.tokenizers[model_idx]
            model_inputs = tokenizer(
                formatted_inputs,
                max_length=self.arguments.max_source_length,
                padding=self.padding,
                truncation=True,
                return_tensors='pt'
            )
            
            # build label input ids
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    verbal_targets,
                    return_tensors='pt', padding="max_length", 
                    max_length=self.arguments.max_target_length,
                    # padding=self.padding,
                    truncation=True,
                )
            
            if self.padding == "max_length" and self.arguments.ignore_pad_token_for_loss:
                labels["input_ids"][labels["input_ids"]==0] = -100
                # labels["input_ids"] = [
                #     [(l if l != self.tokenizer.pad_token_id else -100)
                #     for l in label]
                #     for label in labels["input_ids"]
                # ]
            
            # if encoder only model
            if self.arguments.model_arch == "encoder":
                # for SequenceClassificationModel
                # model_inputs["label"] = [l for l in examples["label"] if l in class_ids]
                model_inputs["labels"] = [l for l in examples["label"] if l in class_ids]
            else:
                # model_inputs["label"] = labels["input_ids"]
                model_inputs["labels"] = labels["input_ids"]
            
            # if evaluation:
            #     model_inputs["class_ids"] = torch.tensor(class_ids)
            multi_model_inputs[model_idx] = model_inputs
            # model_inputs = add_idx_to_inputs_keys(model_inputs, model_idx)
            
            # model_inputs
            multi_model_inputs_l.append(model_inputs)
        multi_model_inputs_d = {}
        for model_inputs in multi_model_inputs_l:
            multi_model_inputs_d.update(model_inputs)
        if evaluation:
            label_name = "label" if self.arguments.dataset_name not in ["trec"] else "coarse_label"
            label_class_ids = [l for l in examples[label_name] if l in class_ids]
            multi_model_inputs_d["class_ids"] = torch.tensor(label_class_ids)
        
        return multi_model_inputs_d

    def load_dataset(self, train, valid):
        """
        dataset loading pipeline:
        1. load dataset from huggingface datasets
        2. preprocess dataset
        3. tokenize dataset and have multi-model inputs like (input_ids_0, input_ids_1, input_ids_2, labels), also padding and convert to tensor
        4. dataloader with tokenizer inside, it requires tokenizer to provide padding token id
        5. 
        
        """
        if self.arguments.dataset_name == "ni":
            assert self.arguments.task_dir is not None, "task_dir is required for NaturalInstructions dataset"
            assert self.arguments.data_dir is not None, "data_dir is required for NaturalInstructions dataset"
            # Get the NaturalInstructions dataset
            raw_datasets = load_dataset(
                "util/ni_dataset.py", 
                data_dir=self.arguments.data_dir, 
                task_dir=self.arguments.task_dir, 
                cache_dir=self.arguments.cache_dir,
                max_num_instances_per_task=self.arguments.max_num_instances_per_task,
                max_num_instances_per_eval_task=self.arguments.max_num_instances_per_eval_task,
                download_mode = "reuse_dataset_if_exists" if not self.arguments.overwrite_cache else "force_redownload",
            )
            
            if self.arguments.dev:
                raw_datasets["validation"] = raw_datasets["validation"].select(range(10))
            self.trainer.train_dataset = raw_datasets["train"]
            self.trainer.eval_dataset = raw_datasets["validation"]
            self.eval_dataset = raw_datasets["validation"]
        else:
            raw_datasets = load_dataset(self.arguments.dataset_name , self.arguments.dataset_config_name)
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

                self.trainer.train_dataset = self.train_dataset

            if valid:
                if self.arguments.dataset_name in ["yelp_review_full", "ag_news", "trec"]:
                    valid_split_name = "test"
                else:
                    valid_split_name = "validation"
                
                if self.arguments.dev:

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
                self.trainer.eval_dataset = self.eval_dataset


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
        


    def convert_to_peft(self, peft_config=None, reset_peft=False):
        """
        1. prepare peft model
        2. set up trainer

        Args:
            peft_config (_type_): _description_
        """
        
        if self.arguments.mode in ["adapter", "compactor"]: # prefix_tuning
            
            # add and activate adapter
            self.model.add_adapter("sst-2", config = peft_config, overwrite_ok=reset_peft)
            self.model.train_adapter("sst-2")
            if self.arguments.model_arch == "encoder":
                self.model.add_classification_head("classification-head-sst-2", num_labels=2, overwrite_ok=reset_peft)
            elif self.arguments.model_arch == "encoder-decoder":
                # self.model.add_seq2seq_lm_head("seq2seq-head-sst-2", overwrite_ok=reset_peft)
                pass
            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.arguments.model_arch}"
                )
            self.model.set_active_adapters("sst-2")
            # self.model.freeze_model(True)
        elif self.arguments.mode == "bitfit":
            # if self.arguments.model_arch == "encoder":
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
            # elif self.arguments.model_arch == "encoder-decoder":
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
            if self.arguments.bias_name == "encoder_bias":
                modules = self.model.encoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif self.arguments.bias_name == "decoder_bias":
                modules = self.model.decoder.block
                for m in modules:
                    layers.append(m.layer[0])
            elif  self.arguments.bias_name == "encoder_decoder_bias":
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
            # lm head takes almost 10% paramaters, so we don't want to train it
            # for name, module in self.model.named_modules():
            #     if "lm_head" in name:
            #         module.weight.requires_grad = True

        else:
            # NOTE: prompt tuning
            # general peft converting based on different peft config
            assert peft_config is not None, "peft config should be provided for non-adapter peft method"
            # convert text to token ids
            verbalizer_ids = [self.tokenizer.encode(v) for v in self.verbalizers]
            
            if reset_peft:
                # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_names_or_path, cache_dir=self.arguments.cache_dir)
                self.model = deepcopy(self.model_cache)
            # add tokens in models and tokenizers + freeze model
            self.model.enable_input_require_grads()
            
            self.model = get_peft_model(self.model, peft_config)
        
        
        return self.check_trainable_parameters()



    def check_trainable_parameters(self, print_params_required_grad = False):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print_params_required_grad = True
        if print_params_required_grad:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    print(p.data.shape, n)
        print(f"Total Params: {total_params}, Trainable Params: {trainable_params}, Trainable Ratio: {trainable_params/total_params}")

        return trainable_params/total_params


    def train(self):
        # set the trainer logging level to warning
        self.load_dataset(train = True, valid = True)
        self.trainer.train()
    
    def evaluate(self, eval_dataset=None):
        if eval_dataset is None:
            if self.eval_dataset is None:
                self.load_dataset(train = False, valid = True)
            eval_dataset = self.eval_dataset
        
        max_verbalizer_len = max([len(v) for v in self.verbalizers])
        if "bloom" in self.arguments.model_names_or_path or "opt" in self.arguments.model_names_or_path:
            eval_preds = []
            correct = 0
            for data in eval_dataset:
                out = self.model.generate(data["input_ids_0"].unsqueeze(0).cuda())
                decoded_out = self.tokenizer.decode(out[0]).lower()
                print(decoded_out)
                decoded_out = decoded_out[-max_verbalizer_len:]
                if self.verbalizers[0].strip().lower() in decoded_out.strip().lower() or "negative" in decoded_out or "no" in decoded_out:
                    pred = 0
                elif self.verbalizers[1].strip().lower() in decoded_out.strip().lower() or "positive" in decoded_out or "yes" in decoded_out:
                    pred = 1
                else:
                    print("missed prediction: ", decoded_out)
                    pred = -1
                if pred == data["class_ids"]:
                    correct += 1
            results = {"acc": correct/len(eval_dataset)}
        else:
            results = self.trainer.evaluate(eval_dataset=eval_dataset)
        print(results)
        return results

    
    
    def compute_metrics(self, eval_preds, is_pred_logits = False, model_idx = 0, metrics = {}):
        eval_dataset = self.eval_dataset
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        def postprocess_text(preds, labels):
            preds = [pred.strip().lower() for pred in preds]
            labels = [[label.strip().lower()] for label in labels]
            return preds, labels
        result = metrics
        
        if self.arguments.dataset_name == "ni":
            save_prefix = None
            from util.compute_metrics import compute_metrics, compute_grouped_metrics
            dataset = self.eval_dataset
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = [e["Instance"]["output"] for e in dataset]
            for pred, ref in zip(decoded_preds[:5], references[:5]):
                print("pred: ", pred, "ref: ", ref)

            result = compute_metrics(predictions=decoded_preds, references=references)
            result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
            result.update(result_per_task)
            categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
            result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
            result.update(result_per_category)
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result
        # import pdb; pdb.set_trace()
        # print('check why acc is so low')
        
        
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
            
            if self.arguments.model_arch == "encoder":
                import evaluate
                accuracy = evaluate.load("accuracy")
                print("preds: ", preds)
                preds = np.argmax(preds, axis=1)
                
                return accuracy.compute(predictions=preds, references=labels)
            
            encoder = model.get_encoder()

            dataloader = DataLoader(eval_dataset,
                                    # collate_fn=collate_fn,
                                    batch_size=self.arguments.per_device_eval_batch_size)
            
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
                    # out = model(input_ids = input_ids.cuda(), attention_mask=attention_mask.cuda(), decoder_input_ids=decoder_input_ids)
                    out = model(input_ids = input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda())
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
