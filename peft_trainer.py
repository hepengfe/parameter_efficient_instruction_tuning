from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModelForSequenceClassification, Seq2SeqAdapterTrainer

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

class PEFTTrainer:
    def __init__(self, arguments):
        self.model_names_or_paths = arguments.model_names_or_paths
        
        self.arguments = arguments
        
        self.verbalizers = self.get_dataset_verbalizers(self.arguments.dataset_name if self.arguments.dataset_name != "super_glue" else self.arguments.dataset_config_name)
        
        self.models = []
        self.tokenizers = []
        self.configs = []
        self.num_soft_tokens = arguments.num_soft_tokens
        self.load_models()
        self.org_vocab_size = self.tokenizers[0].vocab_size
        assert len(self.models) == len(self.tokenizers) == len(self.configs)

        self.new_vocab_sizes = [self.org_vocab_size]
        # temp set
        self.model = self.models[0]
        
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
            prompt_tuning_init_text="Predict sentiment review positive, negative"
            init_text_tokenizer_name_or_path = self.model_names_or_paths[0]
            print("task type is seq_cls")


        if arguments.mode == "prompt_tuning":
            # self.convert_to_prompt_tuning(self.num_soft_tokens)
            if init_from_text:
                peft_config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=self.num_soft_tokens, inference_mode=False, device= self.arguments.device,
                    prompt_tuning_init="TEXT",
                    prompt_tuning_init_text=prompt_tuning_init_text,
                    tokenizer_name_or_path=init_text_tokenizer_name_or_path,
                )
            else:
                peft_config = PromptTuningConfig(
                    task_type=task_type,
                    num_virtual_tokens=self.num_soft_tokens, 
                    inference_mode=False,
                    device= self.arguments.device,
                )
            assert self.num_soft_tokens > 0, "num_soft_tokens should be greater than 0 in prompt tuning mode"
            self.convert_to_peft(peft_config)
        elif arguments.mode == "prefix_tuning":
            # peft version
            # peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
            # self.convert_to_peft(peft_config)
            
            
            # adapter-prefix-tuning version
            self.convert_to_peft()
        
        elif arguments.mode == "lora":
            peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.convert_to_peft(peft_config)
        
        elif arguments.mode == "adapter":
            self.convert_to_peft()
        elif arguments.mode == "bitfit":
            self.convert_to_peft()
            
        elif arguments.mode == "embedding_tuning":
            self.convert_to_embedding_tuning()
            if self.num_soft_tokens > 0:
                self.num_soft_tokens = 0
                print("num_soft_tokens is set to 0 for embedding tuning mode")
        else:
            self.num_soft_tokens = 0
            self.set_up_hf_trainer()
        self.tokenizer = self.tokenizers[0]
        

        # self.set_up_hf_trainer()
        
        
        
        # optimizer = Adafactor(
        #     self.model.parameters(),
        #     # filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr= arguments.learning_rate,
        #     eps=(1e-30, 1e-3),
        #     clip_threshold=1.0,
        #     decay_rate=-0.8,
        #     beta1=None,
        #     weight_decay=1e-5,
        #     # fixed learning rate
        #     scale_parameter=False,
        #     relative_step=False,
        #     warmup_init=False,
        # )

        
        
        # lr_scheduler = get_constant_schedule(optimizer)
        
        
        # # load dataset when needed
        # self.train_dataset = None
        # self.eval_dataset = None # usually val dataset


        # if "t5" in self.arguments.model_names_or_paths[0]:
        #     # self.trainer = MultiModelTrainer(
        #     #     models = self.models,
        #     #     tokenizer = self.tokenizers[0],
        #     #     train_dataset = None,
        #     #     eval_dataset = None,
        #     #     # train_dataset = self.train_dataset,
        #     #     # eval_dataset = self.val_dataset,
        #     #     args = arguments,
        #     #     optimizers=[optimizer, lr_scheduler] if not default_optimizer_n_scheduler else [None, None],
        #     #     compute_metrics=partial(self.compute_metrics, is_pred_logits = not arguments.predict_with_generate),
        #     #     # data_collator=DataCollatorForSeq2Seq,
        #     #     verbalizer_info={'verbalizers': self.verbalizers,
        #     #                 'max_verbalizer_token_len': self.arguments.max_target_length
        #     #                 },
        #     # )
            
        #     self.trainer = Seq2SeqTrainer(
        #         model = self.models[0],
        #         tokenizer = self.tokenizers[0],
        #         train_dataset = None,
        #         eval_dataset = None,
        #         # train_dataset = self.train_dataset,
        #         # eval_dataset = self.val_dataset,
        #         args = arguments,
        #         optimizers=[optimizer, lr_scheduler] if not default_optimizer_n_scheduler else [None, None],
        #         compute_metrics=partial(self.compute_metrics, is_pred_logits = not arguments.predict_with_generate),
        #         data_collator=default_data_collator,
        #         # verbalizer_info={'verbalizers': self.verbalizers,
        #         #             'max_verbalizer_token_len': self.arguments.max_target_length
        #         #             },
        #     )
            
        # elif "bloom" in self.arguments.model_names_or_paths[0] or "opt" in self.arguments.model_names_or_paths[0]:
        #     # diff data collate function for decoder-only model
        #     self.trainer = Trainer(
        #         model = self.models[0],
        #         tokenizer = self.tokenizers[0],
        #         train_dataset = None,
        #         eval_dataset = None,
        #         # train_dataset = self.train_dataset,
        #         # eval_dataset = self.val_dataset,
        #         args = arguments,
        #         data_collator=default_data_collator,
        #         optimizers=[optimizer, lr_scheduler] if not default_optimizer_n_scheduler else [None, None],
        #         compute_metrics=partial(self.compute_metrics, is_pred_logits = not arguments.predict_with_generate),
        #     )
        # else:
        #     raise NotImplementedError("model type not supported", self.arguments.model_names_or_paths[0])
        # self.trainer.model = self.model
        
        
    def set_up_hf_trainer(self):
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

        if self.arguments.mode == "adapter":
            print('adapter trainer is used')
            self.trainer = Seq2SeqAdapterTrainer(
                model = self.model,
                tokenizer = self.tokenizers[0],
                train_dataset = None,
                eval_dataset = None,
                # train_dataset = self.train_dataset,
                # eval_dataset = self.val_dataset,
                args = self.arguments,
                optimizers=[optimizer, lr_scheduler] if not self.default_optimizer_n_scheduler else [None, None],
                compute_metrics=partial(self.compute_metrics, is_pred_logits = not self.arguments.predict_with_generate),
                data_collator=default_data_collator,
                # verbalizer_info={'verbalizers': self.verbalizers,
                #             'max_verbalizer_token_len': self.arguments.max_target_length
                #             },
            )
        else:
            self.trainer = Seq2SeqTrainer(
                model = self.model,
                tokenizer = self.tokenizers[0],
                train_dataset = None,
                eval_dataset = None,
                # train_dataset = self.train_dataset,
                # eval_dataset = self.val_dataset,
                args = self.arguments,
                optimizers=[optimizer, lr_scheduler] if not self.default_optimizer_n_scheduler else [None, None],
                compute_metrics=partial(self.compute_metrics, is_pred_logits = not self.arguments.predict_with_generate),
                data_collator=default_data_collator,
                # verbalizer_info={'verbalizers': self.verbalizers,
                #             'max_verbalizer_token_len': self.arguments.max_target_length
                #             },
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
        else:
            raise NotImplementedError("Dataset not supported: " + dataset)
        return verbalizers

    
    def load_models(self):
        print(
            "Loading",
            self.arguments.model_names_or_paths,
            "(for large models, this might take a while)",
        )
        print("Files will be cached at:", self.arguments.cache_dir)
        print(
            "Ensure this directory is persistent if you do not want to download model files again!"
        )
        
        for m_name_or_path in self.model_names_or_paths:
            from transformers import AutoModelForSeq2SeqLM, AutoAdapterModel
            m_config = AutoConfig.from_pretrained(
                m_name_or_path,
                cache_dir=self.arguments.cache_dir,
                gradient_checkpointing=self.arguments.gradient_checkpointing,
                use_cache=not self.arguments.gradient_checkpointing,
            )
            m_tokenizer = AutoTokenizer.from_pretrained(
                m_name_or_path,
                cache_dir=self.arguments.cache_dir,
                use_fast=True,
                return_tensors="pt"
            )
            # m = T5ForConditionalGeneration.from_pretrained(
            #     m_name_or_path,
            # )
            if "t5" in m_name_or_path or "bart" in m_name_or_path:
                m = AutoModelForSeq2SeqLM.from_pretrained(m_name_or_path, cache_dir=self.arguments.cache_dir,)
                # m = T5PT.from_pretrained(
                #     m_name_or_path,
                #     # from_tf=bool(".ckpt" in m_name_or_path),
                #     # config=m_config,
                #     cache_dir=self.arguments.cache_dir,
                # )
            elif "roberta" in m_name_or_path:
                m = AutoModelForSequenceClassification.from_pretrained(m_name_or_path)
            elif "gpt2" in m_name_or_path or "bloom" in m_name_or_path or "opt" in m_name_or_path:
                from transformers import AutoModelForCausalLM
                m = AutoModelForCausalLM.from_pretrained(
                    m_name_or_path,
                    # from_tf=bool(".ckpt" in m_name_or_path),
                    # config=m_config,
                    cache_dir=self.arguments.cache_dir,
                )
            else:
                raise NotImplementedError("Model not supported: " + m_name_or_path)

            # NOTE: temp implementation
            if self.arguments.mode == "adapter" or self.arguments.mode == "prefix_tuning":
                m = AutoAdapterModel.from_pretrained(m_name_or_path, cache_dir=self.arguments.cache_dir,)

            if m_tokenizer.pad_token is None:
                assert m_name_or_path == "gpt2", "Only gpt2 is expected not having pad tokens for now"
                # gpt2 model
                m_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                m.resize_token_embeddings(len(m_tokenizer))
            
            if self.arguments.model_parallel_gpus > 1 and torch.cuda.device_count() > 1:
                if torch.cuda.device_count() != self.arguments.model_parallel_gpus:
                    print(f"WARNING: model parallel is enabled but the number of GPUs does not match the number of GPUs specified in the model_parallel_gpus argument. Using all available GPUs. ({torch.cuda.device_count()} GPUs found)")
                if hasattr(m, "parallelize"):
                    m.parallelize()
                else:
                    print(f"Model {m_name_or_path} cannot be parallelized")
            self.padding = "max_length" if self.arguments.pad_to_max_length else False
            
            print('gpt2 requires padding to max length')
            if "gpt2" in m_name_or_path:
                self.padding = "max_length"
            self.models.append(m)
            self.configs.append(m_config)
            self.tokenizers.append(m_tokenizer)
            


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

            
            print("train dataset input sample", sample_input, "\n", self.tokenizers[0].decode(sample_input))
            print("train dataset label sample", sample_label,"\n", self.tokenizers[0].decode(sample_label))
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
        for name, param in self.model.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break


    def convert_to_peft(self, peft_config=None):
        """
        1. prepare peft model
        2. set up trainer

        Args:
            peft_config (_type_): _description_
        """
        
        if self.arguments.mode == "adapter":
            # add and activate adapter
            self.model.add_adapter("sst-2")
            self.model.train_adapter("sst-2")
            if self.arguments.model_arch == "encoder":
                self.model.add_classification_head("classification-head-sst-2", num_labels=2)
            elif self.arguments.model_arch == "encoder-decoder":
                self.model.add_seq2seq_lm_head("seq2seq-head-sst-2")
            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.arguments.model_arch}"
                )
            self.model.set_active_adapters("sst-2")
        elif self.arguments.mode == "prefix_tuning":
            # peft prefix tuning version has issue in encoder only model
            
            # adapter prefix-tuning version
            from transformers.adapters import PrefixTuningConfig
            config = PrefixTuningConfig(flat=True, prefix_length=30)
            self.model.add_adapter("sst-2", config=config)
            self.model.train_adapter("sst-2")
            if self.arguments.model_arch == "encoder":
                self.model.add_classification_head("sst-2", num_labels=2)
            elif self.arguments.model_arch == "encoder-decoder":
                self.model.add_seq2seq_lm_head("seq2seq-head-sst-2")
            else:
                raise NotImplementedError(
                    f"Not implemented for model arch: {self.arguments.model_arch}"
                )
            self.model.set_active_adapters("sst-2")
        elif self.arguments.mode == "bitfit":
            # deactivate gradients except for bias terms
            BIAS_TERMS_DICT = {
                'intermediate': 'intermediate.dense.bias',
                'key': 'attention.self.key.bias',
                'query': 'attention.self.query.bias',
                'value': 'attention.self.value.bias',
                'output': 'output.dense.bias',
                'output_layernorm': 'output.LayerNorm.bias',
                'attention_layernorm': 'attention.output.LayerNorm.bias',
                'all': 'bias',
            }
            def convert_to_actual_components(components):
                return [BIAS_TERMS_DICT[component] for component in components]
            assert "roberta" in self.arguments.model_names_or_paths[0], "bitfit only supports roberta model (other model might have different dictionary for bias?) "
            trainable_components = convert_to_actual_components(self.arguments.trainable_components)
            self._deactivate_relevant_gradients(trainable_components)

        else:
            # general peft converting based on different peft config
            assert peft_config is not None, "peft config should be provided for non-adapter peft method"
            # convert text to token ids
            verbalizer_ids = [self.tokenizers[0].encode(v) for v in self.verbalizers]
            
            # add tokens in models and tokenizers + freeze model
            # for idx, (m, t) in enumerate(zip(self.models, self.tokenizers)):
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            print("reset trainer after loading peft module")
        self.set_up_hf_trainer()




    def train(self):
        
        self.load_dataset(train = True, valid = True)
        self.trainer.train()
    
    def evaluate(self, eval_dataset=None):
        if eval_dataset is None:
            if self.eval_dataset is None:
                self.load_dataset(train = False, valid = True)
            eval_dataset = self.eval_dataset
        
        max_verbalizer_len = max([len(v) for v in self.verbalizers])
        if "bloom" in self.arguments.model_names_or_paths[0] or "opt" in self.arguments.model_names_or_paths[0]:
            eval_preds = []
            correct = 0
            for data in eval_dataset:
                out = self.model.generate(data["input_ids_0"].unsqueeze(0).cuda())
                decoded_out = self.tokenizers[0].decode(out[0]).lower()
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
                                    batch_size=4)
            
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
            result[f"acc_{model_idx}"] = correct / len(eval_dataset)

        return result
