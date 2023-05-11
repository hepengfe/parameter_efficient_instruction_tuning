import logging
import random
import string
from transformers.data.data_collator import *
from transformers import (
    # OPTPreTrainedModel,
    GPT2PreTrainedModel,
    LlamaPreTrainedModel
)
logger = logging.getLogger(__name__)
import numpy as np
import torch

@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False


    def __call__(self, batch, eval_mode=False, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        sources = []
        extra_model_inputs = {}
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            if isinstance(instance["Definition"], list):
                extra_model_inputs["task_definition"] = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
            else:
                extra_model_inputs["task_definition"] = "Definition: " + instance["Definition"].strip()

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 
            
            source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        model_inputs = {}
        # 1. prepare labels first in str format
        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            text_labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            # if self.text_only:
            #     labels = text_labels
            # else:
            #     with self.tokenizer.as_target_tokenizer():
            #         labels = self.tokenizer(
            #             text_labels,
            #             max_length=self.max_target_length,
            #             padding=self.padding,
            #             return_tensors=self.return_tensors,
            #             truncation=True,
            #             pad_to_multiple_of=self.pad_to_multiple_of
            #         )
            #     label_mask = labels["attention_mask"].bool()
            #     labels = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            text_labels = []
            labels = None
    
        is_causal_lm =  isinstance(self.model, GPT2PreTrainedModel) or isinstance(self.model, LlamaPreTrainedModel)
        
        # if is_causal_lm and labels:
        #     sources = ["".join(sl) for sl in zip(sources, labels)]

        # 2. prepare model inputs first
        if not is_causal_lm:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    text_labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            labels = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            if self.text_only:
                model_inputs = {"inputs": sources}
                model_inputs["labels"] = text_labels
            else:
                model_inputs = self.tokenizer(
                        sources, 
                        max_length=self.max_source_length, 
                        padding=self.padding,
                        return_tensors=self.return_tensors, 
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                )
                model_inputs["labels"] = labels



            # the following code is useful if label smoothing is used
            # it causes label leakage in eval mode
            # one way is to build decoder input ids only in train mode
            
            # if self.model is not None and not self.text_only and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            #     model_inputs["decoder_input_ids"] = decoder_input_ids

            return model_inputs
    
        else:
            example_texts = []
            self.tokenizer.padding_side = "left"
            assert len(sources) == len(text_labels), f"len(sources) = {len(sources)}, len(text_labels) = {len(text_labels)}"
 
            for s, l in zip(sources, text_labels):
                if not eval_mode:
                    example_texts.append(s + "</SEP>" + l)
                else:
                    example_texts.append(s)

            tokenized_example = self.tokenizer(
                example_texts,
                max_length=self.max_source_length,
                padding="longest", # no padding for causal lm
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            input_ids = tokenized_example.input_ids
            

            # print("input_ids.shape at where ", input_ids.shape)
            # print("input_ids == self.tokenizer.sep_token_id: ", input_ids == self.tokenizer.sep_token_id)
            if not eval_mode:
                labels = input_ids.clone()
                seq_indices = torch.where(input_ids == self.tokenizer.sep_token_id)

                assert len(seq_indices[1]) == len(text_labels), f"number of <\sep> is more than expected in the text. len(seq_indices[1]) = {len(seq_indices[1])}, len(text_labels) = {len(text_labels)}"
                
                
                # e.g. [ remove 3 from the number 2, 3, 1 <sep> ] 2, 1 ]
                # input_ids [ <pad> <pad> remove 3 from the number 2, 3, 1 <sep> ] 2, 1 ]]
                # set labels to -100 before sep token under training (padding during evaluation)
                # (tensor([0, 1]), tensor([122, 113]))

                # for bs_index, seq_index in enumerate(seq_indices[1]):
                #     labels[bs_index, :seq_index] = -100
                # mask out prompt in the example text
                
                # print("input_ids.shape: ", input_ids.shape)
                # print("seq_indices shape: ", seq_indices[0].shape, seq_indices[1].shape)
                # print("torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1): ", torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1).shape)
                # print("seq_indices[1].unsqueeze(1): ", seq_indices[1].unsqueeze(1).shape)
                prompt_mask = torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1) > seq_indices[1].unsqueeze(1)
                labels = labels.masked_fill(~prompt_mask, -100)
            else:
                # labels = labels.masked_fill(~prompt_mask, self.tokenizer.pad_token_id)
                
                labels = self.tokenizer(
                    text_labels, 
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_target_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,)["input_ids"]
                

            
            tokenized_example["labels"] = labels
            
            return tokenized_example
            
                
                

        