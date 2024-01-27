import os
import re
import shutil
import torch
import logging
import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def build_peft_config_name(model_args, peft_args, training_args):
    # peft config
    peft_config_name = ""
    if model_args.tuning_mode in ["lora", "lora_peft", "lora_adapter"]:
        peft_config_name += "r_" + str(peft_args.lora_r) + "_alpha_" + str(peft_args.lora_alpha)
        peft_config_name += "module_" + str(peft_args.lora_modules)
    elif model_args.tuning_mode == "ia3":
        peft_config_name += "r_" + str(peft_args.lora_r)
    elif model_args.tuning_mode == "prompt_tuning":
        peft_config_name +=  "prompt_len_" + str(peft_args.prompt_len)
    elif model_args.tuning_mode == "prefix_tuning":
        peft_config_name += "prefix_len_" + str(peft_args.prefix_len) + "_bottleneck_size_" + str(peft_args.bottleneck_size)
    elif model_args.tuning_mode == "layer_tuning":
        peft_config_name += "layer_name_" + str(peft_args.layer_name)
    elif model_args.tuning_mode == "bitfit":
        peft_config_name += "bias_name_" + str(peft_args.bias_name)
    elif model_args.tuning_mode == "adapter_adapter":
        peft_config_name += "adapter_size_" + str(peft_args.adapter_size)
    elif model_args.tuning_mode == "adapter_peft":
        peft_config_name += "adapter_size_" + str(peft_args.adapter_size)
    elif model_args.tuning_mode == "compactor":
        peft_config_name +=f"_reduction_factor_{peft_args.reduction_factor:.4f}"
        # phm_dimension
        peft_config_name += "phm_dimension_" + str(peft_args.phm_dimension)
    elif model_args.tuning_mode == "parallel_adapter":
        peft_config_name +=f"reduction_factor_{peft_args.reduction_factor:.4f}"
    elif model_args.tuning_mode  == "fine_tuning":
        pass
    elif model_args.tuning_mode ==  "off_the_shelf":
        peft_config_name += model_args.tuning_mode
    elif model_args.tuning_mode == "pelt":
        peft_config_name += "use_pelt_gate_" + str(peft_args.use_pelt_gate)
        raise NotImplementedError("Should be more configs for pelt")
    else:
        raise NotImplementedError(f"tuning mode {model_args.tuning_mode} is not implemented")

    
    # lr
    peft_config_name += "_lr_" + str(training_args.learning_rate)
    # precision
    # peft_config_name += "_bf16_" + str(training_args.bf16)
    
    # effective batch size
    peft_config_name += "_bs_" + str(training_args.per_device_train_batch_size)
    peft_config_name += "_grad_acc_" + str(training_args.gradient_accumulation_steps)

    
    return peft_config_name

def flatten(s, source_char="/", flatten_char="_"):
    """
    source_char can be multiple characters, and all of them will be flattened to flatten_char.
    """
    for sc in source_char:
        s = s.replace(sc, flatten_char)
    return s


def get_latest_checkpoint(output_dir):
    try:
        checkpoint_dirs = [d for d in os.listdir(output_dir) if re.match(r'^checkpoint-\d+$', d)]
    except FileNotFoundError:
        print(f"output_dir {output_dir} does not exist. Error captured in get_latest_checkpoint")
        return None
    if not checkpoint_dirs:
        return None
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    return os.path.join(output_dir, latest_checkpoint)


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        decoder_input_ids = torch.zeros( (batch_input_ids.shape[0], 1), dtype=torch.long).to(batch_input_ids.device)

        batch_logits = model(input_ids=batch_input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


def remove_old_checkpoints(output_dir, num_to_keep=1):
    checkpoint_dirs = [d for d in os.listdir(output_dir) if re.match(r'^checkpoint-\d+$', d)]
    if len(checkpoint_dirs) <= num_to_keep:
        return
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    if num_to_keep == 0:
        logger.info(f"Removing all old checkpoints in {output_dir}")
        checkpoint_dirs_to_remove = checkpoint_dirs
    else:
        logger.info(f"Removing {len(checkpoint_dirs) - num_to_keep} old checkpoints in {output_dir}")
        checkpoint_dirs_to_remove = checkpoint_dirs[:-num_to_keep]
    for d in checkpoint_dirs_to_remove:
        logger.info(f"Removing old checkpoint {os.path.join(output_dir, d)}")
        shutil.rmtree(os.path.join(output_dir, d))

def remove_files_and_folders_other_than(output_dir, file_or_folder_name):
    files = os.listdir(output_dir)
    for f in files:
        if f != file_or_folder_name:
            if os.path.isfile(os.path.join(output_dir, f)):
                os.remove(os.path.join(output_dir, f))
            else:
                shutil.rmtree(os.path.join(output_dir, f))
    # resume training -> check latest checkpoint
    # if latest checkpoint exists -> load it and continue training
    # training is finished but no eval -> load latest checkpoint and do eval
    # how to signaling that training is finished in a minimal storage way? 
    # keep latest step checkpoint's train_state file

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    alpaca dataset format.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
choices = ["A", "B", "C", "D"]

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1, k_shot=5):
    prompts = []
    for i in range(0, test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k_shot)
        prompt = train_prompt + prompt_end
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while tokenized_prompt.shape[-1] > 2048:
            k_shot -= 1
            train_prompt = gen_prompt(dev_df, subject, k_shot)
            prompt = train_prompt + prompt_end
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        use_chat_format = True
        if use_chat_format:
            prompt = "<|user|>\n" + prompt.strip() + "\n<|assistant|>\nThe answer is:"
            
        prompts.append(prompt)

    # get the answer for all examples
    # note: here we cannot directly use convert_tokens_to_ids because the some tokenizers will automatically add space prefix.
    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[0] for answer_choice in choices]

    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def verify_complete_random_states(cp_dir):
    # check if 8 random states is in the checkpoint dir
    for i in range(8):
        if not os.path.exists(os.path.join(cp_dir, f"random_states_{i}.pkl")):
            print(f"random_states_{i}.pkl is not in {cp_dir}")
            return False
    return True

def check_all_checkpoints_and_remove(proj_dir):
    for root, dirs, files in os.walk(proj_dir):
        for d in dirs:
            if d.startswith("checkpoint"):
                cp_dir = os.path.join(root, d)
                if not verify_complete_random_states(cp_dir):
                    print(f"checkpoint {cp_dir} random states is not complete, removing it")
                    shutil.rmtree(cp_dir)
