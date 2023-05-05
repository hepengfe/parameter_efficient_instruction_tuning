import os
import re
import shutil

import logging

logger = logging.getLogger(__name__)

def build_peft_config_name(model_args, peft_args, training_args):
    # peft config
    peft_config_name = ""
    if model_args.tuning_mode in ["lora", "lora_peft", "lora_adapter"]:
        peft_config_name += "r_" + str(peft_args.lora_r) + "_alpha_" + str(peft_args.lora_alpha)
        peft_config_name += "_module_" + str(peft_args.lora_modules)
    elif model_args.tuning_mode == "ia3":
        peft_config_name += "r_" + str(peft_args.lora_r)
    elif model_args.tuning_mode == "prompt_tuning":
        peft_config_name +=  "_prompt_len_{}" + str(peft_args.num_soft_tokens)
    elif model_args.tuning_mode == "prefix_tuning":
        peft_config_name += "prefix_len_" + str(peft_args.prefix_len)
    elif model_args.tuning_mode == "layer_tuning":
        peft_config_name += "layer_name_" + str(peft_args.layer_name)
    elif model_args.tuning_mode == "bitfit":
        peft_config_name += "bias_name_" + str(peft_args.bias_name)
    elif model_args.tuning_mode == "adapter":
        peft_config_name += "_adapter_size_" + str(peft_args.adapter_size)
    elif model_args.tuning_mode == "compactor":
        peft_config_name +=f"_reduction_factor_{peft_args.reduction_factor:.4f}"
        # phm_dimension
        peft_config_name += "_phm_dimension_" + str(peft_args.phm_dimension)
    elif model_args.tuning_mode == "parallel_adapter":
        peft_config_name +=f"_reduction_factor_{peft_args.reduction_factor:.4f}"
    elif model_args.tuning_mode == "fine_tuning":
        pass
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
    checkpoint_dirs = [d for d in os.listdir(output_dir) if re.match(r'^checkpoint-\d+$', d)]
    if not checkpoint_dirs:
        return None
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    return os.path.join(output_dir, latest_checkpoint)

def remove_old_checkpoints(output_dir, num_to_keep=1):
    checkpoint_dirs = [d for d in os.listdir(output_dir) if re.match(r'^checkpoint-\d+$', d)]
    if len(checkpoint_dirs) <= num_to_keep:
        return
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    for d in checkpoint_dirs[:-num_to_keep]:
        logger.info(f"Removing old checkpoint {os.path.join(output_dir, d)}")
        shutil.rmtree(os.path.join(output_dir, d))
