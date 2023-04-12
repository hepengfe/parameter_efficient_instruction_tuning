def build_peft_config_name(model_args, peft_args, training_args):
        peft_config_name = ""
        if model_args.tuning_mode == "lora":
            peft_config_name += "r_" + str(peft_args.lora_r)
            peft_config_name += "_module_" + str(peft_args.lora_modules)
        elif model_args.tuning_mode == "prefix_tuning":
            peft_config_name += "prefix_len_" + str(peft_args.prefix_len)
        elif model_args.tuning_mode == "layer_tuning":
            peft_config_name += "layer_name_" + str(peft_args.layer_name)
        elif model_args.tuning_mode == "bitfit":
            peft_config_name += "bias_name_" + str(peft_args.bias_name)
        elif model_args.tuning_mode == "adaptor":
            peft_config_name += "reduction_factor_" + str(peft_args.reduction_factor)
        elif model_args.tuning_mode == "compactor":
            peft_config_name += "reduction_factor_" + str(peft_args.reduction_factor)
            # phm_dimension
            peft_config_name += "_phm_dimension_" + str(peft_args.phm_dimension)
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
        peft_config_name += "_bf16_" + str(training_args.bf16)
        # trainable_params_percentage is None, use preset config 
        if peft_args.trainable_params_percentage is not None:
            peft_config_name += "_trainable_params_percentage_" + str(peft_args.trainable_params_percentage)
        return peft_config_name

def flatten(s, source_char="/", flatten_char="_"):
    """
    source_char can be multiple characters, and all of them will be flattened to flatten_char.
    """
    for sc in source_char:
        s = s.replace(sc, flatten_char)
    return s