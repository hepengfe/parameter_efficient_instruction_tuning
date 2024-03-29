import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose
# code below import ACT2FN
from transformers.activations import ACT2FN
from transformers.models.opt.modeling_opt import  OPTAttention
from typing import Optional, Tuple
from transformers.activations import get_activation


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

from transformers.activations import get_activation


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)
@dataclass
class AdapterConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        adapter_size (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    adapter_size: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    model_config: Optional[dict] = field(
        default=None,
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTER


class AdapterModel(torch.nn.Module):
    """
    pretrained model wrapper to add adapters
    """
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace() # replace layers with adapters
        # mark_only_adapter_as_trainable(self.model, self.peft_config.bias)
        mark_only_adapter_as_trainable(self.model)
        
        self.forward = self.model.forward
    
    
    def _find_and_replace(self):
        # replace layers with adapters that have customized forward pass
        key_list = [key for key, _ in self.model.named_modules()]
        is_target_modules_in_base_model = False
        # for k, v in self.model.named_modules():
        #     print(k,v )

        for key in key_list:
            # print(key)
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                from .adapters import OPTAdapterDecoderLayer, LlamaAdapterDecoderLayer, T5AdapterLayerSelfAttention,  T5AdapterLayerCrossAttention, T5AdapterLayerFF
                    
                for layer_idx, layer in enumerate(target.children()):
                    if "t5" in type(self.model).__name__.lower():
                        if key.startswith("encoder"):
                            # replace encoder layer
                            # layer.0
                            new_ff = T5AdapterLayerSelfAttention(layer.layer[0], self.peft_config)
                            new_self_attn = T5AdapterLayerFF(layer.layer[1], self.peft_config)
                            layer.layer[0] = new_ff
                            layer.layer[1] = new_self_attn
                            # import pdb; pdb.set_trace()
                            # print('encoder layer check')
                            
                        elif key.startswith("decoder"):
                            new_ff = T5AdapterLayerSelfAttention(layer.layer[0], self.peft_config)
                            # new_cross_attn = T5AdapterLayerCrossAttention(layer.layer[1], self.peft_config)
                            new_self_attn = T5AdapterLayerFF(layer.layer[2], self.peft_config)
                            layer.layer[0] = new_ff
                            # layer.layer[1] = new_cross_attn
                            layer.layer[2] = new_self_attn
                            # import pdb; pdb.set_trace()
                            # print('decoder layer check')
                            
                    elif "opt" in type(self.model).__name__.lower():
                        new_module = OPTAdapterDecoderLayer(layer, self.peft_config)
                        
                        
                        target[layer_idx] = new_module
                    elif "llama" in type(self.model).__name__.lower():
                        new_module = LlamaAdapterDecoderLayer(layer, self.peft_config)
                        target[layer_idx] = new_module
        
                #     # new_module = OPTAdapterDecoderLayer(layer, self.peft_config)
                #     new_module = LlamaAdapterDecoderLayer(layer, self.peft_config)
                #     target[layer_idx] = new_module

                    # setattr(target, )

                # import pdb; pdb.set_trace()
                # print('')
                                
                # print(parent, target, target_name)
                # new_module = OPTAdapterDecoderLayer(target)
                # self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        
        
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
    
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config


class AdapterLayer(nn.Module):
    """
    adapter wrapper to configure adapter and forward function.
    """
    def __init__(self, location_key: str, config):
        super().__init__()
        self.location_key = location_key
        self.config = config
        
        self.adapter = Adapter("adapter", self.config["hidden_size"], self.config["adapter_size"])

    # def _init_adapter_modules(self):
    #     self.adapters = nn.ModuleDict(dict())
        # self.adapter_fusion_layer = nn.ModuleDict(dict())


    def forward(self, hidden_states, residual_input, layer_norm):
        adapter_layer = self.adapter
        input_hidden_states = hidden_states
        hidden_states, _, residual = adapter_layer.pre_forward(hidden_states, residual_input, layer_norm)
        layer_output = adapter_layer(
            hidden_states, residual_input=residual, 
        )
        hidden_states, up = layer_output[0], layer_output[2]
        hidden_states = adapter_layer.post_forward(hidden_states, input_hidden_states, residual_input, layer_norm)
        return hidden_states
        # # Batch sizes might be different due to prefix tuning w. Parallel block
        # (residual_input,) = adjust_tensors_for_parallel(hidden_states, residual_input)
        # # Replicate in both directions as residual might be larger (e.g. GPT-J)
        # (hidden_states,) = adjust_tensors_for_parallel(residual_input, hidden_states)
        # adapter_setup = self.get_active_setup(self.adapters)
        # import pdb; pdb.set_trace()
        # print('check adapter_setup in adapter_layer_forward')
        # if adapter_setup is not None:

        #     # if isinstance(adapter_setup, Stack):
        #     # hidden_states, _, residual_input = self.adapter_stack(
        #     #     adapter_setup, hidden_states, residual_input, layer_norm
        #     # )
            
        #     # adapter_layer = self.adapters["adapter"]
        #     adapter_layer = self.adapter
        #     hidden_states, _, residual = adapter_layer.pre_forward(hidden_states, residual_input, layer_norm)
        #     layer_output = adapter_layer(
        #         hidden_states, residual_input=residual, 
        #     )
        #     hidden_states, up = layer_output[0], layer_output[2]
        #     return hidden_states

        # elif layer_norm:
        #     hidden_states = layer_norm(hidden_states + residual_input)
        # else:
        #     hidden_states = hidden_states + residual_input

        # return hidden_states
    

    # def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor, layer_norm, lvl=0):
    #     """
    #     Forwards the given input through the given stack of adapters.
    #     """
    #     for i, adapter_stack_layer in enumerate(adapter_setup):
    #         adapter_layer = self.adapters[adapter_stack_layer]
    #         hidden_states, _, residual = adapter_layer.pre_forward(hidden_states, input_tensor, layer_norm)
    #         context = ForwardContext.get_context()
    #         layer_output = adapter_layer(
    #             hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores
    #         )
    #         hidden_states, up = layer_output[0], layer_output[2]
    #         self._store_gating_score(adapter_stack_layer, layer_output[-1])
    #         # as this stack might be part of a fusion block, return the adapter up-projection output here
    #         # together with the final output (with potential residuals & norms) if we reached the last block of the stack
    #         if i == len(adapter_setup) - 1:
    #             return hidden_states, up, input_tensor
    
class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    """

    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample,
        # config: AdapterConfig,
    ):
        super().__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.add_layer_norm_before = False
        self.add_layer_norm_after = False
        self.adapter_residual_before_ln = False
        self.use_gating = False

        # Params related to input & output of adapter
        self.residual_before_ln = True
        self.original_ln_before = False
        self.original_ln_after = True

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        # if self.add_layer_norm_before:
        #     self.adapter_norm_before = nn.LayerNorm(self.input_size)
        #     seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # seq_list.append(nn.Linear(self.input_size, self.down_sample))
        
        # # select non-linearity
        self.non_linearity = Activation_Function_Class("silu")

        
        # seq_list.append(self.non_linearity)

        # # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # # residual connection
        # self.adapter_down = nn.Sequential(*seq_list)

        # self.adapter_down = nn.Linear(self.input_size, self.down_sample)
        seq_list = []
        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)


        self.scaling = 1.0
        # # Additional scaling factor (from He et al. (2021))
        # if isinstance(config["scaling"], float):
        #     self.scaling = config["scaling"]
        # elif config["scaling"] == "learned":
        #     self.scaling = nn.Parameter(torch.ones(1))
        # else:
        #     raise ValueError("Unknown scaling type: {}".format(config["scaling"]))

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if self.use_gating:
        #     self.gate = nn.Linear(self.input_size, 1)
        self.adapter_down.apply(self.init_bert_weights)
        self.adapter_up.apply(self.init_bert_weights)
        # if self.use_gating:
        #     self.gate.apply(self.init_bert_weights)
        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        # if config["init_weights"] == "bert":
        #     self.adapter_down.apply(self.init_bert_weights)
        #     self.adapter_up.apply(self.init_bert_weights)
        #     if self.use_gating:
        #         self.gate.apply(self.init_bert_weights)
        # elif config["init_weights"] == "mam_adapter":
        #     with torch.no_grad():
        #         nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
        #         nn.init.zeros_(self.adapter_up.weight)
        #         nn.init.zeros_(self.adapter_down[0].bias)
        #         nn.init.zeros_(self.adapter_up.bias)
        #         if self.use_gating:
        #             self.gate.apply(self.init_bert_weights)
        # else:
        #     raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))


    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if self.original_ln_before:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)
        down= self.non_linearity(down)
        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()





# had to adapt it for `lora_only` to work
def mark_only_adapter_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        
        if "adapters" not in n:
            p.requires_grad = False
        