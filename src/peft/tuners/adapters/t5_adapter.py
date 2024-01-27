
from ..adapter import AdapterLayer
import torch
from typing import Optional, Tuple
import torch.nn as nn

class T5AdapterLayerFF(nn.Module):
    def __init__(self, org_layer, peft_config):
        super().__init__()
        self.org_layer = org_layer
        self.peft_config = peft_config


        self.config = {
            "hidden_size": self.peft_config.model_config.d_model, # self.org_layer.DenseReluDense.wi_0.weight.shape[1],
            "adapter_size": self.peft_config.adapter_size,
        }
        self.output_adapters = AdapterLayer("output_adapter", self.config)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = self.output_adapters(
            hidden_states=self.dropout(forwarded_states), residual_input=hidden_states, layer_norm=None
        )
        return hidden_states

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.org_layer, name)

class T5AdapterLayerSelfAttention(nn.Module):
    def __init__(self, org_layer, peft_config):
        super().__init__()
        
        self.org_layer = org_layer
        self.peft_config = peft_config
        # self.org_layer.SelfAttention.key_value_proj_dim
        self.config = {
            "hidden_size":self.peft_config.model_config.d_model,
            "adapter_size": self.peft_config.adapter_size,
        }
        
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.org_layer, name)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.attention_adapters(
            hidden_states=self.dropout(attention_output[0]), residual_input=hidden_states, layer_norm=None
        )
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5AdapterLayerCrossAttention(nn.Module):
    def __init__(self, org_layer, peft_config):
        super().__init__()
        self.org_layer = org_layer
        self.peft_config = peft_config
        # self.org_layer.EncDecAttention.q.weight.shape[1],
        self.peft_config.model_config.d_model
        self.config = {
            "hidden_size": self.peft_config.model_config.d_model,
            "adapter_size": self.peft_config.adapter_size,
        }
        self.attention_adapters = AdapterLayer("cross_adapter", self.config)
        
    
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = self.attention_adapters(
            hidden_states=self.dropout(attention_output[0]), residual_input=hidden_states, layer_norm=None
        )
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.org_layer, name)