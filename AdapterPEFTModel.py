from peft.peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from transformers import AutoModelForSeq2SeqLM, AutoAdapterModel

class AdapterPeftModelForSeq2Seq(AutoAdapterModel, PeftModel):
    def __init__(self, config, adapter_config=None):
        AutoAdapterModel.__init__(self, config, adapter_config)
        PeftModel.__init__(self, config)
