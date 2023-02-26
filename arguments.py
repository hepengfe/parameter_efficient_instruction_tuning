import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainerArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="./saved_model/")
    eval_data_preprocess: bool = False
    metric: Optional[str] = field(default="sacrebleu")
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default="~/cache/")
    model_names_or_paths: str = field(default="t5-small")
    pad_to_max_length: Optional[bool] = field(default=False)
    max_source_length: Optional[int] = field(default=128)
    max_target_length: Optional[int] = field(default=128)
    ignore_pad_token_for_loss: Optional[bool] = field(default=True)
    preprocessing_num_workers: Optional[int] = field(default=1)
    overwrite_cache: Optional[bool] = field(default=False)
    pad_to_multiple_of: Optional[int] = field(default=4)
    predict_with_generate: Optional[bool] = field(default=True)
    resume_from_checkpoint: Optional[str] = field(default=None)
    model_parallel_gpus: Optional[int] = field(default=1)
    logging_strategy: Optional[str] = field(default="epoch")
    save_strategy: Optional[str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=2)
    mode: Optional[str] = field(default=None)
    optimizer: Optional[torch.optim.Optimizer] = field(default=None)
    torch_ort: Optional[bool] = field(default=False)
    embedding_on_cpu: Optional[bool] = field(default=False)
    deepspeed: Optional[str] = field(default=None)
    dataset_cache: Optional[bool] = field(default=True)
    callbacks: Optional[list] = field(default_factory=list)
    warmup_steps: Optional[int] = field(default=500)
    optimizer_name: Optional[str] = field(default="adam")
    verbalize_labels: Optional[bool] = field(default=True)
    prompt_init_method: Optional[str] = field(default="verbalizer")
    num_soft_tokens: Optional[int] = field(default=0)
    dev: Optional[bool] = field(default=False)
    default_optimizer_n_scheduler: Optional[bool] = field(default=True)
    include_labels_in_input: Optional[bool] = field(default=False)
    lr: Optional[float] = field(default=0.3)