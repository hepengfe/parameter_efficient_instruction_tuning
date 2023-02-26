import ctypes
from re import X
import torch
from transformers import TrainerCallback


def round_up(x, to=8):
    x, to = int(x), int(to)
    return int((x + to - 1) & (-1 * to))


def increase_l2_fetch_granularity():
    # see:
    # https://www.tensorflow.org/guide/profiler#max_out_the_l2_cache
    # "A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes)."
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))


def generate_new_embedding(embedding, initialize_method = "random",):
    """
    Get new embedding from embedding.
    """
    weight = embedding.weight
    if initialize_method == "random":
        new_embedding = torch.rand_like(embedding)
    elif initialize_method == "vocab":
        new_embedding = embedding
    else:
        raise ValueError(f"Initialize method {initialize_method} not supported")
    return new_embedding



def get_embedding_layer(model, mode):
    """
    Get embedding layer from a specific model.
    """
    if mode == "t5_seq2seq":
        if "BartForConditionalGeneration" in str(type(model)):
            embedding_layer =  model.model.shared
        else:
            embedding_layer =  model.shared
    elif mode == "clm":
        embedding_layer = model.transformer.wte
    else:
        raise ValueError(f"Mode {mode} not supported")
    return embedding_layer


def set_embedding_layer(embedding_weight, model, mode):
    """
    Get embedding layer from model.
    """
    if mode == "t5_seq2seq":
        embedding_layer =  model.shared
    elif mode == "clm":
        embedding_layer = model.transformer.wte
        # embedding_layer = model.transformer.pte
    else:
        raise ValueError(f"Mode {mode} not supported")
    return embedding_layer


def get_soft_prompt_token_list(num_soft_prompt_tokens):
    """
    Get soft prompt tokens from vocab.
    """
    return [f"<|softprompt{i}|>" for i in range(num_soft_prompt_tokens)]


class L2FetchCallback(TrainerCallback):
    "Set cudaLimitMaxL2FetchGranularity at beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        increase_l2_fetch_granularity()


def get_all_params(models, filter_requires_grad = True):
    if filter_requires_grad:
        return [p for m in models for p in m.parameters() if p.requires_grad]
    else:
        return [p for m in models for p in m.parameters()]