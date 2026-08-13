try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None
import torch.nn as nn
import copy
import os
import re


def text_prompt(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Use a regular expression to find all text within <prompt> tags
            prompts = re.findall(r'<prompt>(.*?)</prompt>', content, re.DOTALL)
            # Remove whitespace within each prompt
        return prompts
    except FileNotFoundError:
        print("File not found.")
        return []
    except Exception as e:
        print("An error occurred:", e)
        return []

# Calculate the total number of parameters
def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents


def count_parameters(model, description=True):
    total_params = 0
    if description and PrettyTable is not None:
        table = PrettyTable(["Modules", "Parameters"])
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
    else:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()
    return total_params


def check_memory_usage(tensor):
    return tensor.element_size() * tensor.nelement() / 1024**2


def get_clones(module, num_layers=8):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


def get_available_gpu_ids():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible_devices.split(",")]
    else:
        gpu_ids = []  # Empty list means no GPUs available for training.

    return gpu_ids


def print_with_separator(text):
    separator = "# ---------------------------------------------------------------------------- #"
    text_length = len(text)
    padding_total = 78 - 2 - text_length  # Total padding available for the inner line, subtracting 2 for the `#` characters
    left_padding = padding_total // 2
    right_padding = padding_total - left_padding

    print("\033[94m" + separator)
    print("#" + " " * left_padding + text + " " * right_padding + "#")
    print(separator + "\033[0m")