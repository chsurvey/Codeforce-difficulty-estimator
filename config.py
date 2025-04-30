import os, torch

class CFG:
    model_text = "bert-base-uncased"
    model_code = "microsoft/codebert-base"
    max_len_text = 512
    max_len_code = 512
    num_labels = 3

    lr = 2e-5
    weight_decay = 1e-2
    batch_size = 8
    epochs = 10
    warmup_ratio = 0.1

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = "runs"

cfg = CFG()