import os, torch

class CFG:
    # base models
    model_text = "bert-base-uncased"
    model_code = "microsoft/codebert-base"
    max_len_text = 512
    max_len_code = 512
    num_labels = 3

    # optimization
    lr = 2e-5
    weight_decay = 1e-2
    batch_size = 8
    epochs = 10
    warmup_ratio = 0.1

    # runtime
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = "runs"
    
    # MoE toggles
    use_moe           = True      # master on/off switch
    moe_num_experts   = 8         # 4 / 8 / 16 …
    moe_top_k         = 2         # 1 = Switch-Transformer, 2 = standard MoE
    moe_loss_weight   = 0.01      # λ for load-balancing aux loss
                                   # (makes total loss = CE + λ·moe_loss)
cfg = CFG()