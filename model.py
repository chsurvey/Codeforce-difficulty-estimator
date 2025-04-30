import torch
import torch.nn as nn
from transformers import AutoModel

from config import cfg

class CrossAttention(nn.Module):
    """Lightweight cross‑modal attention using CLS queries."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    def forward(self, q, k, v):
        # q: (B, 1, D); k,v: (B, L, D)
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)  # (B, D)

class CBERT(nn.Module):
    """C‑BERT architecture coupling BERT and CodeBERT."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.text_backbone = AutoModel.from_pretrained(cfg.model_text)
        self.code_backbone = AutoModel.from_pretrained(cfg.model_code)

        d_text = self.text_backbone.config.hidden_size
        d_code = self.code_backbone.config.hidden_size
        assert d_text == d_code, "Backbones must share hidden dim"
        dim = d_text

        self.cross_t2c = CrossAttention(dim)
        self.cross_c2t = CrossAttention(dim)

        # Explicit feature dimension (4 as specified in the paper)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_labels),
        )
    def forward(self, text_inputs, code_inputs, features):
        # Encode each modality
        t_outputs = self.text_backbone(**{k: v.squeeze(1) for k, v in text_inputs.items()})
        c_outputs = self.code_backbone(**{k: v.squeeze(1) for k, v in code_inputs.items()})
        t_hidden = t_outputs.last_hidden_state  # (B, L_t, D)
        c_hidden = c_outputs.last_hidden_state  # (B, L_c, D)

        # CLS tokens (index 0)
        t_cls = t_hidden[:, 0:1, :]
        c_cls = c_hidden[:, 0:1, :]

        # Cross‑modal interaction (using CLS queries)
        t_cross = self.cross_t2c(t_cls, c_hidden, c_hidden)  # (B, D)
        c_cross = self.cross_c2t(c_cls, t_hidden, t_hidden)  # (B, D)

        # Global averages for each modality
        t_avg = t_hidden.mean(dim=1)
        c_avg = c_hidden.mean(dim=1)

        x = torch.cat([t_avg + t_cross, c_avg + c_cross, features], dim=-1)
        logits = self.fc(x)
        return logits