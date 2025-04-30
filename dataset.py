import json, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import cfg

class PojDataset(Dataset):
    """JSONL loader returning tokenised statement, code, numeric features, label."""

    def __init__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(l) for l in f]
        self.text_tok = AutoTokenizer.from_pretrained(cfg.model_text)
        self.code_tok = AutoTokenizer.from_pretrained(cfg.model_code)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t = self.text_tok(
            s["statement"], truncation=True, max_length=cfg.max_len_text,
            padding="max_length", return_tensors="pt"
        )
        c = self.code_tok(
            s["code"], truncation=True, max_length=cfg.max_len_code,
            padding="max_length", return_tensors="pt"
        )
        feat = torch.tensor(s["features"], dtype=torch.float)
        lab = torch.tensor(s["label"], dtype=torch.long)
        return t, c, feat, lab
