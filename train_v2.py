# DataParallel version: one process, multi-GPU via nn.DataParallel
# -------------------------------------------------------------
# 실행 예)   python train_dp.py   # CUDA_VISIBLE_DEVICES=0,1,2,3 가정
# -------------------------------------------------------------

import random, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from tqdm.auto import tqdm
from dataset import CodeContestsDataset

# -------------------- Hyper-parameters -------------------- #
BATCH          = 16   # *total* batch (DataParallel가 내부에서 분할)
EPOCHS         = 5
TXT_MAXLEN     = 512
CODE_MAXLEN    = 512
HIDDEN         = 768
PROJ_DIM       = 256
LR_ENCODER     = 2e-5
LR_OTHER       = 1e-4
GRAD_ACCUM     = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Tokenizer & Encoder ------------------ #

tok_txt  = AutoTokenizer.from_pretrained("bert-base-uncased")
enc_txt  = AutoModel.from_pretrained("bert-base-uncased")

tok_code = AutoTokenizer.from_pretrained("microsoft/codebert-base")
enc_code = AutoModel.from_pretrained("microsoft/codebert-base")

# -------------------- Dataset Wrapper --------------------- #
class PosPairDataset(Dataset):
    """Return (description, solution, token2type) triplets."""
    def __init__(self, split="train", seed=42):
        self.inner = CodeContestsDataset(split=split, subset="codeforces", seed=seed)
        all_ids = [tid for m in self.inner.data["token2type"] for tid in m.values()]
        self.n_types = max(all_ids, default=0) + 1
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, idx):
        desc, sol, token2type, *_ = self.inner[idx]
        return desc, sol, token2type

# -------------------- Collate fn -------------------------- #

def collate(batch):
    descs, codes, maps = zip(*batch)
    txt = tok_txt(list(descs), return_tensors="pt", padding=True,
                  truncation=True, max_length=TXT_MAXLEN)

    code_tok = tok_code(list(codes), return_tensors="pt", padding=True,
                        truncation=True, max_length=CODE_MAXLEN)

    L = code_tok["input_ids"].shape[1]
    type_tensors = []
    for code_str, mapping in zip(codes, maps):
        toks = tok_code.tokenize(code_str)
        type_ids = [0, 0] + [mapping.get(t, 0) for t in toks][:CODE_MAXLEN-3] + [0]
        if len(type_ids) < L:
            type_ids += [0]*(L-len(type_ids))
        else:
            type_ids = type_ids[:L]
        type_tensors.append(torch.tensor(type_ids, dtype=torch.long))
    code_type = torch.stack(type_tensors)

    return (txt["input_ids"], txt["attention_mask"],
            code_tok["input_ids"], code_tok["attention_mask"],
            code_type)

# -------------------- Model ------------------------------- #
class Projection(nn.Module):
    def __init__(self, in_dim=HIDDEN, out_dim=PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
    def forward(self, x):
        return self.proj(x)

class ContrastiveModel(nn.Module):
    def __init__(self, n_types: int):
        super().__init__()
        self.txt_enc   = enc_txt
        self.code_enc  = enc_code
        self.type_emb  = nn.Embedding(n_types, HIDDEN, padding_idx=0)
        self.proj_txt  = Projection()
        self.proj_code = Projection()
    def forward(self, txt_ids, txt_mask,
                      code_ids, code_mask, type_ids):
        txt_h   = self.txt_enc(input_ids=txt_ids, attention_mask=txt_mask).last_hidden_state
        txt_vec = self.proj_txt(txt_h[:, 0])

        inputs_embeds = self.code_enc.embeddings(input_ids=code_ids).clone()
        inputs_embeds = inputs_embeds + self.type_emb(type_ids.to(code_ids.device))
        code_h = self.code_enc(inputs_embeds=inputs_embeds, attention_mask=code_mask).last_hidden_state
        code_vec = self.proj_code(code_h[:, 0])
        return txt_vec, code_vec

# -------------------- Training ---------------------------- #

def train():
    ds = PosPairDataset(split="train", seed=0)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=2, collate_fn=collate, pin_memory=True)

    base_model = ContrastiveModel(n_types=ds.n_types).to(device)
    model = nn.DataParallel(base_model)                # ➡ DataParallel

    loss_fn = nn.CosineEmbeddingLoss()

    optim = torch.optim.AdamW([
        {"params": model.module.txt_enc.parameters(),  "lr": LR_ENCODER},
        {"params": model.module.code_enc.parameters(), "lr": LR_ENCODER},
        {"params": list(model.module.type_emb.parameters()) +
                    list(model.module.proj_txt.parameters()) +
                    list(model.module.proj_code.parameters()), "lr": LR_OTHER},
    ], weight_decay=1e-2)

    scheduler = get_linear_schedule_with_warmup(optim, len(dl)//10, len(dl)*EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, EPOCHS+1):
        model.train(); running = 0.0
        for step, batch in enumerate(tqdm(dl, desc=f"Epoch {epoch}")):
            (txt_ids, txt_mask,
             code_ids, code_mask,
             type_ids) = [x.to(device, non_blocking=True) for x in batch]

            perm = torch.randperm(txt_ids.size(0), device=device)
            txt_neg  = code_ids[perm].clone()
            mask_neg = code_mask[perm].clone()
            type_neg = type_ids[perm].clone()

            with torch.cuda.amp.autocast():
                v_txt, v_pos = model(txt_ids, txt_mask,
                                     code_ids, code_mask, type_ids)
                _,     v_neg = model(txt_ids, txt_mask,
                                     txt_neg, mask_neg, type_neg)
                emb1   = torch.cat([v_txt, v_txt], dim=0)
                emb2   = torch.cat([v_pos, v_neg], dim=0)
                target = torch.cat([torch.ones(v_txt.size(0),  device=device),
                                    -torch.ones(v_txt.size(0), device=device)])
                loss = loss_fn(emb1, emb2, target)

            scaler.scale(loss/GRAD_ACCUM).backward()

            if (step+1) % GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()

            running += loss.item()

        print(f"[Epoch {epoch}] mean loss = {running/len(dl):.4f}")

    print("Training done.  total type-vocab :", ds.n_types)

if __name__ == "__main__":
    train()
