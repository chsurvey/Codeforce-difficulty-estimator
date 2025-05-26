# train_val_dp_split.py
import os
import math

# -----------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import wandb
from dataset import CodeContestsDataset

# -----------------------------------------------------------------
# 0.  wandb 세팅
# -----------------------------------------------------------------
wandb.init(project="codecontest-contrastive")

# -----------------------------------------------------------------
# 1. 하이퍼파라미터
# -----------------------------------------------------------------
BATCH, EPOCHS           = 64, 30
VAL_RATIO               = 0.10        # train : val  =  0.9 : 0.1
TXT_MAXLEN = CODE_MAXLEN = 512
HIDDEN, PROJ_DIM        = 768, 256
LR_ENCODER, LR_OTHER    = 2e-5, 1e-4
GRAD_ACCUM, TEMP        = 2, 0.07
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------
# 2. 토크나이저 / 프리트레인 인코더
# -----------------------------------------------------------------
tok_txt  = AutoTokenizer.from_pretrained("bert-base-uncased")
enc_txt  = AutoModel.from_pretrained("bert-base-uncased")
tok_code = AutoTokenizer.from_pretrained("microsoft/codebert-base")
enc_code = AutoModel.from_pretrained("microsoft/codebert-base")

# -----------------------------------------------------------------
# 3. 데이터셋
# -----------------------------------------------------------------
class PosPairDataset(Dataset):
    """(description, solution, token2type) 반환."""
    def __init__(self, split="train", seed=42):
        self.inner = CodeContestsDataset(split=split, subset="codeforces",
                                         seed=seed)
        all_ids = [tid for m in self.inner.data["token2type"] for tid in m.values()]
        self.n_types = max(all_ids, default=0) + 1
    def __len__(self):  return len(self.inner)
    def __getitem__(self, idx):
        desc, sol, token2type, *_ = self.inner[idx]
        return desc, sol, token2type

# -----------------------------------------------------------------
def collate(batch):
    descs, codes, maps = zip(*batch)
    txt  = tok_txt (list(descs), return_tensors="pt", padding=True,
                    truncation=True, max_length=TXT_MAXLEN)
    code = tok_code(list(codes), return_tensors="pt", padding=True,
                    truncation=True, max_length=CODE_MAXLEN)

    L = code["input_ids"].shape[1]
    type_mat = []
    for cstr, mapping in zip(codes, maps):
        toks = tok_code.tokenize(cstr, truncation=True, max_length=CODE_MAXLEN)
        ids  = [0,0]+[mapping.get(t,0) for t in toks][:CODE_MAXLEN-3]+[0]
        ids += [0]*(L-len(ids)) if len(ids)<L else []
        type_mat.append(torch.tensor(ids[:L], dtype=torch.long))
    return (txt["input_ids"], txt["attention_mask"],
            code["input_ids"], code["attention_mask"],
            torch.stack(type_mat))

# -----------------------------------------------------------------
# 4. 모델
# -----------------------------------------------------------------
class Projection(nn.Module):
    def __init__(self, in_dim=HIDDEN, out_dim=PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim,in_dim), nn.ReLU(),
                                  nn.Linear(in_dim,out_dim))
    def forward(self,x): return self.proj(x)

class ContrastiveModel(nn.Module):
    def __init__(self, n_types:int):
        super().__init__()
        self.txt_enc, self.code_enc = enc_txt, enc_code
        self.type_emb = nn.Embedding(n_types,HIDDEN,padding_idx=0)
        self.proj_txt, self.proj_code = Projection(), Projection()
    def forward(self,
                txt_ids, txt_mask,
                code_ids, code_mask, type_ids):
        # -------- text encoder: masked mean pooling -------- #
        txt_out   = self.txt_enc(input_ids=txt_ids,
                                 attention_mask=txt_mask).last_hidden_state  # (B,L,H)
        txt_maskf = txt_mask.unsqueeze(-1)                                   # (B,L,1)
        txt_vec   = (txt_out * txt_maskf).sum(dim=1) / txt_maskf.sum(dim=1).clamp(min=1e-6)
        txt_vec   = self.proj_txt(txt_vec)                                   # (B,H) → projection

        # -------- code encoder: masked mean pooling -------- #
        code_out   = self.code_enc(input_ids=code_ids,
                                   attention_mask=code_mask).last_hidden_state
        code_maskf = code_mask.unsqueeze(-1)
        code_vec   = (code_out * code_maskf).sum(dim=1) / code_maskf.sum(dim=1).clamp(min=1e-6)
        code_vec   = self.proj_code(code_vec)

        return txt_vec, code_vec

# -----------------------------------------------------------------
# 5. InfoNCE + metric
# -----------------------------------------------------------------
def info_nce(anchor, positive, temp=TEMP):
    a, p = F.normalize(anchor, dim=1), F.normalize(positive, dim=1)
    logits = a @ p.T / temp                       # (B,B)
    labels = torch.arange(a.size(0), device=a.device)
    loss   = F.cross_entropy(logits, labels)

    with torch.no_grad():
        acc = (logits.argmax(1)==labels).float().mean().item()
        tgt = torch.eye(a.size(0), device=a.device).flatten()
        try: auc = roc_auc_score(tgt.cpu(), logits.flatten().cpu())
        except ValueError: auc = float('nan')
    return loss, acc, auc

# -----------------------------------------------------------------
# 6. epoch 실행 함수 (p-bar 포함)
# -----------------------------------------------------------------
def run_epoch(model, loader, phase, optim=None, scaler=None,
              scheduler=None, epoch=0):
    training = optim is not None
    model.train() if training else model.eval()
    totL=totA=totU=0.0; n=0
    pbar = tqdm(loader, desc=f"{phase} {epoch}", leave=False)
    for batch in pbar:
        (txt_ids, txt_mask, code_ids, code_mask, type_ids) = \
            [x.to(device, non_blocking=True) for x in batch]

        ctx = torch.cuda.amp.autocast() if training else torch.no_grad()
        with ctx:
            v_txt, v_pos = model(txt_ids, txt_mask,
                                 code_ids, code_mask, type_ids)
            loss, acc, auc = info_nce(v_txt, v_pos)

        if training:
            scaler.scale(loss/GRAD_ACCUM).backward()
            if (n+1) % GRAD_ACCUM == 0:
                scaler.step(optim); scaler.update()
                optim.zero_grad();  scheduler.step()

        totL+=loss.item(); totA+=acc; totU+=0 if auc!=auc else auc; n+=1
        pbar.set_postfix(L=f"{totL/n:.4f}", A=f"{totA/n:.3f}", U=f"{totU/max(1,n):.3f}")
    return totL/n, totA/n, totU/max(1,n)

# -----------------------------------------------------------------
# 7. main
# -----------------------------------------------------------------
def main():
    # --- 전체 train split 로드 후 90:10 분할
    full_ds = PosPairDataset("train")
    val_len = int(len(full_ds)*VAL_RATIO)
    train_len = len(full_ds) - val_len
    ds_tr, ds_va = random_split(full_ds, [train_len, val_len],
                                generator=torch.Generator().manual_seed(42))
    n_types = full_ds.n_types   # Subset엔 속성이 없으므로 따로 저장

    dl_tr = DataLoader(ds_tr,batch_size=BATCH,shuffle=True,
                       num_workers=2,collate_fn=collate,pin_memory=True)
    dl_va = DataLoader(ds_va,batch_size=BATCH,shuffle=False,
                       num_workers=2,collate_fn=collate,pin_memory=True)

    model = nn.DataParallel(ContrastiveModel(n_types).to(device))
    optim = torch.optim.AdamW([
        {"params":model.module.txt_enc.parameters(),"lr":LR_ENCODER},
        {"params":model.module.code_enc.parameters(),"lr":LR_ENCODER},
        {"params": list(model.module.type_emb.parameters()) +
                   list(model.module.proj_txt.parameters()) +
                   list(model.module.proj_code.parameters()), "lr":LR_OTHER}],
        weight_decay=1e-2)
    sched  = get_linear_schedule_with_warmup(optim,len(dl_tr)//10,
                                             len(dl_tr)*EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
  
    best_auc = -math.inf
    for ep in range(1, EPOCHS+1):
        trL,trA,trU = run_epoch(model, dl_tr, "train",
                                optim, scaler, sched, ep)
        vaL,vaA,vaU = run_epoch(model, dl_va, "val",  epoch=ep)

        wandb.log({"epoch":ep,
                   "train/loss":trL,"train/acc":trA,"train/auc":trU,
                   "val/loss":vaL,"val/acc":vaA,"val/auc":vaU})
        print(f"[{ep}] train L{trL:.4f}|A{trA:.3f}|U{trU:.3f}  ||  "
              f"val L{vaL:.4f}|A{vaA:.3f}|U{vaU:.3f}")
        
        
        if not math.isnan(vaU) and vaU > best_auc:
            best_auc = vaU
            os.makedirs("./models", exist_ok=True)
    
            txt_dir  = f"./models/best_bert_finetuned"
            code_dir = f"./models/best_codebert_finetuned"
    
            model.module.txt_enc.save_pretrained(txt_dir)
            tok_txt.save_pretrained(txt_dir)
    
            model.module.code_enc.save_pretrained(code_dir)
            tok_code.save_pretrained(code_dir)
    
            print(f"★ New best val AUC {vaU:.3f} (epoch {ep}) — saved to {txt_dir} & {code_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
