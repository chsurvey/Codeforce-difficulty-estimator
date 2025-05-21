# train.py
import argparse, os, math, random
from pathlib import Path
from typing import Dict, List, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import CodeContestsDataset                        # ← user file
from model   import CBERT                                      # ← user file
# --------------------------------------------------------------------------- #
#                   0.  Argument parsing                                      #
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="codeforces",
                   choices=["all", "codeforces", "codechef"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bsz",    type=int, default=8)
    p.add_argument("--lr",     type=float, default=2e-5)
    p.add_argument("--max_len_text", type=int, default=256)
    p.add_argument("--max_len_code", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()

# --------------------------------------------------------------------------- #
#                   1.  Collate fn                                            #
# --------------------------------------------------------------------------- #
class Collator:
    def __init__(self, txt_tok, code_tok,
                 max_lt: int, max_lc: int, feat_dim: int):
        self.txt_tok, self.code_tok = txt_tok, code_tok
        self.max_lt, self.max_lc = max_lt, max_lc
        self.feat_dim = feat_dim
        self.cls_id_txt  = txt_tok.cls_token_id
        self.sep_id_txt  = txt_tok.sep_token_id
        self.cls_id_code = code_tok.cls_token_id
        self.sep_id_code = code_tok.sep_token_id

    def __call__(self, batch):
        descs, codes, feats, labels = zip(*batch)

        # --- text prompt --------------------------------------------------- #
        t = self.txt_tok(list(descs), add_special_tokens=False,
                         truncation=True, max_length=self.max_lt-3)
        text_ids  = []
        text_mask = []
        for ids, attn in zip(t["input_ids"], t["attention_mask"]):
            ids = [self.cls_id_txt, self.cls_id_txt] + ids + [self.sep_id_txt]
            mask= [1]*len(ids)
            text_ids.append(ids)
            text_mask.append(mask)
        text = self.txt_tok.pad(
            {"input_ids":text_ids, "attention_mask":text_mask},
            padding="longest", return_tensors="pt")

        # --- code solution -------------------------------------------------- #
        c = self.code_tok(list(codes), add_special_tokens=False,
                          truncation=True, max_length=self.max_lc-3)
        code_ids, code_mask, code_type = [], [], []
        for ids, attn in zip(c["input_ids"], c["attention_mask"]):
            ids = [self.cls_id_code, self.cls_id_code] + ids + [self.sep_id_code]
            mask= [1]*len(ids)
            tpe = [0]*len(ids)           # JOERN 타입 정보 없음 → 0
            code_ids.append(ids)
            code_mask.append(mask)
            code_type.append(tpe)
        code = self.code_tok.pad(
            {"input_ids":code_ids, "attention_mask":code_mask},
            padding="longest", return_tensors="pt")
        code["type_ids"] = torch.tensor(
            [seq + [0]*(code["input_ids"].size(1)-len(seq)) for seq in code_type])

        # --- explicit scalar+tag feature ----------------------------------- #
        f = torch.stack(feats)           # (B, feat_dim)

        return (text["input_ids"],  text["attention_mask"],
                code["input_ids"], code["type_ids"], code["attention_mask"],
                f,
                torch.tensor(labels))

# --------------------------------------------------------------------------- #
#                   2.  Utility                                               #
# --------------------------------------------------------------------------- #
def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# --------------------------------------------------------------------------- #
#                   3.  Main                                                  #
# --------------------------------------------------------------------------- #
def main():
    args = get_args(); set_seed(args.seed)
    dev = torch.device(args.device)

    # ① Dataset & dataloader
    train_ds = CodeContestsDataset(subset=args.subset, split="train", seed=args.seed)
    val_ds   = CodeContestsDataset(subset=args.subset, split="validation", seed=args.seed)

    txt_tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    code_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    feat_dim = 3 + train_ds.n_tags             # time, mem, io + multi-hot tags
    collate  = Collator(txt_tok, code_tok,
                        args.max_len_text, args.max_len_code, feat_dim)
    train_ld = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                          collate_fn=collate, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)

    # ② Model (num_labels = max difficulty id + 1)
    num_labels = max(ex[-1] for ex in train_ds.data) + 1
    model = CBERT(num_labels)
    # classifier 첫 층 입력크기 수정 (D*2 + feat_dim)
    D = model.text.config.hidden_size
    model.classifier[0] = nn.Linear(D*2 + feat_dim, D)
    model = model.to(dev)

    # ③ Optim / sched / loss
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_ld) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt,
                num_warmup_steps=int(0.05*total_steps),
                num_training_steps=total_steps)
    crit = nn.CrossEntropyLoss()

    # ④ Train loop
    for ep in range(1, args.epochs+1):
        model.train()
        tot_loss, tot = 0.0, 0
        for batch in train_ld:
            (tid, tmask, cid, ctype, cmask, feat, y) = [x.to(dev) for x in batch]
            opt.zero_grad()
            logits = model(tid, tmask, cid, ctype, cmask, feat)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot_loss += loss.item()*y.size(0); tot += y.size(0)
        print(f"[E{ep}] train loss {tot_loss/tot:.4f}")

        # ---- validation --------------------------------------------------- #
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_ld:
                (tid, tmask, cid, ctype, cmask, feat, y) = [x.to(dev) for x in batch]
                logits = model(tid, tmask, cid, ctype, cmask, feat)
                pred = logits.argmax(-1)
                correct += (pred==y).sum().item()
                total   += y.size(0)
        acc = correct/total if total else 0
        print(f"[E{ep}] valid acc  {acc:.3%}")

        # ---- checkpoint --------------------------------------------------- #
        ckpt = f"ckpt_epoch{ep}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"saved → {ckpt}")

if __name__ == "__main__":
    main()
