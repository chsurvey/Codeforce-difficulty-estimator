# train.py
import argparse, os, math, random
from pathlib import Path
from typing import Dict, List, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import CodeContestsDataset                        # user file
from model   import CBERT                                      # user file

from config import cfg
import wandb
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# NEW
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
#                   0.  Argument parsing                                      #
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="codeforces",
                   choices=["all", "codeforces", "codechef"])
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--bsz",    type=int, default=cfg.batch_size)
    p.add_argument("--lr",     type=float, default=cfg.lr)
    p.add_argument("--max_len_text", type=int, default=512)
    p.add_argument("--max_len_code", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()

# --------------------------------------------------------------------------- #
#                   1.  Collate fn                                            #
# --------------------------------------------------------------------------- #

class Collator:
    def __init__(
        self,
        txt_tok,
        code_tok,
        max_lt: int,
        max_lc: int
    ):
        self.txt_tok, self.code_tok = txt_tok, code_tok
        self.max_lt, self.max_lc = max_lt, max_lc

        self.cls_id_txt = txt_tok.cls_token_id
        self.sep_id_txt = txt_tok.sep_token_id
        self.cls_id_code = code_tok.cls_token_id
        self.sep_id_code = code_tok.sep_token_id

    # --------------------------------------------------------------------- #
    #                                CALL                                   #
    # --------------------------------------------------------------------- #
    def __call__(
        self, batch: List[Tuple[str, str, torch.Tensor, int]]
    ):
        batch = [b for b in batch if b[-1] is not None]
        descs, codes, feats, tags, labels = zip(*batch)

        B = len(batch)

        # ---------------------- 1.  TEXT (prompt) ------------------------- #
        enc_t = self.txt_tok(
            list(descs),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_lt - 3,      # leave room for 2CLS + SEP
            padding=True,
            return_tensors="pt",
        )
        prefix_txt = torch.full((B, 2), self.cls_id_txt, dtype=torch.long)
        sep_txt    = torch.full((B, 1), self.sep_id_txt,  dtype=torch.long)

        text_ids  = torch.cat([prefix_txt, enc_t["input_ids"], sep_txt], dim=1)
        text_mask = torch.cat(
            [torch.ones_like(prefix_txt), enc_t["attention_mask"], torch.ones_like(sep_txt)],
            dim=1,
        )

        # ---------------------- 2.  CODE (solution) ----------------------- #
        enc_c = self.code_tok(
            list(codes),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_lc - 3,      # leave room for 2CLS + SEP
            padding=True,
            return_tensors="pt",
        )
        prefix_code = torch.full((B, 2), self.cls_id_code, dtype=torch.long)
        sep_code    = torch.full((B, 1), self.sep_id_code,  dtype=torch.long)

        code_ids  = torch.cat([prefix_code, enc_c["input_ids"], sep_code], dim=1)
        code_mask = torch.cat(
            [torch.ones_like(prefix_code), enc_c["attention_mask"], torch.ones_like(sep_code)],
            dim=1,
        )
        # ---------------------- 3.  Explicit features --------------------- #
        feats = torch.stack(feats)            # (B, feat_dim)
        tags = torch.stack(tags)              # (B, feat_dim)

        return (
            text_ids,
            text_mask,
            code_ids,
            code_mask,
            feats,
            tags,
            torch.tensor(labels),
        )

# --------------------------------------------------------------------------- #
#                   2.  Utility                                               #
# --------------------------------------------------------------------------- #

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def calc_clf_metrics(logits_cat, labels_cat, num_labels):
    probs = torch.softmax(logits_cat, dim=-1).cpu().numpy()
    y_true = labels_cat.cpu().numpy()
    y_pred = logits_cat.argmax(-1).cpu().numpy()

    if num_labels == 2:
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
    f1  = f1_score(y_true, y_pred, average='macro')
    acc = (y_pred == y_true).mean()
    return acc, auc, f1

# --------------------------------------------------------------------------- #
#                   3.  Main                                                  #
# --------------------------------------------------------------------------- #

def main():
    args = get_args(); set_seed(args.seed)
    wandb.init(project="CBERT train", config=vars(args))
    dev = torch.device(args.device)

    VAL_RATIO = 0.1
    full_ds = CodeContestsDataset(subset=args.subset, split="train", seed=args.seed)
    val_len = int(len(full_ds) * VAL_RATIO)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator())

    txt_tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    code_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    num_workers = 4
    num_labels = 3  
    num_tag = len(train_ds[0][-2])
    collate  = Collator(txt_tok, code_tok, args.max_len_text, args.max_len_code)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.bsz,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = nn.DataParallel(CBERT(num_labels, num_tag).to(dev))#, '/home/guest-cjh/playground/models/best_bert_finetuned/', '/home/guest-cjh/playground/models/best_codebert_finetuned/').to(dev))

    # Optim / sched / loss
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_ld) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    crit = nn.CrossEntropyLoss()

    best_auc = -1.0

    # ----------------------------- Train loop ---------------------------- #
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss, tot = 0.0, 0
        train_logits, train_labels = [], []

        train_pbar = tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs} [train]", leave=False)
        for batch in train_pbar:
            (tid, tmask, cid, cmask, feat, tag, y) = [x.to(dev) for x in batch]
            opt.zero_grad()
            logits, moe_loss = model(tid, tmask, cid, cmask, feat, tag)
            
            ce_loss = crit(logits, y) + moe_loss
            loss = ce_loss + moe_loss*cfg.moe_loss_weight if cfg.use_moe else ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            tot_loss += loss.item() * y.size(0); tot += y.size(0)
            train_logits.append(logits.detach())
            train_labels.append(y.detach())

            train_pbar.set_postfix(loss=tot_loss / tot if tot else 0.0)

        train_loss = tot_loss / tot
        train_acc, train_auc, train_f1 = calc_clf_metrics(
            torch.cat(train_logits), torch.cat(train_labels), num_labels)

        # ---------------------------- Validation ------------------------- #
        model.eval()
        val_loss, vtot = 0.0, 0
        val_logits, val_labels = [], []
        val_pbar = tqdm(val_ld, desc=f"Epoch {ep}/{args.epochs} [val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                (tid, tmask, cid, cmask, feat, tag, y) = [x.to(dev) for x in batch]
                logits, _ = model(tid, tmask, cid, cmask, feat, tag)
                loss = crit(logits, y)

                val_loss += loss.item() * y.size(0); vtot += y.size(0)
                val_logits.append(logits)
                val_labels.append(y)

                val_pbar.set_postfix(loss=val_loss / vtot if vtot else 0.0)

        val_loss /= vtot
        val_acc, val_auc, val_f1 = calc_clf_metrics(
            torch.cat(val_logits), torch.cat(val_labels), num_labels)

        # ----------------------- Epoch summary --------------------------- #
        print(
            f"[E{ep}] "
            f"train loss {train_loss:.4f} | acc {train_acc:.3%} | AUC {train_auc:.4f} | F1 {train_f1:.4f}  ||  "
            f"val loss {val_loss:.4f} | acc {val_acc:.3%} | AUC {val_auc:.4f} | F1 {val_f1:.4f}"
        )

        # wandb logging
        wandb.log({
            "epoch": ep,
            "train/loss": train_loss,
            "train/moe loss": moe_loss,
            "train/acc": train_acc,
            "train/auc": train_auc,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/auc": val_auc,
            "val/f1": val_f1,
        })

        # checkpoint (by val AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_auc.pt")
            print(f"New best AUC {best_auc:.4f} model saved")


if __name__ == "__main__":
    main()
