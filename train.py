# train.py
import argparse, os, math, random
from pathlib import Path
from typing import Dict, List, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import CodeContestsDataset                        # ??user file
from model   import CBERT                                      # ??user file
from config import cfg
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
    """
    Batching function that
      ??prepends **two** CLS tokens and appends one SEP token to *both*
        the description prompt and the code solution,
      ??lets the ?§ó FastTokenizer handle padding in a single call,
      ??concatenates the fixed tokens with simple tensor ops
        (no Python for-loops ??faster),
      ??returns (text_ids, text_mask, code_ids, code_type_ids, code_mask,
                 explicit_feats, labels).
    """
    def __init__(
        self,
        txt_tok,
        code_tok,
        max_lt: int,
        max_lc: int,
        feat_dim: int,
    ):
        self.txt_tok, self.code_tok = txt_tok, code_tok
        self.max_lt, self.max_lc = max_lt, max_lc
        self.feat_dim = feat_dim

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
        descs, codes, feats, tags, labels = zip(*batch)
        B = len(batch)

        # ---------------------- 1.  TEXT (prompt) ------------------------- #
        enc_t = self.txt_tok(
            list(descs),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_lt - 3,      # leave room for 2√óCLS + SEP
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
            max_length=self.max_lc - 3,      # leave room for 2√óCLS + SEP
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
        _, L = code_ids.size()
        device = code_ids.device

        """
        type_tensors = []
        for b in range(B):
            code = codes[b]                # (L,)  ?†ÌÅ∞ ID ?úÌÄÄ??            mapping = type_mappings[b]           # dict {token_id: type_id}
            code = "[CLS]"+code+"[SEP]"
            # ???†ÌÅ∞ ID ???Ä??ID Îß§Ìïë (?ÜÏúºÎ©?0)
            t = torch.tensor([mapping.get(c_token, 0)  for c_token in self.code_tok.tokenize(code)],
                            dtype=torch.long,
                            device=device)

            # ???ÑÏöî?òÎ©¥ ?πÏàò ?†ÌÅ∞(CLS√ó2, SEP) ?òÎèô ÏßÄ??            # CLS_TYPE, SEP_TYPE = 0, 0  # Í∑∏Î?Î°??êÍ±∞????Î≤àÌò∏ Î∂Ä??            # t[0:2] = CLS_TYPE
            # t[-1]  = SEP_TYPE

            type_tensors.append(t)

        code_type = torch.stack(type_tensors, dim=0)   # (B, L)
        """
        code_type = torch.zeros_like(code_ids)
        # ---------------------- 3.  Explicit features --------------------- #
        feats = torch.stack(feats)            # (B, feat_dim)
        tags = torch.stack(tags)            # (B, feat_dim)

        return (
            text_ids,
            text_mask,
            code_ids,
            code_type,
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

# --------------------------------------------------------------------------- #
#                   3.  Main                                                  #
# --------------------------------------------------------------------------- #
def main():
    args = get_args(); set_seed(args.seed)
    dev = torch.device(args.device)

    VAL_RATIO = 0.1
    full_ds = CodeContestsDataset(subset=args.subset, split="train", seed=args.seed)
    val_len = int(len(full_ds)*VAL_RATIO)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                generator=torch.Generator().manual_seed(42))

    txt_tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    code_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    num_workers = 0
    feat_dim = 3 + train_ds.n_tags             # time, mem, io + multi-hot tags
    collate  = Collator(txt_tok, code_tok,
                        args.max_len_text, args.max_len_code, feat_dim)
    train_ld = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                          collate_fn=collate, num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False,
                          collate_fn=collate, num_workers=num_workers, pin_memory=True)

    num_labels = max(ex[-1] for ex in train_ds) + 1
    num_tag = len(train_ds[0][-2])
    model = CBERT(num_labels, num_tag, cfg.num_code_types)
    
    D = model.text.config.hidden_size
    model.classifier[0] = nn.Linear(D*2 + feat_dim, D)
    model = model.to(dev)

    # Optim / sched / loss
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_ld) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt,
                num_warmup_steps=int(0.05*total_steps),
                num_training_steps=total_steps)
    crit = nn.CrossEntropyLoss()

    # Train loop
    for ep in range(1, args.epochs+1):
        model.train()
        tot_loss, tot = 0.0, 0
        for batch in train_ld:
            (tid, tmask, cid, cmask, feat, tag, y) = [x.to(dev) for x in batch]
            opt.zero_grad()
            logits = model(tid, tmask, cid, cmask, feat, tag)
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
                (tid, tmask, cid, cmask, feat, tag, y) = [x.to(dev) for x in batch]
                logits = model(tid, tmask, cid, cmask, feat, tag)
                pred = logits.argmax(-1)
                correct += (pred==y).sum().item()
                total   += y.size(0)
        acc = correct/total if total else 0
        print(f"[E{ep}] valid acc  {acc:.3%}")

        # ---- checkpoint --------------------------------------------------- #
        ckpt = f"ckpt_epoch{ep}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"saved ??{ckpt}")

if __name__ == "__main__":
    main()
