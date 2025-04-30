"""Training with tqdm progress bars **and Weights & Biases logging**."""
import argparse, os, torch, wandb
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import cfg
from dataset import PojDataset
from model import CBERT
from utils.seed import set_seed
from utils.metrics import calc_metrics


def train_fold(dataset_path, fold_idx, splits, num_labels):
    set_seed(cfg.seed + fold_idx)

    model = CBERT(num_labels).to(cfg.device)
    if fold_idx == 0:
        wandb.watch(model, log="all", log_freq=100)

    train_idx, val_idx = splits[fold_idx]
    train_set = Subset(PojDataset(dataset_path), train_idx)
    val_set = Subset(PojDataset(dataset_path), val_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * cfg.warmup_ratio), total_steps)

    best_f1 = 0.0
    os.makedirs(cfg.log_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.log_dir, f"models/model_fold{fold_idx}.pt")

    global_step = 0
    for epoch in range(cfg.epochs):
        # ---------------- training ----------------
        model.train()
        bar = tqdm(train_loader, desc=f"Fold {fold_idx} | Epoch {epoch} [train]", leave=False)
        for text_in, code_in, feat, label in bar:
            text_in = {k: v.to(cfg.device) for k, v in text_in.items()}
            code_in = {k: v.to(cfg.device) for k, v in code_in.items()}
            feat, label = feat.to(cfg.device), label.to(cfg.device)

            logits = model(text_in, code_in, feat)
            loss = torch.nn.functional.cross_entropy(logits, label)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 100 == 0:
                wandb.log({f"fold{fold_idx}/train_loss": loss.item(), "step": global_step})
            global_step += 1

        # ---------------- validation --------------
        model.eval(); all_logits, all_labels = [], []
        with torch.no_grad():
            for text_in, code_in, feat, label in tqdm(val_loader, desc=f"Fold {fold_idx} | Epoch {epoch} [val]", leave=False):
                text_in = {k: v.to(cfg.device) for k, v in text_in.items()}
                code_in = {k: v.to(cfg.device) for k, v in code_in.items()}
                feat, label = feat.to(cfg.device), label.to(cfg.device)
                all_logits.append(model(text_in, code_in, feat).cpu())
                all_labels.append(label.cpu())
        metrics = calc_metrics(torch.cat(all_logits), torch.cat(all_labels), num_labels)
        wandb.log({f"fold{fold_idx}/val_acc": metrics['acc'],
                   f"fold{fold_idx}/val_f1": metrics['f1'],
                   f"fold{fold_idx}/val_auc": metrics['auc'],
                   "epoch": epoch})
        print(f"Fold {fold_idx} | Epoch {epoch} → ACC {metrics['acc']:.4f} F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']; torch.save(model.state_dict(), ckpt_path)
            print("  ↳ new best model saved")

# ---------------- splitting util ----------------
def build_splits(dataset_path, n_splits):
    dset = PojDataset(dataset_path)
    labels = [s['label'] for s in dset.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    return list(skf.split(range(len(dset)), labels))

# -------------------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["codeforces", "codechef"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--wandb_project", default=os.getenv("WANDB_PROJECT", "cbert"))
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    # init wandb
    wandb.init(project=args.wandb_project,
               name=args.wandb_name or f"{args.dataset}_folds{args.folds}",
               config={**cfg.__dict__, "dataset": args.dataset, "folds": args.folds})

    cfg.num_labels = 3 if args.dataset == "codeforces" else 5
    data_path = os.path.join("data", f"{args.dataset}.jsonl")
    splits = build_splits(data_path, args.folds)
    for fid in range(args.folds):
        train_fold(data_path, fid, splits, cfg.num_labels)

    wandb.finish()