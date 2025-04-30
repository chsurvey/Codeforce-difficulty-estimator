import argparse
import torch
from torch.utils.data import DataLoader
import os

from config import cfg
from dataset import PojDataset
from model import CBERT
from utils.metrics import calc_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["codeforces", "codechef"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    data_path = os.path.join("data", f"{args.dataset}.jsonl")
    cfg.num_labels = 3 if args.dataset == "codeforces" else 5

    dset = PojDataset(data_path)
    loader = DataLoader(dset, batch_size=cfg.batch_size)

    model = CBERT(cfg.num_labels).to(cfg.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=cfg.device))
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for text_in, code_in, feat, label in loader:
            text_in = {k: v.to(cfg.device) for k, v in text_in.items()}
            code_in = {k: v.to(cfg.device) for k, v in code_in.items()}
            feat = feat.to(cfg.device)
            label = label.to(cfg.device)
            logits = model(text_in, code_in, feat)
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = calc_metrics(logits_cat, labels_cat, cfg.num_labels)
    print("Evaluation | ACC {acc:.4f} F1 {f1:.4f} AUC {auc:.4f}".format(**metrics))