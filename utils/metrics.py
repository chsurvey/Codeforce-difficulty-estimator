from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

def calc_metrics(preds: torch.Tensor, labels: torch.Tensor, num_labels: int):
    preds_np = preds.argmax(-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average="macro")
    # AUC requires one‑hot targets
    one_hot = torch.nn.functional.one_hot(torch.tensor(labels_np), num_labels)
    auc = roc_auc_score(one_hot, preds.cpu().numpy(), multi_class="ovr")
    return {"acc": acc, "f1": f1, "auc": auc}
