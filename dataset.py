from typing import List, Tuple, Dict, Optional, Union
import random
import re
import io
import tokenize
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import json
import numpy as np

__all__ = ["CodeContestsDataset"]

# --------------------------------------------------------------------------- #
#                       helpers                                               #
# --------------------------------------------------------------------------- #

# ──   map the integer `source` id used by the dataset → human-readable label
SOURCE_ID2NAME = {
    0: "UNKNOWN",
    1: "CODECHEF",
    2: "CODEFORCES",
    3: "HACKEREARTH",
    4: "CODEJAM",
    5: "ATCODER",
    6: "AIZU",
}

# ──   rating bins requested by the user for Codeforces problems
CF_RATING_BINS = [(0, 1500), (1500, 2000), (2000, 10_000)]  # →  three-class label


def _remove_python_comments(code: str) -> str:
    """
    Strip comments from Python code while preserving indentation
    (uses `tokenize` so it is string- and docstring-safe).
    """
    out: List[str] = []
    for tok_type, tok_str, *_ in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok_type == tokenize.COMMENT:
            continue
        out.append(tok_str)
    return "".join(out)


_CPP_COMMENT_PAT = re.compile(
    r"""
    //.*?$          |   # C++ single-line comment
    /\*.*?\*/           # C   multi-line comment
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)


def _remove_cpp_comments(code: str) -> str:
    return re.sub(_CPP_COMMENT_PAT, "", code)


def _strip_comments(code: str, lang_id: int) -> str:
    """
    lang_id follows dataset convention:
        1 -> PY2, 2 -> CPP, 3 -> PY3, 4 -> JAVA
    """
    if lang_id in {1, 3}:  # python / python3
        return _remove_python_comments(code)
    elif lang_id == 2:  # C++
        return _remove_cpp_comments(code)
    # fall-back: just return as-is
    return code


def _cf_rating_to_bin(r: int) -> int:
    for idx, (lo, hi) in enumerate(CF_RATING_BINS):
        if lo <= r < hi:
            return idx
    return len(CF_RATING_BINS) - 1  # extreme upper tail


# --------------------------------------------------------------------------- #
#                       main PyTorch dataset                                  #
# --------------------------------------------------------------------------- #


class CodeContestsDataset(TorchDataset):
    """
    A minimal PyTorch-compatible wrapper around the DeepMind **code_contests**
    dataset that returns exactly the fields requested by the user.

    Each **getitem** yields:
        x0 : str   —  concatenated `description` + public examples
        x1 : str   —  ONE correct solution (with comments stripped)
        x2 : torch.FloatTensor(4)   —  scalar feature vector:
                 [ time_limit_sec ,
                   memory_limit_mb ,
                   avg_public_io_bytes ,
                   category_label_id ]
        y  : int   —  difficulty label
            * Codeforces → three-class rating bin (0 / 1 / 2)
            * CodeChef   → dataset-native difficulty id (0-5 …)
            * others     → dataset-native difficulty id
    Parameters
    ----------
    subset : {"all", "codeforces", "codechef"}
        Which problem sources to keep.
    split  : {"train","validation","test"}
        Standard HF split name.
    seed   : int
        RNG seed used when sampling a single correct solution.
    """

    _HF_NAME = "deepmind/code_contests"

    def __init__(self, subset: str = "all", split: str = "train", seed: int = 0):
        subset = subset.lower()
        if subset not in {"all", "codeforces", "codechef"}:
            raise ValueError("subset must be one of all / codeforces / codechef")
        self.rng = random.Random(seed)
        
        # load once → filter once → keep in memory as standard dict list
        ds: Dataset = load_dataset(self._HF_NAME, split=split)
        with open("/home/guest-cjh/playground/data/"+subset+"/solutions.json", "r") as f:
          tokenized_solution = json.load(f)
        
        if subset != "all":
            keep_id = 2 if subset == "codeforces" else 1  # cf=2, cc=1
            ds = ds.filter(lambda ex: ex["source"] == keep_id)

        # ⚠  store as Python list for fast random access by index
        self.data: List[Dict] = ds.with_format("python")[:]
        self.data["token2type"] = []
        invalid_idx = []
        for idx, name in enumerate(self.data["name"]):
            token2type = tokenized_solution.get(name, {"token2type":None})["token2type"]
            if token2type == None:
              invalid_idx.append(idx)
            self.data["token2type"].append(token2type)
        
        
        ratings = self.data["cf_rating"]
        self.mu   = float(np.mean(ratings))
        self.sigma = float(np.std(ratings) + 1e-8)            # div-by-zero 안전장치
        
        invalid_idx = set(invalid_idx)
        for key in self.data.keys():
          self.data[key] = [val for idx, val in enumerate(self.data[key]) if idx not in invalid_idx]
            
        # ── ① 모든 Codeforces 문제에서 tag vocabulary 수집
        tag_set = set()
        for ex in self.data["cf_tags"]:
            for t in ex:
                tag_set.add(t.lower())
        self.tag2idx: Dict[str, int] = {t: i for i, t in enumerate(sorted(tag_set))}
        self.n_tags = len(self.tag2idx)

    def _compute_difficulty(self, r: int) -> int:
        """Codeforces rating r → {0,1,2} 레이블"""
        if r < self.mu - self.sigma:
            return 0
        elif r < self.mu + self.sigma:
            return 1
        return 2
        
    # ──────────────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self.data["cf_tags"])

    # ──────────────────────────────────────────────────────────────────────── #

    def _encode_tags(self, tags: List[str]) -> torch.Tensor:
        vec = torch.zeros(self.n_tags, dtype=torch.float32)
        for t in tags:
            i = self.tag2idx.get(t.lower())
            if i is not None:
                vec[i] = 1.0
        return vec
    
    def _build_feature_vector(self, ex: Dict) -> torch.Tensor:
        # (1)-(3) 기존 스칼라
        time_limit = ex.get("time_limit") or {"seconds": 0, "nanos": 0}
        time_sec = time_limit["seconds"] + time_limit["nanos"] / 1e9
        mem_mb = ex.get("memory_limit_bytes", 0) / 1_000_000.0
        inputs, outputs = ex["public_tests"]["input"], ex["public_tests"]["output"]
        avg_io = (
            (sum(len(s.encode()) for s in inputs) +
             sum(len(s.encode()) for s in outputs))
            / max(len(inputs) + len(outputs), 1)
        )

        scalar = torch.tensor([time_sec, mem_mb, avg_io], dtype=torch.float32)

        # (4) Codeforces tags → multi-hot, else 0-vector
        tag_vec = self._encode_tags(ex.get("cf_tags", []))

        return (scalar, tag_vec)

    # ──────────────────────────────────────────────────────────────────────── #

    def __getitem__(self, idx: int) -> Tuple[str, str, torch.Tensor, int]:
        ex = {key:self.data[key][idx] for key in self.data.keys()}

        token2type = self.data["token2type"][idx]
        # ----  x0 : description + public tests (inputs + outputs)  ---- #
        desc = ex["description"].strip()

        # ----  x1 : ONE correct solution (comments stripped)  ---- #
        sols = ex["solutions"]
        if len(sols["solution"]) == 0:
            return "", "", None, -1
        j = self.rng.randrange(len(sols["solution"]))
        raw_sol, lang_id = sols["solution"][j], sols["language"][j]

        # ----  x2 : scalar feature vector  ---- #
        feat_vec, tag_vec = self._build_feature_vector(ex)

        # ----  y : difficulty label  ---- #
        y = self._compute_difficulty(ex["cf_rating"])

        return desc, raw_sol, feat_vec, tag_vec, y
        