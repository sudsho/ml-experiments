# little helpers I keep reusing
import pandas as pd
import numpy as np

def quick_summary(df):
    print('shape:', df.shape)
    print('dtypes:')
    print(df.dtypes)
    print('nulls per col:')
    print(df.isnull().sum())
    print('memory mb:', df.memory_usage(deep=True).sum() / 1024 / 1024)

def split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def value_counts_pct(s):
    out = pd.concat([s.value_counts(), s.value_counts(normalize=True).round(3)], axis=1)
    out.columns = ['n', 'pct']
    return out

def reduce_mem(df):
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    return df

def high_cardinality(df, threshold=50):
    out = {}
    for c in df.select_dtypes(include=['object']).columns:
        n = df[c].nunique()
        if n > threshold:
            out[c] = n
    return out



def to_device(batch, dev):
    return [b.to(dev) for b in batch]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def best_trial_summary(study):
    t = study.best_trial
    return {'value': t.value, 'params': t.params, 'number': t.number}



def simple_tokenize(s):
    import re
    return [w for w in re.findall(r'[a-z]+', s.lower()) if len(w) > 1]

def onehot_safe(s):
    return pd.get_dummies(s, dummy_na=True)

def set_seed(seed=42):
    import random as _r
    import numpy as _np
    import torch as _t
    _r.seed(seed); _np.random.seed(seed); _t.manual_seed(seed)

def now_str():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def list_files(path, ext='.csv'):
    from pathlib import Path
    return [p for p in Path(path).rglob(f'*{ext}')]


def freeze_layers(model, until_name=None):
    """freeze model params up to (and including) until_name. used for finetuning."""
    frozen = True
    for n, p in model.named_parameters():
        p.requires_grad = not frozen
        if until_name is not None and until_name in n:
            frozen = False


def cosine_warmup(step, warmup, total, base_lr=1e-3):
    import math
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def to_jsonl(rows, path):
    import json
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def safe_div(a, b, default=0.0):
    return a / b if b else default


def dict_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def dict_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def topk_acc(logits, y, k=5):
    import torch
    _, idx = logits.topk(k, dim=-1)
    return (idx == y.unsqueeze(-1)).any(dim=-1).float().mean().item()


def dict_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def dict_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
