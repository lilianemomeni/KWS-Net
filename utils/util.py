import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

from zsvision.zs_iterm import zs_dispFig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def canonical_state_dict_keys(state_dict):
    remap = {
        "Classifier": "classifier",
        ".Enc.": ".enc_dec.encoder.",
        ".Dec.": ".enc_dec.decoder.",
    }
    canonical = {}
    for key, val in state_dict.items():
        for remap_key, remap_val in remap.items():
            key = key.replace(remap_key, remap_val)
        canonical[key] = val
    return canonical

