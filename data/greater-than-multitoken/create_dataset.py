#%%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch

from greater_than_dataset import YearDataset
from eap.utils import model2family
#%%
def create_dataset(model_name:str):
    N = 10000

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = YearDataset(torch.arange(1102,1898), model_name, N, Path("potential_nouns.txt"), tokenizer)

    d = {'clean': ds.good_sentences, 'corrupted': ds.bad_sentences,  'label1': ds.label1, 'label2': ds.label2}

    df = pd.DataFrame.from_dict(d)
    df = df.sample(frac=1)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)

# %%
