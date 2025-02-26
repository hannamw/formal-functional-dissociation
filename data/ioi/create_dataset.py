#%%
import argparse

import pandas as pd 
from transformers import AutoTokenizer

from ioi_dataset import IOIDataset
from eap.utils import model2family


def create_dataset(model_name: str):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = IOIDataset('mixed', N=1000, tokenizer=tokenizer)
    abc_dataset = (  # TODO seeded
        ds.gen_flipped_prompts(("S2", "RAND"))
    )
    abb_dataset = (  # TODO seeded
        ds.gen_flipped_prompts(("S2", "IO"))
    )

    d = {'clean': [], 'corrupted': [], 'corrupted_hard': [], 'correct_idx': [], 'incorrect_idx': []}
    for i in range(len(ds)):
        clean = ' '.join(ds.sentences[i].split()[:-1])
        corrupted = ' '.join(abc_dataset.sentences[i].split()[:-1])
        corrupted_hard = ' '.join(abb_dataset.sentences[i].split()[:-1])
        correct = tokenizer.encode(f' {ds.ioi_prompts[i]["IO"]}', add_special_tokens=False)[0] 
        #ds.toks[i, ds.word_idx['IO'][i]].item()
        incorrect = tokenizer.encode(f' {ds.ioi_prompts[i]["S"]}', add_special_tokens=False)[0] 
        #incorrect = ds.toks[i, ds.word_idx['S'][i]].item()
        d['clean'].append(clean)
        d['corrupted'].append(corrupted)
        d['corrupted_hard'].append(corrupted_hard)
        d['correct_idx'].append(correct)
        d['incorrect_idx'].append(incorrect)

    df = pd.DataFrame.from_dict(d)
    df = df.sample(frac=1)
    df.to_csv(f'{model2family(model_name)}.csv')
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)
# %%
