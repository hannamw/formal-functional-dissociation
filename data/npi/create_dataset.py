#%%
import argparse
from typing import List, Dict
import json
from collections import defaultdict
import random

import pandas as pd 
from transformers import AutoTokenizer

from eap.utils import model2family

def load_content(regions: List[Dict], npi: str) -> str:
    texts = []
    for region in regions:
        if region['content'] == npi:
            break 
        texts.append(region['content'])
    return ' '.join(texts)

def create_dataset(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ever_idx = tokenizer(' ever', add_special_tokens=False).input_ids[0]
    never_idx = tokenizer(' never', add_special_tokens=False).input_ids[0]

    any_idx = tokenizer(' any', add_special_tokens=False).input_ids[0]
    some_idx = tokenizer(' some', add_special_tokens=False).input_ids[0]

    whole_dataset_dict = defaultdict(list)
    for npi in ['any', 'ever']:
        licensed_label = ever_idx if npi == 'ever' else any_idx
        unlicensed_label = never_idx if npi == 'ever' else some_idx
        for relcl in ['subj', 'obj']:
            data = json.load(open(f'{npi}_{relcl}.json'))
            data_dict = defaultdict(list)
            
            for item in data['items']:
                for condition in item['conditions']:
                    data_dict[condition['condition_name']].append(load_content(condition['regions'], npi))

            assert len(data_dict) == 4 and all(f'{first}_{second}' in data_dict for first in ['pos', 'neg'] for second in ['pos', 'neg'])
            df_dict = defaultdict(list)

            for i in range(len(data_dict['neg_neg'])):
                licensed_keys = ['neg_neg', 'neg_pos']
                unlicensed_keys = ['pos_neg', 'pos_pos']

                ul = random.random() < 0.5  # which of the unlicensed gets paired with neg_neg
                cl1 = random.random() < 0.5  # is the clean version licensed (0) or unlicensed (1)
                cl2 = random.random() < 0.5  # is the clean version licensed (0) or unlicensed (1)

                first = ('neg_neg', unlicensed_keys[ul])
                first_clean_idx, first_corrupted_idx = (unlicensed_label, licensed_label) if cl1 else (licensed_label, unlicensed_label)
                second = ('neg_pos', unlicensed_keys[1 - ul])
                second_clean_idx, second_corrupted_idx = (unlicensed_label, licensed_label) if cl2 else (licensed_label, unlicensed_label)

                df_dict['clean'].append(data_dict[first[cl1]][i])
                df_dict['corrupted'].append(data_dict[first[1 - cl1]][i])
                df_dict['clean_idx'].append(first_clean_idx)
                df_dict['corrupted_idx'].append(first_corrupted_idx)

                df_dict['clean'].append(data_dict[first[cl2]][i])
                df_dict['corrupted'].append(data_dict[first[1 - cl2]][i])
                df_dict['clean_idx'].append(second_clean_idx)
                df_dict['corrupted_idx'].append(second_corrupted_idx)

            for k,v in df_dict.items():
                whole_dataset_dict[k] += v 

            npi_relcl_df = pd.DataFrame.from_dict(df_dict)
            npi_relcl_df.to_csv(f'{model2family(model_name)}_{npi}_{relcl}.csv', index=False)

    df = pd.DataFrame.from_dict(whole_dataset_dict)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)
# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)
# %%
