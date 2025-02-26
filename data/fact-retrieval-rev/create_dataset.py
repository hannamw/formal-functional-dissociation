# %%
import argparse
import random 
from collections import defaultdict
from typing import Literal

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm 

from eap.utils import model2family

def create_dataset(model_name: str):
    family = model2family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(f'country_capitals.csv')

    length_country_dict = defaultdict(list)
    for country in df['country']:
        country = country.strip()
        token_length = len(tokenizer(f' {country},', add_special_tokens=False).input_ids)
        length_country_dict[token_length].append(country)

    for fset, l in length_country_dict.items():
        if len(l) <= 1:
            print(f"found fset without proper alternatives {fset}: {l}")

    country_capital_mapping = {country: capital for country, capital in zip(df['country'], df['capital'])}

    dataset = {k:[] for k in ['clean', 'capital', 'country', 'capital_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_capital_idx']}
    
    def generate_sentence(country):
        return f' {country}, whose capital,'

    for country, capital in tqdm(list(zip(df['country'], df['capital']))):
        country_length = len(tokenizer(f' {country},', add_special_tokens=False).input_ids)
        
        clean = generate_sentence(country)
        capital_idx = tokenizer(f' {capital.strip()}', add_special_tokens=False).input_ids[0]

        valid_corrupted_countries = length_country_dict[country_length]
        if len(valid_corrupted_countries) == 1:
            continue
        corrupted_country = country
        while corrupted_country == country:
            corrupted_country = random.choice(valid_corrupted_countries)

        corrupted_capital = country_capital_mapping[corrupted_country]
        corrupted = generate_sentence(corrupted_country)

        corrupted_capital_idx = tokenizer(f' {corrupted_capital.strip()}', add_special_tokens=False).input_ids[0]

        for k, v in zip(['clean', 'capital', 'country', 'capital_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_capital_idx'], [clean, capital, country, capital_idx, corrupted, corrupted_capital, corrupted_country, corrupted_capital_idx]):
            dataset[k].append(v)

    df2 = pd.DataFrame.from_dict(dataset)
    df2.to_csv(f'{family}.csv', index=False)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)
# %%
