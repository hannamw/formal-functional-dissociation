# %%
import argparse
import random 
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm 

from eap.utils import model2family

def create_dataset(model_name: str):
    family = model2family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv('/home/Ubuntu/formal_functional/data/fact-retrieval-rev/country_capitals.csv')
    df['capital2'] = df['capital']
    df['capital'] = df['country']
    df['country'] = df['capital2']

    length_city_dict = defaultdict(list)
    for city in df['capital']:
        city = city.strip()
        token_length = len(tokenizer(f' {city},', add_special_tokens=False).input_ids)
        length_city_dict[token_length].append(city)

    for fset, l in length_city_dict.items():
        if len(l) <= 1:
            print(f"found fset without proper alternatives {fset}: {l}")

    capital_country_mapping = {capital: country for country, capital in zip(df['country'], df['capital'])}

    dataset = {k:[] for k in ['clean', 'capital', 'country', 'country_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_country_idx']}

    for country, capital in tqdm(list(zip(df['country'], df['capital']))):
        capital_length = len(tokenizer(f' {capital},', add_special_tokens=False).input_ids)
        clean = f' {capital}, whose capital,'
        country_idx = tokenizer(f' {country.strip()}', add_special_tokens=False).input_ids[0]

        valid_corrupted_capitals = length_city_dict[capital_length]
        if len(valid_corrupted_capitals) == 1:
            continue
        corrupted_capital = capital
        while corrupted_capital == capital:
            corrupted_capital = random.choice(valid_corrupted_capitals)

        corrupted_country = capital_country_mapping[corrupted_capital]
        corrupted = f' {corrupted_capital}, whose capital,'

        corrupted_country_idx = tokenizer(f' {corrupted_country.strip()}', add_special_tokens=False).input_ids[0]

        for k, v in zip(['clean', 'capital', 'country', 'country_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_country_idx'], [clean, capital, country, country_idx, corrupted, corrupted_capital, corrupted_country, corrupted_country_idx]):
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
