#%%
import random 
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer

from eap.utils import model2family
#%%

def create_dataset(model_name: str):
    #model_name = 'meta-llama/Meta-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_word_token = tokenizer.convert_ids_to_tokens(tokenizer(" the the")['input_ids'][-1])[0]

    candidates = set()
    for i in range(1000, len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(i)
        if token.startswith(start_word_token) and len(tokenizer(' ' + token[1:], add_special_tokens=False)['input_ids']) == 1:
            # Ġ  ▁
            candidates.add(token[1:])

    n_shots = 2
    n_examples_per_shot = 4
    candidate_tuple = tuple(candidates)
    d = defaultdict(list)
    for i in range(10000):
        samples = random.choices(candidate_tuple, k=(n_shots + 1) * n_examples_per_shot + 1)
        clean = ''
        corrupted = ''
        for i in range(n_shots):
            ex = samples[i*n_examples_per_shot:(i+1)*n_examples_per_shot]
            clean += f" {' '.join(ex)} : {ex[-1]} \n"
            corrupted += f" {' '.join(ex)} : {ex[-1]} \n"
        final_batch = samples[-(n_examples_per_shot + 1):-2]
        clean_label = samples[-2]
        corrupted_label = samples[-1]
        
        clean += f"{ ' '.join(final_batch)} {clean_label} :"
        corrupted += f"{ ' '.join(final_batch)} {corrupted_label} :"
        clean_idx = tokenizer.convert_tokens_to_ids(start_word_token + clean_label)
        corrupted_idx = tokenizer.convert_tokens_to_ids(start_word_token + corrupted_label)
        d['clean'].append(clean) 
        d['corrupted'].append(corrupted)
        d['correct_idx'].append(clean_idx)
        d['incorrect_idx'].append(corrupted_idx)

    df = pd.DataFrame(d)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)

create_dataset('google/gemma-2-2b')
create_dataset('meta-llama/Meta-Llama-3-8B')
create_dataset('Qwen/Qwen2-1.5B')
create_dataset('mistralai/Mistral-7B-v0.3')
create_dataset('allenai/OLMo-7b-hf')
