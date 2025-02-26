# %%
import random 
import argparse
import json
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer
import torch 
from tqdm import tqdm 

from eap.utils import model2family

from hypernymy_utils import format_hyponym


def create_dataset(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open('battig_manual_filtered.json', 'r') as g:
        d = json.load(g)

    singular_hypernyms = {'country', 'crime', 'sport', 'music'}
    special_hypernyms = {'weather', 'vegetable'}

    def is_are(h, hypernym):
        if hypernym in special_hypernyms:
            return 'are' if h[-1] == 's' else 'is'
        elif hypernym in singular_hypernyms:
            return 'is'
        else:
            return 'are'

    d = {hypernym: [format_hyponym(hyponym, hypernym) for hyponym in hyponyms] for hypernym, hyponyms in d.items()}

    hyponym_to_hypernym = {hyponym: hypernym for hypernym, hyponyms in d.items() for hyponym in hyponyms}
    hypernym_lengths = {}
    hyponym_lengths = {}
    hyponyms_of_length = defaultdict(list)
    for hypernym, hyponyms in d.items():
        hypernym_length = len(tokenizer(f' {hypernym}', add_special_tokens=False).input_ids)
        hypernym_lengths[hypernym] = hypernym_length
        for hyponym in hyponyms:
            #hyponym = format_hyponym(hyponym, hypernym)
            hyponym_length = len(tokenizer(f' {hyponym}', add_special_tokens=False).input_ids)
            hyponym_lengths[hyponym] = hyponym_length
            hyponyms_of_length[hyponym_length].append(hyponym)

    #assert all(len(set([hyponym_to_hypernym[hyponym] for hyponym in v])) > 1 for v in hyponyms_of_length.values())
    to_delete = []
    for k, v in hyponyms_of_length.items():
        checkset = set([hyponym_to_hypernym[hyponym] for hyponym in v])
        if len(checkset) <= 1:
            hyp = checkset.pop()
            print(f"Found set (length {k}) with only one hypernym: {hyp}")
            print(f"deleting the following hyponyms: {v}")
            for hyponym in v:
                del hyponym_lengths[hyponym]
                i = d[hyp].index(hyponym)
                del d[hyp][i]
                del hyponym_to_hypernym[hyponym]
            to_delete.append(k) 
    for k in to_delete:
        del hyponyms_of_length[k]


    with open('valid_answers.json', 'r') as f:
        valid_answers = json.load(f)

    valid_answers = {k: [x[0] for x in v] for k, v in valid_answers.items()}

    dataset = {k:[] for k in ['clean', 'h', 'hypernym', 'answers', 'answers_idx', 'corrupted', 'c', 'corrupted_hypernym', 'corrupted_answers', 'corrupted_answers_idx']}

    for hypernym, hyponyms in tqdm(list(d.items())):
        for h in hyponyms:
            #h = format_hyponym(h, hypernym)
            h_length = hyponym_lengths[h]
            # clean = f' {h} {is_are(h, hypernym)}, and other'
            clean = f' {h}, and other'
            answers = valid_answers[hypernym]
            answers_idx = [tokenizer(f' {answer.strip()}', add_special_tokens=False).input_ids[0] for answer in answers]

            corrupted_hyponym = hyponym
            found = True
            while corrupted_hyponym == hyponym:
                if len(hyponyms_of_length[h_length]) <= 1:
                    raise ValueError(f"Only one hyponym of length {h_length} for {hypernym}: {hyponym}")
                candidates = [x for x in hyponyms_of_length[h_length] if (x != hyponym) and (hyponym_to_hypernym[x] != hypernym) and (hypernym_lengths[hyponym_to_hypernym[x]] == hypernym_lengths[hypernym])] 
                if not candidates:
                    found = False
                    break
                random_corrupted = random.choice(candidates) #hyponyms_of_length[h_length])
                random_corrupted_hypernym = hyponym_to_hypernym[random_corrupted]
                if hypernym_lengths[hypernym] == hypernym_lengths[random_corrupted_hypernym] and random_corrupted_hypernym != hypernym:
                    corrupted_hyponym = random_corrupted 
                    corrupted_hypernym = random_corrupted_hypernym
            if not found:
                continue

            #corrupted = f' {corrupted_hyponym} {is_are(corrupted_hyponym, corrupted_hypernym)} a type of'
            corrupted = f' {corrupted_hyponym}, and other'
            corrupted_answers = valid_answers[corrupted_hypernym]
            corrupted_answers_idx = [tokenizer(f' {answer.strip()}', add_special_tokens=False).input_ids[0] for answer in corrupted_answers]

            for k, v in zip(['clean', 'h', 'hypernym', 'answers', 'answers_idx', 'corrupted', 'c', 'corrupted_hypernym', 'corrupted_answers', 'corrupted_answers_idx'], [clean, h, hypernym, answers, answers_idx, corrupted, corrupted_hyponym, corrupted_hypernym, corrupted_answers, corrupted_answers_idx]):
                dataset[k].append(v)

    df = pd.DataFrame.from_dict(dataset)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)
