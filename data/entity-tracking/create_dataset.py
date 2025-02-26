#%%
from eap.utils import model2family

import argparse
import random
import jsonlines
import pandas as pd
from transformers import AutoTokenizer
#%%
def create_dataset(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_csv('finetuning/data/objects.csv')
    
    objects = [' ' + name for name in df['object_name'].to_list()]

    for o in objects:
        if len(tokenizer(o, add_special_tokens=False)['input_ids']) != 1: 
            print(o)

    sentences = []
    ids = []
    labels = []
    with open('finetuning/data/dataset.jsonl', 'r') as f:
        reader = jsonlines.Reader(f)
        for line in reader:
            full_sentence = line['sentence'][:-1]
            split_sentence = full_sentence.split(' ')
            rejoined_sentence = ' '.join(split_sentence[:-1])
            label = split_sentence[-1]
            label_token = tokenizer(' ' + label, add_special_tokens=False)['input_ids']
            assert len(label_token) == 1, f'{label_token} {label}'
            label_token = label_token[0]

            sentences.append(rejoined_sentence)
            labels.append(label_token)
            ids.append(line['sample_id'])

    data_df = pd.DataFrame({'clean': sentences, 'sample_id': ids, 'label': labels})

    # sentence_lengths = [len(tokenizer(s, add_special_tokens=False)['input_ids']) for s in sentences]
    corrupted_sentences = []
    corrupted_labels = []
    for sample_id, sentence in zip(ids, sentences):
        n_id_match = (data_df['sample_id'] == sample_id).sum()
        corrupted_sentence = sentence 
        while corrupted_sentence == sentence:
            random_id = random.randint(0, n_id_match-1)
            row = data_df[data_df['sample_id'] == sample_id].iloc[random_id]
            corrupted_sentence = row['clean']
            corrupted_label = row['label']
        corrupted_sentences.append(corrupted_sentence)
        corrupted_labels.append(corrupted_label)

    data_df['corrupted'] = corrupted_sentences 
    data_df['corrupted_label'] = corrupted_labels
    data_df.to_csv(f'{model2family(model_name)}.csv', index=False)
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)

# %%
