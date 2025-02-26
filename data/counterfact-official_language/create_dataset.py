#%%
import json
import random
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eap.utils import model2family

with open('counterfact.json', 'r') as f:
    data = json.load(f)

data = [(item['requested_rewrite']['subject'], item['requested_rewrite']['target_true']['str']) for item in data if item['requested_rewrite']['relation_id'] == 'P37']
# %%
model_name = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda()
#%%
prompt = 'The official language in {subject} is'
data_with_tokens = []
correct_probs = []
for subject, label in tqdm(data):
    subject_tokens = tokenizer(f' {subject}', add_special_tokens=False).input_ids
    label_tokens = tokenizer(f' {label}', add_special_tokens=False).input_ids
    data_with_tokens.append((subject, label, subject_tokens, label_tokens))

    with torch.inference_mode():
        logits = model(**tokenizer(prompt.format(subject=subject), return_tensors='pt').to('cuda')).logits.cpu()
        probs = torch.softmax(logits, -1).squeeze(0)[-1]
        prob = probs[label_tokens[0]].item()
        correct_probs.append(prob)
# %%
threshold = 0.5
data_with_tokens_filtered = [x for x, prob in zip(data_with_tokens, correct_probs) if prob > threshold]
print(len(data_with_tokens_filtered), len(data_with_tokens))
# %%
length_to_subject = defaultdict(set)
subject_to_label = {}
label_to_first_token = {}
for subject, label, subject_tokens, label_tokens in data_with_tokens_filtered:
    length_to_subject[len(subject_tokens)].add(subject)
    assert subject not in subject_to_label
    subject_to_label[subject] = label
    if label in label_to_first_token:
        assert label_to_first_token[label] == label_tokens[0]
    label_to_first_token[label] = label_tokens[0]

#%%
d = {'clean': [], 'corrupted': [], 'label': [], 'corrupted_label': []}
for subject, label, subject_tokens, label_tokens in data_with_tokens_filtered:
    subject_length = len(subject_tokens)
    potential_subjects = length_to_subject[subject_length]
    valid_subjects = [subj for subj in potential_subjects if label_to_first_token[subject_to_label[subj]] != label_to_first_token[label]]
    if len(valid_subjects) == 0:
        print(f'no valid subjects for {subject}')
        continue
    
    j = 0
    clean_len = 0
    corrupted_len = 1
    while j < 100 and clean_len != corrupted_len:
        i = random.randint(0, len(valid_subjects) - 1)
        corrupted_subject = valid_subjects[i]
        clean = prompt.format(subject=subject)
        corrupted = prompt.format(subject=corrupted_subject)
        clean_len = len(tokenizer(clean, add_special_tokens=False).input_ids)
        corrupted_len = len(tokenizer(corrupted, add_special_tokens=False).input_ids)
        j += 1

    if j >= 100:
        print(f'no valid subjects for {subject} (timeout)')
        continue

    d['clean'].append(clean)
    d['corrupted'].append(corrupted)
    d['label'].append(label_to_first_token[label])
    d['corrupted_label'].append(label_to_first_token[subject_to_label[corrupted_subject]])

# %%
df = pd.DataFrame(d)
df.to_csv(f'{model2family(model_name)}.csv', index=False)
# %%
