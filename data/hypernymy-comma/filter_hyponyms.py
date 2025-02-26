# %%
import json
from transformer_lens import HookedTransformer
import torch 
from collections import Counter, defaultdict
from tqdm import tqdm 

import nodebox_linguistics_extended as nle 
# model_name = 'gpt2-medium'
model_name = 'gpt2'
model = HookedTransformer.from_pretrained(model_name)
k = 5 

with open('battig_manual_filtered.json', 'r') as g:
    d = json.load(g)

def cross_product(xs):
    return [(x, x2) for i, x in enumerate(xs) for j, x2 in enumerate(xs[i:]) if i != j]

def total_cross_product(xs):
    return [(xs[i], xs[j]) for i in range(len(xs)) for j in range(len(xs)) if i != j]

word_probs = defaultdict(Counter)
word_counts = defaultdict(Counter)
cross_product_lens = dict()

for hypernym, hyponyms in tqdm(d.items(), total=len(d)):
    cross = total_cross_product(hyponyms)
    cross_product_lens[hypernym] = len(cross)
    for h1, h2 in cross:
        h1 = nle.plural.noun_plural(h1)
        h2 = nle.plural.noun_plural(h2)
        # s = f'{h1}, {h2}, and other'
        s = f'{h1} and {h2} are two types of'
        with torch.inference_mode():
            probs = torch.softmax(model(s).squeeze(0)[-1], -1)
        topk = torch.topk(probs, k=k)
        topk_indices = topk.indices.cpu().tolist()
        topk_values = topk.values.cpu().tolist()
        topk_tokens = [model.tokenizer.convert_ids_to_tokens(item) for item in topk_indices]
        topk_tokens_values = list(zip(topk_tokens, topk_values))

        word_counts[hypernym].update(topk_tokens)
        for token, value in topk_tokens_values:
            word_probs[hypernym][token] += value

        #print(s, topk_tokens_values)
# set a threshold for usable or not based on % of good answers

top_answer_proportions = {hypernym: [(word,count/cross_product_lens[hypernym]) for word, count in c.most_common(5)] for hypernym, c in word_counts.items()}
top_answer_probs = {hypernym: [(word,prob/cross_product_lens[hypernym]) for word, prob in c.most_common(5)] for hypernym, c in word_probs.items()}
#%%
with open('top_answers.json', 'w') as f:
    json.dump(top_answer_probs, f)
#%%
good_answers = {}
for hypernym, potentials in top_answer_proportions.items():
    best_word, best_prob = potentials[0]
    best_word_list = [best_word]
    best_prob_list = [best_prob]
    for w, p in potentials[1:]:
        if (best_word == nle.plural.noun_plural(w)) or (w == nle.plural.noun_plural(best_word)):
            best_word_list.append(w)
            best_prob_list.append(p)
    good_answers[hypernym] = (best_word_list, best_prob_list)
# %%
for hypernym, hyponyms in d.items():
    good_token_probs = torch.zeros((len(hyponyms), len(hyponyms)))
    for i, h1 in enumerate(hyponyms):
        good_token_ids = torch.tensor(model.tokenizer.convert_tokens_to_ids(good_answers[hypernym][0]), device='cuda')
        for j, h2 in enumerate(hyponyms):
            h1 = nle.plural.noun_plural(h1)
            h2 = nle.plural.noun_plural(h2)
            # s = f'{h1}, {h2}, and other'
            s = f'{h1} and {h2} are two types of'
            #s = f'{h1}, {h2} or other'
            with torch.inference_mode():
                probs = torch.softmax(model(s).squeeze(0)[-1], -1)
            gtp = probs[good_token_ids].sum().cpu().item()
            good_token_probs[i,j] = gtp

    for i in range(len(hyponyms)):
        if i == 0:
            good_token_probs[i,i] = torch.cat([good_token_probs[i+1:, i], good_token_probs[i, i+1:]]).mean()
        elif i == len(hyponyms) - 1:
            good_token_probs[i,i] = torch.cat([good_token_probs[:i, i], good_token_probs[i, :i]]).mean()
        else:
            good_token_probs[i,i] = torch.cat([good_token_probs[:i, i], good_token_probs[i+1:, i], good_token_probs[i, :i], good_token_probs[i, i+1:]]).mean()
    break
# %%
col_mean = good_token_probs.mean(0)
filtered_probs = good_token_probs[col_mean > 0.01]
filtered_probs = filtered_probs[:, col_mean > 0.01]

#%%
for hypernym, (words, freqs) in good_answers.items():
    total = sum(freqs)
    if total >= 0.65: 
        print(hypernym, words, freqs)

#%% 
# manual part