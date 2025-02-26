# %%
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from fractions import Fraction
from wuggy import WuggyGenerator
from nltk.corpus import words as nltk_word_list
from pluralizer import Pluralizer
from transformers import AutoTokenizer

from eap.utils import model2family
#%%
def generate_words():
    with open('words.txt', 'r') as f:
        words = f.readlines()

    words = [word.strip() for word in words][:500]
    generated_words = set()
    nltk_word_set = set(x.lower() for x in nltk_word_list.words())
    g = WuggyGenerator()
    g.load("orthographic_english")
    ncandidates = 5
    for word in tqdm(words):
        try:
            g.set_reference_sequence(g.lookup_reference_segments(word))
        except AttributeError:
            continue
        g.set_attribute_filter('sequence_length')
        g.set_attribute_filter('segment_length')
        g.set_statistic('overlap_ratio')
        g.set_statistic('plain_length')
        g.set_statistic('transition_frequencies')
        g.set_statistic('lexicality')
        g.set_statistic('ned1')
        #g.set_output_mode('syllabic')
        j = 0
        for i in range(1, 10, 1):
            g.set_frequency_filter(2**i, 2**i)
            for sequence in g.generate_advanced(clear_cache=False):
                if (g.statistics['overlap_ratio'] == Fraction(2, 3) and
                        g.statistics['lexicality'] == "N" and sequence not in nltk_word_set):
                    generated_words.add(sequence)
                    j = j+1
                    if j > ncandidates:
                        break
            if j > ncandidates:
                break
    
    singular, plural = [], []
    p = Pluralizer()
    pluralized_nltk_words = set(p.plural(word) for word in nltk_word_set)
    all_nltk_words = nltk_word_set.union(pluralized_nltk_words)

    for gw in generated_words:
        if len(gw) > 10 or len(gw) < 2 or (gw[-1] == 's' and gw[-2] != 's') or gw[-1] == 'z' or gw[-1] == 'h' or gw[-1] == 'j':
            continue 
        gw_plural = p.plural(gw)
        if gw_plural not in all_nltk_words:
            singular.append(gw)
            plural.append(gw_plural)

    df = pd.DataFrame({'singular': singular, 'plural': plural})
    df.to_csv('generated_words.csv', index=False)

# %%
def create_dataset(model_name:str):
    try:
        df = pd.read_csv('generated_words.csv')
    except FileNotFoundError:
        generate_words()
        df = pd.read_csv('generated_words.csv')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d = defaultdict(list)
    for sing, plur in zip(df['singular'], df['plural']):
        sing_ids = tokenizer(' ' + sing, add_special_tokens=False)['input_ids']
        plur_ids = tokenizer(' ' + plur, add_special_tokens=False)['input_ids']
        sing_prefix = sing_ids[:-1]
        plur_prefix = plur_ids[:-1]
        if not(sing_prefix and sing_prefix == plur_prefix):
            continue

        prefix_text = tokenizer.decode(sing_prefix)
        # for some reason, mistral doesn't give you the space in front of tokens when calling decode??
        if model2family(model_name) == 'mistral':
            prefix_text = ' ' + prefix_text
        #plur_prefix_text = tokenizer.decode(plur_prefix)
        singular_sentence = f'First, there was one {sing}. Now there are two{prefix_text}'
        plural_sentence = f'First, there were two {plur}. Now there is one{prefix_text}'

        # first we add one way 
        d['clean'].append(singular_sentence)
        d['corrupted'].append(plural_sentence)
        d['correct_idx'].append(plur_ids[-1])
        d['incorrect_idx'].append(sing_ids[-1])
        d['nonce_singular'].append(sing)
        d['nonce_plural'].append(plur)
        d['label'].append(0)

        # then the other 
        d['clean'].append(plural_sentence)
        d['corrupted'].append(singular_sentence)
        d['correct_idx'].append(sing_ids[-1])
        d['incorrect_idx'].append(plur_ids[-1])
        d['nonce_singular'].append(sing)
        d['nonce_plural'].append(plur)
        d['label'].append(1)
        #d['prefix'].append(sing_prefix)
        #d['singular_id'].append(sing_ids[-1])
        #d['plural_id'].append(plur_ids[-1])

    output_df = pd.DataFrame(d)
    output_df.to_csv(f'{model2family(model_name)}.csv', index=False)

create_dataset('google/gemma-2-2b')
create_dataset('meta-llama/Meta-Llama-3-8B')
create_dataset('Qwen/Qwen2-1.5B')
create_dataset('mistralai/Mistral-7B-v0.3')
create_dataset('allenai/OLMo-7b-hf')

# %%
model_name = 'mistralai/Mistral-7B-v0.3'
try:
    df = pd.read_csv('generated_words.csv')
except FileNotFoundError:
    generate_words()
    df = pd.read_csv('generated_words.csv')

tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
d = defaultdict(list)
for sing, plur in zip(df['singular'], df['plural']):
    sing_ids = tokenizer(' ' + sing, add_special_tokens=False)['input_ids']
    plur_ids = tokenizer(' ' + plur, add_special_tokens=False)['input_ids']
    sing_prefix = sing_ids[:-1]
    plur_prefix = plur_ids[:-1]
    if not(sing_prefix and sing_prefix == plur_prefix):
        continue

    sing_prefix_text = tokenizer.decode(sing_prefix)
    plur_prefix_text = tokenizer.decode(plur_prefix)
    singular_sentence = f'First, there was one {sing}. Now there are two{plur_prefix_text}'
    plural_sentence = f'First, there were two {plur}. Now there is one{sing_prefix_text}'
# %%
