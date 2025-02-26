#%%
import json
import random
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eap.utils import model2family


model_name = 'meta-llama/Meta-Llama-3-8B'
dfs = [pd.read_csv(f'../{taskname}/{model2family(model_name)}.csv') for taskname in ['counterfact-plays_instrument', 'counterfact-has_profession', 'counterfact-citizen_of', 'counterfact-official_language']]

df = pd.concat(dfs)
df = df.sample(len(df))
df.to_csv(f'{model2family(model_name)}.csv', index=False)
# %%
