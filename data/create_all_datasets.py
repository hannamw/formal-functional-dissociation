#%%
import importlib

model_name = 'mistralai/Mistral-7B-v0.3'
tasks = ['ioi', 'colored-objects', 'entity-tracking', 'greater-than-multitoken', 'fact-retrieval-comma', 'fact-retrieval-rev', 'gendered-pronoun', 'sva', 'hypernymy-comma', 'npi']

for task in tasks:
    module = importlib.import_module(f'{task}.create_dataset')
    module.create_dataset(model_name)
# %%

