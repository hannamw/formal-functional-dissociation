#%%
import pandas as pd 
#%%
model_name = 'gpt2'
model_name_noslash = model_name.split('/')[-1]
filename = f'dataset_{model_name_noslash}.csv'
df = pd.read_csv(filename)

# drop the second hypernym and corrupted
df = df.drop(['h2', 'c2'], axis=1)
#%%
singular_hypernyms = {'country', 'crime', 'sport', 'music'}
special_hypernyms = {'weather', 'vegetable'}

def is_are(h, hypernym):
    if hypernym in special_hypernyms:
        return 'are' if h[-1] == 's' else 'is'
    elif hypernym in singular_hypernyms:
        return 'is'
    else:
        return 'are'

df['clean'] = [f', {h1} and other' for h1 in df['h1']]
df['clean2'] =  [f'{h1} {is_are(h1, hypernym)} a type of' for h1, hypernym in zip(df['h1'], df['hypernym'])]

df['corrupted'] = [f', {h1} and other' for h1 in df['c1']]
df['corrupted2'] =  [f'{h1} {is_are(h1, hypernym)} a type of' for h1, hypernym in zip(df['c1'], df['corrupted_hypernym'])]

#%%
keep = []
seen = set()
for h1, hypernym in zip(df['h1'], df['hypernym']):
    if (h1, hypernym) in seen:
        keep.append(False)
    else:
        seen.add((h1, hypernym))
        keep.append(True) 
df = df[keep]

#%%
df.to_csv(f'dataset_single_{model_name_noslash}.csv', index=False)
# %%
