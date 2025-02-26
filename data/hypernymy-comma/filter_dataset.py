#%%
import pandas as pd

model_name = 'EleutherAI/pythia-160m'
model_name_noslash = model_name.split('/')[-1]
df = pd.read_csv(f'dataset_single_{model_name_noslash}.csv')
df2 = pd.read_csv('dataset_single.csv')
keep = (df2['prob_diff'] > 0.1).to_numpy()
df = df[keep]
df.to_csv(f'{model_name_noslash}.csv', index=False)
# %%
