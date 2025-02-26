#%%
import pandas as pd 

from eap.utils import model2family
#%%
def create_dataset(model_name:str):
    df = pd.read_csv(f'../math/{model2family(model_name)}.csv')
    df = df[df['operation'] == '-']
    df.to_csv(f'{model2family(model_name)}.csv', index=False)


#%%
if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3-8B'
    create_dataset(model_name)
