#%%
import json 
import pandas as pd

from eap.utils import display_name_dict 

tasks = ['ioi', 
         'fact-retrieval-comma', 
         'gendered-pronoun', 
         'sva', 
         'entity-tracking', 
         'colored-objects', # evaluate starting from here
         'npi', 
         'hypernymy-comma', 
         'fact-retrieval-rev', 
         'greater-than-multitoken',
         'wug'
         ]

llama_tasks = ['echo', 'math', 'math-add', 'math-sub', 'math-mul'] + ['counterfact-citizen_of', 'counterfact-official_language', 'counterfact-has_profession', 'counterfact-plays_instrument']
    
gemma_tasks = ['sva-multilingual-en', 'sva-multilingual-nl', 'sva-multilingual-de', 'sva-multilingual-fr', 'fact-retrieval-rev-multilingual-en', 'fact-retrieval-rev-multilingual-nl', 'fact-retrieval-rev-multilingual-de', 'fact-retrieval-rev-multilingual-fr']

models = [x.split('/')[-1] for x in ['meta-llama/Meta-Llama-3-8B', 'Qwen/Qwen2.5-7B', 'allenai/OLMo-7B-hf', 'mistralai/Mistral-7B-v0.3', 'google/gemma-2-2b']]

accuracies = {task: [] for task in tasks}
llama_accuracies = {task: [] for task in llama_tasks}
gemma_accuracies = {task: [] for task in gemma_tasks}

for model in models:
    accuracy_path = f'{model}/json/accuracies.json'
    def load_accuracies(path):
        with open(path, 'r') as f:
            return json.load(f)


    model_accuracies = load_accuracies(accuracy_path)
    
    try:
        for task in tasks:
                accuracies[task].append(model_accuracies[task])
            
        if 'llama' in model.lower():
            for task in llama_tasks:
                llama_accuracies[task].append(model_accuracies[task])
            
        if 'gemma' in model.lower():
            for task in gemma_tasks:
                gemma_accuracies[task].append(model_accuracies[task])
    except KeyError as e:
        print(f"Task {task} not found in {model}")
        raise e

accuracies = {display_name_dict[k]: v for k, v in accuracies.items()}
llama_accuracies = {display_name_dict[k]: v for k, v in llama_accuracies.items()}
gemma_accuracies = {display_name_dict[k]: v for k, v in gemma_accuracies.items()}

# Create dataframes
df = pd.DataFrame(accuracies, index=models).transpose()
df = df.style.format(precision=2)
llama_df = pd.DataFrame(llama_accuracies, index=[model for model in models if 'llama' in model.lower()]).transpose()
llama_df = llama_df.style.format(precision=2)
gemma_df = pd.DataFrame(gemma_accuracies, index=[model for model in models if 'gemma' in model.lower()]).transpose()
gemma_df = gemma_df.style.format(precision=2)

# Convert dataframes to LaTeX tables
latex_table = df.to_latex()
llama_latex_table = llama_df.to_latex()
gemma_latex_table = gemma_df.to_latex()

print(latex_table)
print(llama_latex_table)
print(gemma_latex_table)

# Save LaTeX tables to files
#with open('accuracy_table.tex', 'w') as f:
#    f.write(latex_table)

#with open('llama_accuracy_table.tex', 'w') as f:
#    f.write(llama_latex_table)

#with open('gemma_accuracy_table.tex', 'w') as f:
#    f.write(gemma_latex_table)
