#%%
from transformer_lens import HookedTransformer
import pandas as pd 
import numpy as np

from prompt_generation import generate_prompts
from eap.utils import model2family
#%%
def create_dataset(model_name:str):
    max_op = 300
    op_ranges = {'+': (0, max_op), '-': (0, max_op), '*': (0, max_op), '/': (1, max_op)}

    model = HookedTransformer.from_pretrained(model_name)
    op_prompts = generate_prompts(model, operand_ranges=op_ranges, correct_prompts=True, num_prompts_per_operator=None)
    
    OPS = ['+', '-', '*']
    ops_prompts_answers = [(op, prompt, answer) for promptlist, op in zip(op_prompts[:3], OPS) for prompt, answer in promptlist]
    ops, prompt_texts, answers = zip(*ops_prompts_answers)
    
    #sampling corrupted answers
    corrupted = []
    for op_promptlist in op_prompts[:3]:
        indices = np.random.permutation(len(op_promptlist))
        orig = np.arange(len(op_promptlist))
        while np.any(indices == orig):
            indices = np.random.permutation(len(op_promptlist))
        op_corrupted = [op_promptlist[i] for i in indices]
        corrupted.extend(op_corrupted)

    corrupted_prompts, corrupted_answers = zip(*corrupted)
    
    answer_idx = [model.tokenizer(answer, add_special_tokens=True).input_ids[1:] for answer in answers]
    corrupted_idx = [model.tokenizer(answer, add_special_tokens=True).input_ids[1:] for answer in corrupted_answers]
    df = pd.DataFrame({'operation': ops, 'clean': prompt_texts, 'answer': answers, 'corrupted': corrupted_prompts, 'corrupted_answer': corrupted_answers, 'answer_idx': answer_idx, 'corrupted_idx': corrupted_idx})
    df.to_csv(f'{model2family(model_name)}.csv', index=False)


#%%
if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3-8B'
    create_dataset(model_name)
