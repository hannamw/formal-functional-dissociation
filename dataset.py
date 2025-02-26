#%%
from functools import partial
from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from eap.utils import model2family

def collate_EAP(xs, task):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    if 'hypernymy' not in task:
        labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, task:str, model_name:str, filename:Optional[str]=None):
        if filename is None:
            if 'multilingual' in task:
                task_parts = task.split('-')
                self.df = pd.read_csv(f'data/{"-".join(task_parts[:-1])}/{task_parts[-1]}/{model2family(model_name)}.csv')
            if 'dummy' in task:
                dummy_idx = int(task.split('-')[-1])
                task = 'dummy'
                self.df = pd.read_csv(f'data/{task}/{model2family(model_name)}.csv')
                self.df = self.df[self.df['split_number'] == dummy_idx]
                assert len(self.df) > 0, f"No data for dummy task {dummy_idx}"
            else:
                self.df = pd.read_csv(f'data/{task}/{model2family(model_name)}.csv')
        else: 
            self.df = pd.read_csv(f'data/{task}/{filename}')

        self.task = task
        self.model_name = model_name
        self.model_family = model2family(model_name)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = None
        if self.task == 'ioi' or self.task == 'ioi-abb' or self.task=='ioi-dana':
            label = [row['correct_idx'], row['incorrect_idx']]
        elif 'greater-than-multitoken' in self.task:
            if self.model_family == 'olmo':
                label = row['label1']
            else:
                label = [row['label1'], row['label2']]
        elif 'greater-than' in self.task:
            label = row['correct_idx']
        elif 'math' in self.task:
            label = [row['answer_idx'], row['corrupted_idx']]
        elif self.task == 'echo':
            label = [row['correct_idx'], row['incorrect_idx']]
        elif self.task == 'wug':
            label = [row['correct_idx'], row['incorrect_idx']]
        elif 'hypernymy' in self.task:
            answer = torch.tensor(eval(row['answers_idx']))
            corrupted_answer = torch.tensor(eval(row['corrupted_answers_idx']))
            label = [answer, corrupted_answer]
        elif 'fact-retrieval-comma' in self.task:
            label = [row['country_idx'], row['corrupted_country_idx']]
        elif 'fact-retrieval-rev' in self.task:
            label = [row['capital_idx'], row['corrupted_capital_idx']]
        elif 'gender' in self.task:
            label = [row['clean_answer_idx'], row['corrupted_answer_idx']]
        elif self.task == 'sva':
            label = row['plural']
        elif self.task == 'colored-objects':
            label = [row['clean_idx'], row['corrupted_idx']]
        elif self.task == 'npi':
            label = [row['clean_idx'], row['corrupted_idx']]
        elif 'dummy' in self.task:
            label = 0 
        elif self.task == 'entity-tracking':
            label = [row['label'], row['corrupted_label']]
        elif 'sva-multilingual' in self.task:
            label = row['label']
        elif 'counterfact' in self.task:
            label = [row['label'], row['corrupted_label']]
        else:
            raise ValueError(f'Got invalid task: {self.task}')
        return row['clean'], row['corrupted'], label
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(collate_EAP, task=self.task))