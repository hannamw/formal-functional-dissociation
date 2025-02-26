import random
from typing import List, Union
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer

from eap.utils import model2family


def generate_sentences(noun: str, year: int, use_decade=True) -> str:
    century = year // 100
    decade = year // 10
    if use_decade:
        good_sentence = f"The {noun} lasted from the year {year} to the year {decade}"
        bad_sentence = f"The {noun} lasted from the year {century}01 to the year {decade}"
    else:
        good_sentence = f"The {noun} lasted from the year {year} to the year {century}"
        bad_sentence = f"The {noun} lasted from the year {century}01 to the year {century}"
    return good_sentence, bad_sentence

def is_valid_year(year: str, tokenizer) -> bool:
    _year = " " + year
    token = tokenizer(_year)["input_ids"]
    detok = tokenizer.convert_ids_to_tokens(token)
    return len(detok) == 2 and len(detok[1]) == 2


class YearDataset:
    years_to_sample_from: torch.Tensor
    N: int
    ordered: bool

    nouns: List[str]
    years: torch.Tensor
    years_YY: torch.Tensor
    good_sentences: List[str]
    bad_sentences: List[str]
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        years_to_sample_from,
        model_name: str,
        N: int,
        nouns: Union[str, List[str], Path],
        balanced: bool = True,
    ):
        self.model_name = model_name
        self.model_family = model2family(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.years_to_sample_from = years_to_sample_from
        self.N = N

        if isinstance(nouns, str):
            noun_list = [nouns]
        elif isinstance(nouns, list):
            noun_list = nouns
        elif isinstance(nouns, Path):
            with open(nouns, "r") as f:
                noun_list = [line.strip() for line in f]
                noun_list = [noun for noun in noun_list if len(tokenizer(noun, add_special_tokens=False).input_ids) == 1]
        else:
            raise ValueError(f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}")

        self.nouns = random.choices(noun_list, k=N)

        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(N):
                sample_pool = self.years_to_sample_from[years_to_sample_from_YY == current_year]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            self.years = torch.tensor(years)
        else:
            self.years = torch.tensor(self.years_to_sample_from[torch.randint(0, len(self.years_to_sample_from), (N,))])

        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        last_sentences = [
            generate_sentences(noun, int(year.item()), use_decade=(self.model_family!='olmo')) for noun, year in zip(self.nouns, self.years)
        ]
        last_clean, last_corrupted = zip(*last_sentences)

        self.good_sentences = last_clean
        self.bad_sentences = last_corrupted

        if self.model_family == 'gemma' or self.model_family == 'qwen2' or self.model_family == 'mistral':
            self.label1, self.label2 = list(zip(*[((year % 100) // 10 , year % 10) for year in self.years]))
        elif self.model_family == 'llama-3':
            self.label1, self.label2 = list(zip(*[(year // 10 , year % 10) for year in self.years]))
        elif self.model_family == 'olmo':
            self.label1, self.label2 = list(zip(*[(year % 100 , None) for year in self.years]))
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")
        
        self.label1 = [x.item() for x in self.label1]
        if self.model_family != 'olmo':
            self.label2 = [x.item() for x in self.label2]
        

    def __len__(self):
        return self.N

    