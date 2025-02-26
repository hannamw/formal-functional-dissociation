#%%
from typing import Optional, List, Union, Literal, Tuple
from functools import partial 
from pathlib import Path
import json

import spacy
import pandas as pd
import torch 
from torch.nn.functional import kl_div
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from utils import model2family

task_to_defaults = {
    'ioi': ('logit_diff', 2.5),
    'ioi-abb': ('logit_diff', 2.5),
    'ioi-dana': ('logit_diff', 2.0),
    'colored-objects': ('logit_diff', 1),
    'entity-tracking': ('logit_diff', 1.25),
    'greater-than-multitoken': ('prob_diff', 2.5),
    'fact-retrieval-comma': ('logit_diff', 2.5),
    'fact-retrieval-rev': ('logit_diff', 2.5),
    'fact-retrieval-rev-multilingual-en': ('logit_diff', 2.5),
    'fact-retrieval-rev-multilingual-nl': ('logit_diff', 2.5),
    'fact-retrieval-rev-multilingual-de': ('logit_diff', 2.5),
    'fact-retrieval-rev-multilingual-fr': ('logit_diff', 2.5),
    'gendered-pronoun': ('logit_diff', 2.5),
    'gender-bias': ('logit_diff', 2.5),
    'math': ('logit_diff', 3),
    'math-add': ('logit_diff', 3),
    'math-sub': ('logit_diff', 3),
    'math-mul': ('logit_diff', 3),
    'sva': ('prob_diff', 2.5),
    'sva-multilingual-en': ('prob_diff', 2.5),
    'sva-multilingual-nl': ('prob_diff', 2.5),
    'sva-multilingual-de': ('prob_diff', 2.5),
    'sva-multilingual-fr': ('prob_diff', 2.5),
    'hypernymy-comma': ('prob_diff', 2.5),
    'npi': ('logit_diff', 2.5),
    'hypernymy': ('prob_diff', 2.5),
    'greater-than': ('prob_diff', 2.5),
    'greater-than-price': ('prob_diff', 2.5),
    'greater-than-sequence': ('prob_diff', 2.5),
    'wug': ('logit_diff', 2.5),
    'echo': ('logit_diff', 2.5),
    'counterfact-citizen_of': ('logit_diff', 2.0), 
    'counterfact-official_language': ('logit_diff', 2.0), 
    'counterfact-has_profession': ('logit_diff', 2.0), 
    'counterfact-plays_instrument': ('logit_diff', 2.0),
    'counterfact-all': ('logit_diff', 2.0),
}

def get_metric(metric_name: str, task: str, tokenizer:Optional[PreTrainedTokenizer]=None, model: Optional[HookedTransformer]=None):
    if tokenizer is None and model is not None:
        tokenizer = model.tokenizer
    if metric_name == 'kl_divergence' or metric_name == 'kl':
        return partial(divergence, divergence_type='kl')
    elif metric_name == 'js_divergence' or metric_name == 'js':
        return partial(divergence, divergence_type='js')
    elif metric_name == 'accuracy' and (not any(excluded_task in task for excluded_task in ['gendered-pronoun', 'sva', 'npi', 'greater-than'])):
        return accuracy
    elif metric_name == 'logit_diff' or metric_name == 'prob_diff' or 'accuracy' in metric_name:
        prob = ('prob' in metric_name)
        if 'greater-than-multitoken' in task:
            if model is None:
                raise ValueError("model must be set for greater-than-multitoken and prob / logit diff")
            if model2family(model.cfg.model_name) == 'olmo':
                logit_diff_fn = get_logit_diff_greater_than(model.tokenizer)
            else:
                logit_diff_fn = get_logit_diff_greater_than_multitoken(model)
        elif 'greater-than' in task:
            if tokenizer is None:
                raise ValueError("Either tokenizer or model must be set for greater-than and prob / logit diff")
            logit_diff_fn = get_logit_diff_greater_than(tokenizer)
        elif 'hypernymy' in task:
            logit_diff_fn = logit_diff_hypernymy
        elif task == 'sva':
            if model is None:
                raise ValueError("model must be set for sva and prob / logit diff")
            logit_diff_fn = get_logit_diff_sva(model)
        elif 'sva-multilingual' in task:
            #if tokenizer is None:
            #    raise ValueError("Either tokenizer or model must be set for greater-than and prob / logit diff")
            if model is None:
                raise ValueError("model must be set for sva and prob / logit diff")
            language = task.split('-')[-1]
            logit_diff_fn = get_logit_diff_sva_multilingual(model, language)
        else:
            logit_diff_fn = logit_diff

        if 'accuracy' in metric_name:
            return partial(accuracy_wrapper, logit_diff_fn, prob=prob)
        return partial(logit_diff_fn, prob=prob)
    else: 
        raise ValueError(f"got bad metric_name: {metric_name}")

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor, position: int = -1):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits = logits[idx, input_length + position]
    
    return logits

def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl_div(m, p.log(), log_target=True, reduction='none').mean(-1) + kl_div(m, q.log(), log_target=True, reduction='none').mean(-1))

def divergence(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, divergence_type: Union[Literal['kl'], Literal['js']]='kl', mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    if divergence_type == 'kl':
        probs = torch.log_softmax(logits, dim=-1)
        clean_probs = torch.log_softmax(clean_logits, dim=-1)
        results = kl_div(probs, clean_probs, log_target=True, reduction='none').mean(-1)
    elif divergence_type == 'js':
        probs = torch.softmax(logits, dim=-1)
        clean_probs = torch.softmax(clean_logits, dim=-1)
        results = js_div(probs, clean_probs)
    else: 
        raise ValueError(f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'")
    return results.mean() if mean else results

def accuracy(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    top1 = torch.argmax(clean_logits, dim=-1)
    correct = (top1 == labels[:, 0].to(top1.device)).float()
    if mean:
        correct = correct.mean()
    return correct

def accuracy_wrapper(fn, clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    result = fn(clean_logits, corrupted_logits, input_length, labels, mean=False, prob=prob, loss=loss)
    result = (result > 0).float()
    if mean:
        result = result.mean()
    return result

def logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    good_bad = torch.gather(cleans, -1, labels.to(cleans.device))
    results = good_bad[:, 0] - good_bad[:, 1]

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results

def logit_diff_hypernymy(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: List[torch.Tensor], mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

    results = []
    for i, (ls,corrupted_ls) in enumerate(labels):
        r = cleans[i][ls.to(cleans.device)].sum() - cleans[i][corrupted_ls.to(cleans.device)].sum()
        results.append(r)
    results = torch.stack(results)
    
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results


def get_logit_diff_greater_than_multitoken(model: HookedTransformer):
    tokenizer = model.tokenizer
    model_family = model2family(model.cfg.model_name)
    digit_indices = torch.tensor([tokenizer(f'{digit}', add_special_tokens=False).input_ids[-1] for digit in range(10)])
    if model_family == 'llama-3':
        decade_indices = torch.tensor([tokenizer(f'{digit:03d}', add_special_tokens=False).input_ids[0] for digit in range(1000)])
    else:
        decade_indices = digit_indices
        
    def logit_diff_greater_than(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
        if not prob:
            raise ValueError("Can't compute multi-token logit diff")
        clean_logits_dec = get_logit_positions(clean_logits, input_length, position=-2)
        clean_logits_yr = get_logit_positions(clean_logits, input_length, position=-1)
        cleans_dec = torch.softmax(clean_logits_dec, dim=-1)
        cleans_yr = torch.softmax(clean_logits_yr, dim=-1)
                
        results = []
        for decade_probs, year_probs, decade, year in zip(cleans_dec[:, decade_indices], cleans_yr[:, digit_indices], labels[:, 0], labels[:, 1]):
            decade_prob = decade_probs[decade]
            good_decade_probs = decade_probs[decade+1:]
            bad_decade_probs = decade_probs[:decade]
            good_year_probs = year_probs[year+1:]
            bad_year_probs = year_probs[:year+1]
            good_probs = good_decade_probs.sum() + decade_prob * good_year_probs.sum()
            bad_probs = bad_decade_probs.sum() + decade_prob * bad_year_probs.sum()
            results.append(good_probs - bad_probs)
                
        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return logit_diff_greater_than


def get_logit_diff_greater_than(tokenizer: PreTrainedTokenizer):
    year_indices = torch.tensor([tokenizer(f'{year:02d}', add_special_tokens=False).input_ids[0] for year in range(100)])

    def logit_diff_greater_than(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
        # Prob diff (negative, since it's a loss)
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        cleans = cleans[:, year_indices]

        results = []
        if prob:
            for prob, year in zip(cleans, labels):
                results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
        else:
            for logit, year in zip(cleans, labels):
                results.append(logit[year + 1 :].mean() - logit[: year + 1].mean())

        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return logit_diff_greater_than

def get_singular_and_plural(model, strict=False) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out

    df: pd.DataFrame = pd.read_csv('data/sva/combined_verb_list.csv')
    singular = df['sing'].to_list()
    plural = df['plur'].to_list()
    singular_set = set(singular)
    plural_set = set(plural)
    verb_set = singular_set | plural_set
    assert len(singular_set & plural_set) == 0, f"{singular_set & plural_set}"
    singular_indices, plural_indices = [], []

    space_token = tokenizer.convert_ids_to_tokens(tokenizer(" and", add_special_tokens=False).input_ids[0])[0]

    for i in range(tokenizer_length):
        token = tokenizer._convert_id_to_token(i)
        if token is not None:
            if token[0] == space_token:
                token = token[1:]
                if token in verb_set:    
                    if token in singular_set:
                        singular_indices.append(i)
                    else:  # token in plural_set:
                        idx = plural.index(token)
                        third_person_present = singular[idx]
                        third_person_present_tokenized = tokenizer(f' {third_person_present}', add_special_tokens=False)['input_ids']
                        if len(third_person_present_tokenized) == 1 and third_person_present_tokenized[0] != tokenizer.unk_token_id:
                            plural_indices.append(i)
                        elif not strict:
                            plural_indices.append(i)

    if len(singular_indices) == 0 or len(plural_indices) == 0:
        raise ValueError(f"Found missing singular or plural indices: singular ({len(singular_indices)}), plural ({len(plural_indices)})")    
    return torch.tensor(singular_indices, device=model.cfg.device, dtype=torch.long), torch.tensor(plural_indices, device=model.cfg.device, dtype=torch.long)

def get_logit_diff_sva(model, strict=True):
    singular_indices, plural_indices = get_singular_and_plural(model, strict=strict)
    def sva_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False) -> torch.Tensor:
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        
        if prob:
            singular = cleans[:, singular_indices].sum(-1)
            plural = cleans[:, plural_indices].sum(-1)
        else:
            singular = cleans[:, singular_indices].mean(-1)
            plural = cleans[:, plural_indices].mean(-1)

        results = torch.where(labels.to(cleans.device) == 0, singular - plural, plural - singular)
        if loss: 
            results = -results
        if mean:
            results = results.mean()
        return results
    return sva_logit_diff


def get_logit_diff_sva_multilingual_hardcoded(tokenizer: PreTrainedTokenizer, language):
    family = model2family(tokenizer.name_or_path)
    df = pd.read_csv(f'data/sva-multilingual/{language}/singular_plural_{family}.csv')
    singular_indices = torch.tensor([tokenizer(' ' + singular, add_special_tokens=False).input_ids[0] for singular in df['singular']])
    plural_indices = torch.tensor([tokenizer(' ' + plural, add_special_tokens=False).input_ids[0] for plural in df['plural']])

    def sva_multilingual_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False) -> torch.Tensor:
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        
        if prob:
            singular = cleans[:, singular_indices].sum(-1)
            plural = cleans[:, plural_indices].sum(-1)
        else:
            singular = cleans[:, singular_indices].mean(-1)
            plural = cleans[:, plural_indices].mean(-1)

        results = torch.where(labels.to(cleans.device) == 0, singular - plural, plural - singular)
        if loss: 
            results = -results
        if mean:
            results = results.mean()
        return results
    return sva_multilingual_logit_diff


def get_singular_and_plural_multilingual(model: HookedTransformer, language) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out
    family = model2family(model.cfg.model_name)

    if Path(f'data/sva-multilingual/{language}/answer_indices_{family}.pt').exists():
        d = torch.load(f'data/sva-multilingual/{language}/answer_indices_{family}.pt')
        return d['singular_indices'].to(model.cfg.device), d['plural_indices'].to(model.cfg.device)

    if language == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif language == 'de':
        nlp = spacy.load('de_core_news_sm')
    elif language == 'nl':
        nlp = spacy.load('nl_core_news_sm')
    elif language == 'fr':
        nlp = spacy.load('fr_core_news_sm')
    else:
        raise ValueError(f"Unknown language {language}")

    singular_mapping, plural_mapping, inf_mapping = {}, {}, {}

    for i in range(tokenizer_length):
        token = tokenizer._convert_id_to_token(i)
        if token is not None:
            if token[0] == 'Ġ' or ('gemma' in model.cfg.model_name and token[0] ==  '▁'):
                token = token[1:]
                if not len(token):
                    continue
                spacy_token = nlp(token)[0]
                pos = spacy_token.pos_
                verbform = spacy_token.morph.get('VerbForm')
                #tense = spacy_token.morph.get('Tense')
                lemma = spacy_token.lemma_
                number = spacy_token.morph.get('Number')
                if (pos == 'VERB' or pos == 'AUX') and ('Sing' in number or 'Plur' in number) or ((language == 'de' or language == 'nl' or language=='en') and 'Inf' in verbform):
                    if 'Inf' in verbform:  
                        inf_mapping[token] = (i, lemma)
                    elif 'Sing' in number:
                        singular_mapping[token] = (i, lemma)
                    else:  # token in plural_set:
                        plural_mapping[token] = (i, lemma)
    
    singular_lemmas = {lemma for _, lemma in singular_mapping.values()}
    plural_lemmas = {lemma for _, lemma in plural_mapping.values()}
    inf_lemmas = {lemma for _, lemma in inf_mapping.values()}
    both_lemmas = singular_lemmas & (plural_lemmas | inf_lemmas)

    singular_indices, plural_indices = [], []
    singular_tokens, plural_tokens = [], []
    for token, (i, lemma) in singular_mapping.items():
        if lemma in both_lemmas:
            singular_indices.append(i)
            singular_tokens.append(token)
    
    for token, (i, lemma) in plural_mapping.items():
        if lemma in both_lemmas:
            plural_indices.append(i)
            plural_tokens.append(token)

    for token, (i, lemma) in inf_mapping.items():
        if lemma in both_lemmas:
            plural_indices.append(i)
            plural_tokens.append(token)

    
    singular_indices =  torch.tensor(singular_indices, dtype=torch.long)
    plural_indices = torch.tensor(plural_indices, dtype=torch.long)

    d = {'singular_indices': singular_indices, 'plural_indices': plural_indices}
    torch.save(d, f'data/sva-multilingual/{language}/answer_indices_{family}.pt')

    d2 = {'singular_tokens': singular_tokens, 'plural_tokens': plural_tokens}
    with open(f'data/sva-multilingual/{language}/answer_tokens_{family}.json', 'w') as f:
        json.dump(d2, f)

    return singular_indices.to(model.cfg.device), plural_indices.to(model.cfg.device)

def get_logit_diff_sva_multilingual(model, language):
    singular_indices, plural_indices = get_singular_and_plural_multilingual(model, language)
    def sva_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False) -> torch.Tensor:
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        
        if prob:
            singular = cleans[:, singular_indices].sum(-1)
            plural = cleans[:, plural_indices].sum(-1)
        else:
            singular = cleans[:, singular_indices].mean(-1)
            plural = cleans[:, plural_indices].mean(-1)

        results = torch.where(labels.to(cleans.device) == 0, singular - plural, plural - singular)
        if loss: 
            results = -results
        if mean:
            results = results.mean()
        return results
    return sva_logit_diff
# %%
