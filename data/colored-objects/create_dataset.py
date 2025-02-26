#%%
import argparse
from typing import List, Dict
import random

import pandas as pd
from num2words import num2words
from transformers import AutoTokenizer

from eap.utils import model2family

# removed fidget spinner
# objects = [
# "pencil",
# "notebook",
# "pen",
# "crayon",
# "cup",
# "plate",
# "jug",
# "mug",
# "puzzle",
# "textbook",
# "paperclip",
# "scrunchie",
# "phone charger",
# "booklet",
# "envelope",
# "cat toy",
# "stress ball",
# "keychain",
# "necklace",
# "bracelet",
# "teddy bear",
# "sheet of paper",
# "dog leash",
# "pair of sunglasses",
# ]
objects = ["pencil", "notebook", "pen", "cup", "plate", "jug", "mug", "puzzle", "textbook", "leash", "necklace", "bracelet", "bottle", "ball", "envelope", "lighter", "bowl"]
all_colors = [
"red",
"orange",
"yellow",
"green",
"blue",
"brown",
"magenta",
"fuchsia",
"mauve",
"teal",
"turquoise",
"burgundy",
"silver",
"gold",
"black",
"grey",
"purple",
"pink"]

all_simple_colors = [
"red",
"orange",
"yellow",
"green",
"blue",
"purple",
"brown",
"black"]

# removed nightstand
surfaces = ["table", "desk", "floor"]
subject_starts = ["there is", "you see", "I see"]

def generate_items_text(chosen_colors, chosen_objects):

    items = []
    for color, obj in zip(chosen_colors, chosen_objects):
        item = "a{} {} {}".format("n" if color[0] in "aeiou" else "", color, obj)
        items.append(item)

    if len(chosen_objects) == 2:
        items_text = " and ".join(items)
    else:
        items_text = ", ".join(items[:-1]) + ", and " + items[-1]
    return items_text

def rand_bool():
    return random.random() > 0.5

def gen_count_scores(true_count, max_number=6):
    scores = {}
    for i in range(max_number+1):
        scores[str(i)] = (1 if i == true_count else 0)
        scores[textify_number(i)] = (1 if i == true_count else 0)
    return scores

def gen_color_scores(true_color):
    return {c: (1 if true_color == c else 0) for c in all_colors}

def pluralize_obj(obj):
    if obj == "sheet of paper":
        return "sheets of paper"
    elif obj == "dog leash":
        return "dog leashes"
    elif obj == "pair of sunglasses":
        return "pairs of sunglasses"
    else:
        return obj + "s"

def textify_number(i):
    return num2words(i)


def generate_what_color_examples(objects: List[str], all_colors: List[str], max_objects:int, color2idx: Dict[str,int], seed=666, num_to_generate=50):
    random.seed(seed)

    sentence_pattern = "On the {surface}, {ss} {items}. What color is the {q_object}?"

    clean_sentences, clean_labels, clean_idxs, corrupted_sentences, corrupted_labels, corrupted_idxs = [], [], [], [], [], []
    for i in range(num_to_generate):
        q_color = True 
        corrupted_q_color = True
        while q_color == corrupted_q_color:
            # demo
            surface = random.choice(surfaces)
            subject_start = random.choice(subject_starts)
            num_objects = random.randint(2, max_objects)

            chosen_colors = random.sample(all_colors, num_objects)
            chosen_objects = random.sample(objects, num_objects)
            
            items_text = generate_items_text(chosen_colors, chosen_objects)

            q_index = random.randint(0, num_objects-1)
            q_object = chosen_objects[q_index]
            demo_q_color = chosen_colors[q_index]

            demo_sentence = sentence_pattern.format(
                    surface=surface,
                    ss=subject_start,
                    q_object=q_object,
                    items=items_text)
            
            # actual sentence
            surface = random.choice(surfaces)
            subject_start = random.choice(subject_starts)
            num_objects = random.randint(2, max_objects)

            chosen_colors = random.sample(all_colors, num_objects)
            chosen_objects = random.sample(objects, num_objects)
            #chosen_object_lens = [object_lens[obj] for obj in chosen_objects]
            
            items_text = generate_items_text(chosen_colors, chosen_objects)

            q_index = random.randint(0, num_objects-1)
            q_object = chosen_objects[q_index]
            q_color = chosen_colors[q_index]

            partial_sentence = sentence_pattern.format(
                    surface=surface,
                    ss=subject_start,
                    q_object=q_object,
                    items=items_text)
            
            sentence = f"Q: {demo_sentence} A: {demo_q_color} Q: {partial_sentence} A:"
            
            # again for the corrupted sentence
            chosen_colors = random.sample(all_colors, num_objects)
            chosen_objects = chosen_objects = random.sample(objects, num_objects) #[random.sample(len_to_obj[l], 1)[0] for l in chosen_object_lens]
            
            items_text = generate_items_text(chosen_colors, chosen_objects)

            q_index = random.randint(0, num_objects-1)
            q_object = chosen_objects[q_index]
            corrupted_q_color = chosen_colors[q_index]

            partial_sentence = sentence_pattern.format(
                    surface=surface,
                    ss=subject_start,
                    q_object=q_object,
                    items=items_text)
            
            corrupted_sentence = f"Q: {demo_sentence} A: {demo_q_color} Q: {partial_sentence} A:"
        
        clean_sentences.append(sentence)
        clean_labels.append(q_color)
        clean_idxs.append(color2idx[q_color])
        corrupted_sentences.append(corrupted_sentence)
        corrupted_labels.append(corrupted_q_color)
        corrupted_idxs.append(color2idx[corrupted_q_color])

    d = {'clean': clean_sentences, 'clean_label': clean_labels, 'clean_idx': clean_idxs, 'corrupted': corrupted_sentences, 'corrupted_label': corrupted_labels, 'corrupted_idx': corrupted_idxs}
    df = pd.DataFrame.from_dict(d)
    return df

def create_dataset(model, all_colors=all_colors, objects=objects):
    max_objects = 4
    tokenizer = AutoTokenizer.from_pretrained(model)
    all_colors = [c for c in all_colors if len(tokenizer(f' {c}', add_special_tokens=False)['input_ids']) == 1]
    color2idx = {c:tokenizer(f' {c}', add_special_tokens=False)['input_ids'][0] for c in all_colors}
    print(color2idx.keys())
    objects = [c for c in objects if len(tokenizer(f' {c}', add_special_tokens=False)['input_ids']) == 1]

    # object_lens = {obj: len(tokenizer(f' {obj}')['input_ids']) for obj in objects}
    # counter = Counter(object_lens.values())
    # object_lens = {k:v for k,v in object_lens.items() if counter[v] > 1}
    # objects = list(object_lens.keys())
    # len_to_obj = defaultdict(list)
    # for k,v in object_lens.items():
    #     len_to_obj[v].append(k)

    df = generate_what_color_examples(objects, all_colors, max_objects, color2idx, num_to_generate=1000)

    for clean, corr in zip(df['clean'], df['corrupted']):
        cleantok = tokenizer(clean)['input_ids']
        corrtok = tokenizer(corr)['input_ids']
        if len(cleantok) != len(corrtok):
            print(clean, corr)
            raise ValueError()
    df.to_csv(f'{model2family(model)}.csv', index=False)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.model)
