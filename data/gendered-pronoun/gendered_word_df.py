#%% 
import json 
import pandas as pd

male_words, female_words = {}, {}

with open('gendered_words/gendered_words.json', 'r') as f:
    data =  json.load(f)
    for obj in data:
        word = obj['word']
        if 'gender_map' not in obj or '_' in word:
            continue

        gender = obj['gender']
        opposite_gender = 'm' if gender == 'f' else 'f'
        opposite_word = obj['gender_map'][opposite_gender][0]['word']
        
        if gender == 'm':
            male_words[word] = opposite_word
            female_words[opposite_word] = word
        else:
            female_words[word] = opposite_word
            male_words[opposite_word] = word

male, female = list(zip(*male_words.items()))
# %%
df = pd.DataFrame({'male': list(male), 'female':list(female)})
df.to_csv('gendered_words.csv', index=False)
# %%
