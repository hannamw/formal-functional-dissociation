import json

import nodebox_linguistics_extended as nle 

singular_hypernyms = {'country', 'crime', 'sport', 'weather', 'music', "vegetable", "fish"}
capital_hypernyms = {'country'}

def format_hyponym(h: str, hypernym: str) -> str:
    h = h.strip()
    if hypernym not in singular_hypernyms:
        h = nle.plural.noun_plural(h)
    if hypernym in capital_hypernyms:
        h = h.capitalize()
    return h