# data_utils.py

import pandas as pd
from rapidfuzz import process, fuzz
from nltk.corpus import wordnet as wn

STANDARD_COLUMNS = ["date", "region", "product", "quantity", "profit"]

def normalize_columns(df: pd.DataFrame, threshold: int = 80) -> pd.DataFrame:
    """
    Renombra columnas usando fuzzy matching + WordNet.
    """
    orig = list(df.columns)
    mapping = {}
    for std in STANDARD_COLUMNS:
        match, score, _ = process.extractOne(std, orig, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            mapping[match] = std
            continue
        for col in orig:
            syns = wn.synsets(col, lang='eng') or wn.synsets(col)
            lemmas = {l.lower().replace('_',' ') for s in syns for l in s.lemma_names()}
            if std in lemmas:
                mapping[col] = std
                break
    return df.rename(columns=mapping)
