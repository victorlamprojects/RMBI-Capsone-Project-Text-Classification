# Global Config
from common.GlobalConfig import *

import numpy as np
import pandas as pd
from string import punctuation
import re
# NLTK
from nltk.corpus import stopwords
# Spacy
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from langdetect import DetectorFactory
nlp = spacy.load('en_core_web_lg')
import os
if not os.path.exists(clean_folder_path):
    os.makedirs(clean_folder_path)

category = pd.read_csv(hashtag_list_path)
stop_words = set(stopwords.words('english') + list(punctuation) + ['-PRON-'])
clean_result = pd.DataFrame(columns=['tag', 'count', 'avg.length'])

# Add language detector
def create_lang_detector(nlp, name):
    # fix the random behaviour
    DetectorFactory.seed = random_seed
    return LanguageDetector()
Language.factory("language_detector", func=create_lang_detector)
nlp.add_pipe('language_detector', last=True)

# Helpter functions
def load_data(file_name=''):
    if file_name != '':
        return pd.read_csv(f'{data_folder_path}/{file_name}.csv')
def remove_other_languages(d):
    res = d.copy()
    for i, r in res.iterrows():
        doc = nlp(r['content'])
        lang = doc._.language
        if lang['language'] != 'en' or float("{:.6f}".format(lang['score'])) < 0.999995:
            res.drop(i, inplace=True)
    return res
def capture_related_hash_tags_and_remove_from_text(r):
    n_text = r['content']
    matches = re.findall(r"#(\w+)", n_text, re.IGNORECASE)
    # Remove tags from text
    for m in matches:
        n_text = re.sub("#"+m, '', n_text)
    # Update row
    r['related_tags'] = ','.join(list(set(matches)))
    r['content'] = n_text
    return r
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I | re.A).lower().replace('\n', '').strip()
    text = re.sub(' +', ' ', text)
    text = nlp(text)
    lemmatized = list()
    for token in text:
        lemma = token.lemma_
        if lemma not in stop_words and not lemma.isnumeric():
            lemmatized.append(''.join(lemma.split()))
    return " ".join(lemmatized)
def clean_data(tag):
    # Load Data
    raw_data = load_data(tag)
    # Remove null value
    clean_data = raw_data[raw_data['content'].notnull()]
    # Keep Text with majority is English
    clean_data = remove_other_languages(clean_data)
    # Capture all related hashtags
    clean_data = clean_data.apply(capture_related_hash_tags_and_remove_from_text, axis=1)
    # Clean text
    clean_data['content'] = clean_data['content'].apply(clean_text)
    clean_data = clean_data[clean_data['content'].notnull()]
    clean_data = clean_data[clean_data['content'].notna()]
    # Filter out short content
    clean_data = clean_data[clean_data['content'].str.len() > threshold]
    clean_data['related_tags'].fillna('', inplace=True)
    # Log if category with too little content
    if len(clean_data)  < threshold_post:
        print(f"Category with number of posts less than {threshold_post}")
        return clean_data
    return clean_data

tag_list = category['Hashtag']
c = 0
for tag in tag_list:
    print(f"Cleaning Tag - {tag} ...{c+1}/{len(tag_list)}", end="\r")
    if not os.path.exists(f'{data_folder_path}/{tag}.csv'):
        print("CSV file not found...skipped")
        continue
    d = clean_data(tag)
    # Add to Clean Result
    clean_result = clean_result.append({
        'tag': tag,
        'count': len(d),
        'avg.length': d['content'].apply(lambda x: len(x)).mean(),
        'min.length': d['content'].apply(lambda x: len(x)).min(),
        'max.length': d['content'].apply(lambda x: len(x)).max()
    }, ignore_index=True)
    # Save as csv
    d.to_csv(f'{clean_folder_path}/{tag}.csv', index=False)
    print(f"Cleaning Tag - {tag} (Done)     ")
    c += 1
print(clean_result)