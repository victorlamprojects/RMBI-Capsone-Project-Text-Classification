# Global Config
from common.GlobalConfig import *

## 1. Import
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
# HuggingFace
from transformers import BertTokenizer
# Keras
from keras.preprocessing.sequence import pad_sequences
# Spacy
import spacy
nlp = spacy.load('en_core_web_lg')
import os
if not os.path.exists(preprocessed_folder_path):
    os.makedirs(preprocessed_folder_path)
    
## 2. Global variable
category = pd.read_csv(hashtag_list_path)
preprocess_result = pd.DataFrame(columns=['tag', 'count', 'train_size', 'test_size'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Test ratio
split_ratio=0.2

## 3. Helper functions
def load_data(file_name=''):
    if file_name != '':
        return pd.read_csv(f'{clean_folder_path}/{file_name}.csv')
def encode_output(output, categories, groups):
    # tag mappings
    index_to_tag = pd.DataFrame({'tag': categories})
    tag_to_index = index_to_tag.set_index('tag')
    tag_to_index['index'] = index_to_tag.index
    # tag to cat mappings
    tag_to_cat = category[['Hashtag','Group']]
    tag_to_cat.rename(columns={'Hashtag': 'tag', 'Group': 'cat'}, inplace=True)
    tag_to_cat = tag_to_cat.set_index('tag')
    # cat mappings
    index_to_cat = pd.DataFrame({'cat': groups})
    cat_to_index = index_to_cat.set_index('cat')
    cat_to_index['index'] = index_to_cat.index
    
    # Save all mappings
    index_to_tag.to_csv(f'{preprocessed_folder_path}/index_to_tag.csv', index=False)
    index_to_cat.to_csv(f'{preprocessed_folder_path}/index_to_cat.csv', index=False)
    tag_to_index.to_csv(f'{preprocessed_folder_path}/tag_to_index.csv')
    cat_to_index.to_csv(f'{preprocessed_folder_path}/cat_to_index.csv')
    tag_to_cat.to_csv(f'{preprocessed_folder_path}/tag_to_cat.csv')
    
    output_size = len(categories)
    onehot = []
    for o in output:
        v = np.zeros(output_size)
        v[tag_to_index.loc[o, 'index']] = 1
        onehot.append(v)
    labels = []
    for o in output:
        labels.append([tag_to_index.loc[o, 'index']])
    return np.asarray(onehot), np.asarray(labels)
def get_encode_vec(text):
    encoded = tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=max_length)
    return np.asarray(encoded['input_ids'])

def pad_seq(seq):
    return np.asarray(pad_sequences(seq.transpose(), dtype='float32', maxlen=max_length).transpose())
def get_word_vec(text):
    seq = np.array([nlp.vocab.get_vector(word) for word in text.split() if nlp.vocab.has_vector(word)])
    if seq.size > 0:
        seq = pad_seq(seq)
    else:
        seq = np.zeros((max_length, vocab_size))
    return np.asarray(seq)

def preprocess(tag):
    d = load_data(tag)
    # Skip out category with too little content
    if len(d)  < threshold_post:
        return (pd.DataFrame(), pd.DataFrame())
    d = d.sample(n=threshold_post, random_state=random_seed)
    return train_test_split(d, test_size=split_ratio)

## 4. Preprocess Tag
tag_list = category['Hashtag']
total_train = pd.DataFrame()
total_test = pd.DataFrame()
unqualified_cat = []
c = 0
for tag in tag_list:
    c += 1
    print(f"Preprocessing Tag - {tag} ...{c}/{len(tag_list)}", end="\r")
    train, test = preprocess(tag)
    if train.empty or test.empty:
        print("")
        unqualified_cat.append(tag)
        preprocess_result = preprocess_result.append({
            'tag': tag,
            'count': 'N/A',
            'train_size': 'N/A',
            'test_size': 'N/A'
        }, ignore_index=True)
        continue
    preprocess_result = preprocess_result.append({
        'tag': tag,
        'count': len(train)+len(test),
        'train_size': len(train),
        'test_size': len(test)
    }, ignore_index=True)
    total_train = total_train.append(train)
    total_test = total_test.append(test)
    print(f"Preprocessing Tag - {tag} (Done)               ")
if len(unqualified_cat) > 0:
    print(f"Unqualified categories:")
    for c in unqualified_cat:
        print(c)

## 5. Encoding
total_tags = total_train['tag'].unique()
total_cats = category['Group'].unique()
shuffle_train = total_train.sample(frac=1).reset_index(drop=True)
shuffle_test = total_test.sample(frac=1).reset_index(drop=True)

train_X = shuffle_train['content'].astype('str').apply(get_word_vec).values
train_X = np.array([i.astype(np.float32) for i in train_X])
train_BERT_X = shuffle_train['content'].astype('str').apply(get_encode_vec).values
train_Y, train_BERT_Y = encode_output(shuffle_train['tag'], total_tags, total_cats)

test_X = shuffle_test['content'].astype('str').apply(get_word_vec).values
test_X = np.array([i.astype(np.float32) for i in test_X])
test_BERT_X = shuffle_test['content'].astype('str').apply(get_encode_vec).values
test_Y, test_BERT_Y = encode_output(shuffle_test['tag'], total_tags, total_cats)

## 6. Save encoded data
np.save(f'{preprocessed_folder_path}/train_X.npy', train_X)
np.save(f'{preprocessed_folder_path}/test_X.npy', test_X)
np.save(f'{preprocessed_folder_path}/train_Y.npy', train_Y)
np.save(f'{preprocessed_folder_path}/test_Y.npy', test_Y)
np.save(f'{preprocessed_folder_path}/train_BERT_X.npy', train_BERT_X)
np.save(f'{preprocessed_folder_path}/train_BERT_Y.npy', train_BERT_Y)
np.save(f'{preprocessed_folder_path}/test_BERT_X.npy', test_BERT_X)
np.save(f'{preprocessed_folder_path}/test_BERT_Y.npy', test_BERT_Y)

## 7. Print result
print(preprocess_result)