#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pdb import set_trace


# In[2]:


import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

import random
import string
from time import time

random.seed(1234)


def create_doc_key():
    return 'nw%s'%(''.join(random.choices(string.ascii_uppercase + string.digits, k=32)))

def preprocess(text):
    raw_sentences = sent_tokenize(text)
    sentences = [word_tokenize(s) for s in raw_sentences]
    speakers = [["spk%d"%(i) for _ in sentence] for i,sentence in enumerate(sentences)]
    return sentences, speakers

placeholder = 'AAAAAAAAA'
def create_mention_index(x):
    try:
        text, A, offset = x

        A_tok_count = len([_ for _ in word_tokenize(A)])
        if A_tok_count == 1:
            A_tok_count = 0

        _text = [c for c in text]
        _text[offset:offset+len(A)] = placeholder
        _text = ''.join(_text)
        _text = [word for sentence in sent_tokenize(_text) for word in word_tokenize(sentence)]
        start = _text.index(placeholder)
        return (start, start+A_tok_count)
    except:
        return -1
    
def create_cluster(x):
    A_coref, A_mention, B_coref, B_mention, Pronoun_mention = x
    true_mention = None
    if A_coref:
        return ((A_mention, Pronoun_mention), [(B_mention)])
    elif B_coref:
        return ((B_mention, Pronoun_mention), [(A_mention)])
    else:
        return ([(Pronoun_mention)], [(B_mention)], [(A_mention)])

# In[3]:


def reform_for_e2e_coref(filepath, training=True):
    df = pd.read_csv(filepath, sep='\t')
    
    df['A_mention'] = df[['Text', 'A', 'A-offset']].apply(create_mention_index, axis=1)
    df['B_mention'] = df[['Text', 'B', 'B-offset']].apply(create_mention_index, axis=1)
    df['Pronoun_mention'] = df[['Text', 'Pronoun', 'Pronoun-offset']].apply(create_mention_index, axis=1)
    
    df = df[(df['Pronoun_mention'] != -1) & (df['A_mention'] != -1) & (df['B_mention'] != -1)]
    
    if training:
        df['clusters'] = df[['A-coref', 'A_mention', 'B-coref', 'B_mention', 'Pronoun_mention']].apply(create_cluster, axis=1)
        df['label'] = df[['A-coref', 'B-coref']].apply(lambda x: 0 if x[0] == True else 1 if x[1] == True else 2, axis=1)
    else:
        df['clusters'] = [(())]*df.shape[0]
        df['label'] = [0]*df.shape[0]
    
    reformed = pd.DataFrame()
    reformed['ID'] = df['ID']
    reformed['A_mention'] = df['A_mention']
    reformed['B_mention'] = df['B_mention']
    reformed['Pronoun_mention'] = df['Pronoun_mention']
    reformed['label'] = df['label']
    
    reformed['tmp'] = df['Text'].apply(preprocess)
    reformed['doc_key'] = [create_doc_key() for _ in range(reformed.shape[0])]
    reformed['sentences'] = reformed['tmp'].apply(lambda x: x[0])
    reformed['speakers'] = reformed['tmp'].apply(lambda x: x[1])
    reformed['clusters'] = df['clusters']
    del reformed['tmp']
    
    reformed.to_json(filepath.replace('tsv', 'json'), orient='records', lines=True)
# In[4]:


#reform_for_e2e_coref('gap-development.tsv')
#reform_for_e2e_coref('gap-validation.tsv')
#reform_for_e2e_coref('gap-test.tsv')
reform_for_e2e_coref('test_stage_1.tsv', training=False)
