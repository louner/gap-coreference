#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pdb import set_trace


# In[2]:


import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import random
import string
from time import time

random.seed(1234)

import tensorflow_hub as hub
from bert import tokenization
import tensorflow as tf
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
bert_module = hub.Module(
  BERT_MODEL_HUB,
  trainable=False)

tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
with tf.Session() as sess:
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def word_tokenize(sentence):
    return tokenizer.tokenize(sentence)

def create_doc_key():
#    return 'nw%s'%(''.join(random.choices(string.ascii_uppercase + string.digits, k=32)))
    return 'nw%s'%(''.join([random.choice(string.ascii_uppercase + string.digits) for _ in range(32)]))

def preprocess(text):
    raw_sentences = sent_tokenize(text)
    sentences = [word_tokenize(s) for s in raw_sentences]
    speakers = [["spk%d"%(i) for _ in sentence] for i,sentence in enumerate(sentences)]

    # bert inputs
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in sentences]
    input_mask = [[1]*len(input_id) for input_id in input_ids]
    segment_ids = [[0]*len(input_id) for input_id in input_ids]

    return sentences, speakers, input_ids, input_mask, segment_ids

# bert tokenizer modifies tokens, so placeholder should be a token tokenizer won't modify
placeholder = 'aaa'
def create_mention_index(x):
    try:
        text, A, offset = x

        A_tok_count = len([_ for _ in word_tokenize(A)])
        _text = [c for c in text]
        _text[offset:offset+len(A)] = placeholder
        _text = ''.join(_text)
        _text = [word for sentence in sent_tokenize(_text) for word in word_tokenize(sentence)]
        start = _text.index([tok for tok in _text  if placeholder in tok][0])
        return (start, start+A_tok_count-1)
    except:
        return -1
    
def create_cluster(x):
    if len(x) == 5:
        A_coref, A_mention, B_coref, B_mention, Pronoun_mention = x
    elif len(x) == 3:
        A_mention, B_mention, Pronoun_mention = x
        A_coref, B_coref = False, False

    true_mention = None
    if A_coref:
        return ((Pronoun_mention, A_mention), [(B_mention)])
    elif B_coref:
        return ((Pronoun_mention, B_mention), [(A_mention)])
    else:
        return ([(Pronoun_mention)], [(B_mention)], [(A_mention)])

def create_cluster_index(x):
    if len(x) == 5:
        A_coref, A_mention, B_coref, B_mention, Pronoun_mention = x
    elif len(x) == 3:
        A_mention, B_mention, Pronoun_mention = x
        A_coref, B_coref = False, False

    true_mention = None
    if A_coref:
        return (0, 1, 2)
    elif B_coref:
        return (0, 2, 1)
    else:
        return (0, 2, 1)
# In[3]:


def reform_for_e2e_coref(filepath, training=True):
    df = pd.read_csv(filepath, sep='\t')
    
    df['A_mention'] = df[['Text', 'A', 'A-offset']].apply(create_mention_index, axis=1)
    df['B_mention'] = df[['Text', 'B', 'B-offset']].apply(create_mention_index, axis=1)
    df['Pronoun_mention'] = df[['Text', 'Pronoun', 'Pronoun-offset']].apply(create_mention_index, axis=1)
   
    print(df.shape)
    df = df[(df['Pronoun_mention'] != -1) & (df['A_mention'] != -1) & (df['B_mention'] != -1)]
    print(df.shape)
    if training:
        df['clusters'] = df[['A-coref', 'A_mention', 'B-coref', 'B_mention', 'Pronoun_mention']].apply(create_cluster, axis=1, raw=True)
        df['clusters_index'] = df[['A-coref', 'A_mention', 'B-coref', 'B_mention', 'Pronoun_mention']].apply(create_cluster_index, axis=1)
        df['label'] = df[['A-coref', 'B-coref']].apply(lambda x: 0 if x[0] == True else 1 if x[1] == True else 2, axis=1)
    else:
        df['clusters'] = df[['A_mention', 'B_mention', 'Pronoun_mention']].apply(create_cluster, axis=1)
        df['clusters_index'] = df[['A_mention', 'B_mention', 'Pronoun_mention']].apply(create_cluster_index, axis=1)
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

    reformed['input_ids'] = reformed['tmp'].apply(lambda x: x[2])
    reformed['input_mask'] = reformed['tmp'].apply(lambda x: x[3])
    reformed['segment_ids'] = reformed['tmp'].apply(lambda x: x[4])

    reformed['clusters'] = df['clusters']
    reformed['clusters_index'] = df['clusters_index']
    del reformed['tmp']
    
    reformed.to_json('bert-%s'%(filepath.replace('tsv', 'json')), orient='records', lines=True)
# In[4]:


reform_for_e2e_coref('gap-development.tsv')
reform_for_e2e_coref('gap-validation.tsv')
reform_for_e2e_coref('gap-test.tsv')
reform_for_e2e_coref('test_stage_1.tsv', training=False)
