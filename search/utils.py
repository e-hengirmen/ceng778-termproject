import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import pyterrier as pt
import os
import ir_datasets
from urllib.parse import urlparse, parse_qs
import re
from sklearn.model_selection import train_test_split
import copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Input

if not pt.started():
    pt.init()

index_dir = '/media/ersel/Expansion/istella22_index3'
index = pt.IndexFactory.of(index_dir)
bm25_retriever = pt.BatchRetrieve(index, wmodel="BM25") % 100

dataset = ir_datasets.load('istella22/test')

model = load_model('gas_model.keras')



def dcg_at_k(relevances, k=None):
    if k:
        relevances = relevances[:k]
    return sum([rel / np.log2(i + 1) for i, rel in enumerate(relevances, 1)])

def ndcg_at_k(relevances, k=None):
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0

"""    bm25_res = bm25_retriever.search(q_dict['text'])
doc_dict = q_dict['docs']
state_flag = False
for doc in bm25_res.iloc:
    if doc.docno in doc_dict:
        state_flag = True
        break
if state_flag:
    q_dict['bm25'] = bm25_res
    queries2[qid] = q_dict"""


def first_ranker(query):
    results = bm25_retriever.search(query)
    results = [(
        int(doc.docid),
        doc.docno,
        doc.score
    ) for doc in results.iloc]
    return results





# features
# 0
def title_query_term_overlap(doc, query):
    title_terms = set(doc.title.split())
    query_terms = set(query.split())
    return len(title_terms & query_terms)

# 1
def title_query_jaccard_similarity(doc, query):
    title_terms = set(doc.title.split())
    query_terms = set(query.split())
    intersection = len(title_terms & query_terms)
    union = len(title_terms | query_terms)
    return intersection / union if union else 0

# 2
def title_query_dice_similarity(doc, query):
    title_terms = set(doc.title.split())
    query_terms = set(query.split())
    intersection = len(title_terms & query_terms)
    return (2 * intersection) / (len(title_terms) + len(query_terms)) if title_terms and query_terms else 0

# 3
def title_query_position(doc, query):
    title_words = doc.title.split()
    query_terms = query.split()
    for i, word in enumerate(title_words):
        if word in query_terms:
            return i
    return -1

# 4
def exact_match_title_query(doc, query):
    return 1 if doc.title.strip().lower() == query.strip().lower() else 0

# 5
def query_length(query):
    return len(query.split())

# 6
def query_character_length(query):
    return len(query)

# 7
def term_overlap(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.text.split())
    return len(query_terms & doc_terms)

# 8
def jaccard_similarity(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.text.split())
    intersection = len(query_terms & doc_terms)
    union = len(query_terms | doc_terms)
    return intersection / union if union else 0

# 9
def dice_similarity(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.text.split())
    intersection = len(query_terms & doc_terms)
    return (2 * intersection) / (len(query_terms) + len(doc_terms)) if query_terms and doc_terms else 0

# 10
def term_overlap_extra(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.extra_text.split())
    return len(query_terms & doc_terms)

# 11
def jaccard_similarity_extra(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.extra_text.split())
    intersection = len(query_terms & doc_terms)
    union = len(query_terms | doc_terms)
    return intersection / union if union else 0

# 12
def dice_similarity_extra(doc, query):
    query_terms = set(query.split())
    doc_terms = set(doc.extra_text.split())
    intersection = len(query_terms & doc_terms)
    return (2 * intersection) / (len(query_terms) + len(doc_terms)) if query_terms and doc_terms else 0

# 13
def document_length(doc):
    return len(doc.text.split())

# 14
def document_character_length(doc):
    return len(doc.text)

# 15
def average_sentence_length(doc):
    sentences = re.split(r'[.!?]', doc.text)
    sentences = [sent.strip() for sent in sentences if sent.strip()]  # Remove empty sentences and leading/trailing spaces
    return sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0

# 16
def unique_word_count(doc):
    return len(set(doc.text.split()))

# 17
def url_depth(doc):
    return doc.url.count('/')

# 18
def has_query_parameters(doc):
    return '?' in doc.url

def return_same(val):
    return val

features_list = [19, 6, 5, 4, 18, 17, 2, 16, 9, 15, 0, 12, 14, 8, 1, 3, 10, 13, 7, 11]
num_of_features = 19

def get_features(doc, query, bm25_score):
    feature_func_list = (
        (title_query_term_overlap, doc, query),  # 0
        (title_query_jaccard_similarity, doc, query),  # 1
        (title_query_dice_similarity, doc, query),  # 2
        (title_query_position, doc, query),  # 3
        (exact_match_title_query, doc, query),  # 4
        (query_length, query),  # 5
        (query_character_length, query),  # 6
        (term_overlap, doc, query),  # 7
        (jaccard_similarity, doc, query),  # 8
        (dice_similarity, doc, query),  # 9
        (term_overlap_extra, doc, query),  # 10
        (jaccard_similarity_extra, doc, query),  # 11
        (dice_similarity_extra, doc, query),  # 12
        (document_length, doc),  # 13
        (document_character_length, doc),  # 14
        (average_sentence_length, doc),  # 15
        (unique_word_count, doc),  # 16
        (url_depth, doc),  # 17
        (has_query_parameters, doc),  # 18
        (return_same, bm25_score),  # 19
    )
    feature_score_list = []
    for feature_index in features_list[:num_of_features]:
        feature_tuple = feature_func_list[feature_index]
        feature_score = feature_tuple[0](*feature_tuple[1:])
        feature_score_list.append(feature_score)
    return feature_score_list



def second_ranker(first_results, query):
    feature_group_list = []
    for docindex, docno, score in first_results:
        doc = dataset.docs[docindex]

        features = get_features(doc, query, score)
        feature_group_list.append(features)
    feature_group_list = np.array(feature_group_list)
    print(len(feature_group_list))
    print(len(features))
    
    predictions = model.predict(feature_group_list)
    scored_docs = [(pred_score[0], docindex) for pred_score, (docindex, _, _) in zip(predictions, first_results)]
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    ordered_docs_ordered = [docindex for _, docindex in sorted_docs]
    return ordered_docs_ordered

