import numpy as np
import pandas as pd
import pyterrier as pt
import os
import ir_datasets

if not pt.started():
    pt.init()

dataset = ir_datasets.load('istella22/test')

INDEX_EXISTS = True
index_dir = '/media/ersel/Expansion/istella22_index3'

def doc_to_dict_generator(docs):
    global error_no
    for doc in docs:
        try:
            yield {"docno": doc.doc_id, "text": doc.text}
        except:
            yield {"docno": str(error_no), "text": "istella document error"}
            error_no -= 1

indexer = pt.index.IterDictIndexer(index_dir)
index = indexer.index(doc_to_dict_generator(dataset.docs))

