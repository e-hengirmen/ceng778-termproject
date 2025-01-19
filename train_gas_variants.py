import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

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
import time
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model

import lightgbm as lgb
from sklearn.datasets import load_svmlight_file


def dcg_at_k(relevances, k=None):
    if k:
        relevances = relevances[:k]
    return sum([rel / np.log2(i + 1) for i, rel in enumerate(relevances, 1)])

def ndcg_at_k(relevances, k=None):
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0

def normal_access(container, i1, i2):
    return container[i1][i2]

def csr_matrix_access(container, i1, i2):
    return container[(i1, i2)]

def kendalls_tau(f1_index, f2_index, features_list, access=normal_access):
    concordont_count = 0
    discordont_count = 0
    try:
        n = len(features_list)
    except:
        n = features_list.shape[0]
    for d1 in range(n):
        d1_f1 = access(features_list, d1, f1_index)
        d1_f2 = access(features_list, d1, f2_index)
        for d2 in range(d1+1, n):
            d2_f1 = access(features_list, d2, f1_index)
            d2_f2 = access(features_list, d2, f2_index)
            if (
                    (
                        (d1_f1 > d2_f1)
                        and
                        (d1_f2 > d2_f2)
                    ) or
                    (
                        (d1_f1 < d2_f1)
                        and
                        (d1_f2 < d2_f2)
                    )
            ):
                concordont_count += 1
            elif (
                    (
                        (d1_f1 > d2_f1)
                        and
                        (d1_f2 < d2_f2)
                    ) or
                    (
                        (d1_f1 < d2_f1)
                        and
                        (d1_f2 > d2_f2)
                    )
            ):
                discordont_count += 1
    return (concordont_count - discordont_count) / (n*(n-1)/2)

def get_sim_key(f1, f2):
    return str(min(f1, f2)) + "#" + str(max(f1, f2))

def get_best_feature(features, placeholder):
    best_feature = None
    best_feature_index = None
    for index, feature in features:
        if not best_feature or feature >= best_feature:
            best_feature = feature
            best_feature_index = index 
    return best_feature_index, best_feature

def get_best_max_k(features, similarity_dict, k=5):
    best_feature = None
    best_feature_index = None
    for index, feature in features:
        if not best_feature:
            best_feature = feature
            best_feature_index = index
            continue
        
        other_features = []
        for index2, feature2 in features:
            if index == index2:
                continue
            key = get_sim_key(index, index2)
            sim_val = similarity_dict[key]
            other_features.append(feature2 - sim_val)
        feature_score = sum(sorted(other_features, reverse=True)[:5])
        
        if feature_score >= best_feature:
            best_feature = feature_score
            best_feature_index = index
    return best_feature_index, best_feature

def get_best_average_change(features, similarity_dict):
    best_feature = None
    best_feature_index = None
    for index, feature in features:
        if not best_feature:
            best_feature = feature
            best_feature_index = index
            continue
        
        average_decrement = 0
        for index2, _ in features:
            if index == index2:
                continue
            key = get_sim_key(index, index2)
            sim_val = similarity_dict[key]
            average_decrement += sim_val
        average_decrement /= len(features) - 1
        
        feature_score = feature - average_decrement
        if feature_score >= best_feature:
            best_feature = feature_score
            best_feature_index = index
    return best_feature_index, best_feature


def update_features(features, best_feature_index, similarity_dict):
    for feature_tuple in features:
        feature_index, feature_score = feature_tuple
        if feature_index != best_feature_index:
            key = get_sim_key(feature_index, best_feature_index)
            sim_val = similarity_dict[key]
            feature_tuple[1] -= sim_val

def GAS(features, similarity_dict, get_best_func=get_best_feature):
    features2 = [[index, score] for index, score in enumerate(features)]
    n = len(features2)
    features_ordered = []
    for i in range(n-1):
        best_feature_index, best_feature_score = get_best_func(features2, similarity_dict)
        print(best_feature_index, best_feature_score)
        features_ordered.append(best_feature_index)
        update_features(features2, best_feature_index, similarity_dict)
        features2 = [feature_tuple for feature_tuple in features2 if feature_tuple[0] != best_feature_index]
    features_ordered.append(features2[0][0])
    return features_ordered

#from rankeval.dataset import Dataset
#from rankeval.model import RTEnsemble
TEST_FILE = '/media/ersel/Expansion/code/CENG778-istella22_trials/lambdamart/data/test.monoT5.svm'
MODEL_FILE = '/media/ersel/Expansion/code/CENG778-istella22_trials/lambdamart/models/lambdamart.monoT5.lgb'

X, y, q = load_svmlight_file(TEST_FILE, query_id=True)

divide_point = 1074050

if divide_point is None:
    old_qid = q[0]
    qid_count = 1
    for index, qid in enumerate(q):
        if qid != old_qid:
            qid_count += 1
            old_qid = qid
            if qid_count == 1501:
                divide_point = index
                break

X_train = X[:divide_point]
X_test = X[divide_point:]
y_train = y[:divide_point]
y_test = y[divide_point:]
q_train = q[:divide_point]
q_test = q[divide_point:]

feature_count = X_train.shape[1]
train_query_count = 100 # not 1500 because it takes 10 hours just to 

qid_limits_train = []
old_qid = q_train[0]
start_index = 0
for index, qid in enumerate(q_train):
    if qid != old_qid:
        old_qid = qid
        qid_limits_train.append((start_index, index))
        start_index = index
qid_limits_train.append((start_index, index + 1))
qid_rel_count_train = [limits[1] - limits[0] for limits in qid_limits_train]

qid_limits_test = []
old_qid = q_test[0]
start_index = 0
for index, qid in enumerate(q_test):
    if qid != old_qid:
        old_qid = qid
        qid_limits_test.append((start_index, index))
        start_index = index
qid_limits_test.append((start_index, index + 1))
qid_rel_count_test = [limits[1] - limits[0] for limits in qid_limits_test]

features_already_calculated = True
if not features_already_calculated:
    similarity_dict = {
        get_sim_key(f1_index, f2_index): 0
        for f1_index in range(feature_count)
        for f2_index in range(f1_index + 1, feature_count)
    }



    start_time = 0
    for i in range(train_query_count):
        start, end = qid_limits_train[i]
        print(i, 'start,end:', start, end, 'time:', time.time() - start_time)
        start_time = time.time()
        for f1_index in range(feature_count):
            for f2_index in range(f1_index + 1, feature_count):
                key = get_sim_key(f1_index, f2_index)
                f1_container = X_train[start:end, f1_index].toarray().flatten().tolist()
                f2_container = X_train[start:end, f2_index].toarray().flatten().tolist()
                score = cosine_similarity([f1_container], [f2_container])[0][0]
                similarity_dict[key] += score

    for key in similarity_dict:
        similarity_dict[key] /= train_query_count

    feature_scores = [0 for i in range(feature_count)]
    for i in range(train_query_count):
        start, end = qid_limits_train[i]
        print(i, 'start,end:', start, end, 'time:', time.time() - start_time)
        start_time = time.time()
        scores = y_train[start:end].flatten().tolist()
        for f_index in range(feature_count):
            f_container = X_train[start:end, f_index].toarray().flatten().tolist()
            c_feature_list = list(zip(f_container, scores))
            feature_sorted_list = [second for _, second in sorted(c_feature_list, key=lambda x: x[0])]
            ndcg_score = ndcg_at_k(feature_sorted_list)
            feature_sorted_list.reverse()
            ndcg_score2 = ndcg_at_k(feature_sorted_list)
            feature_scores[f_index] += max(ndcg_score, ndcg_score2)

    for i in range(feature_count):
        feature_scores[i] /= train_query_count

    features_ordered1 = GAS(feature_scores, similarity_dict, get_best_func=get_best_feature)
    features_ordered2 = GAS(feature_scores, similarity_dict, get_best_func=get_best_max_k)
    features_ordered3 = GAS(feature_scores, similarity_dict, get_best_func=get_best_average_change)
else:
    features_ordered1 = [186, 220, 79, 0, 193, 218, 212, 205, 204, 202, 197, 94, 93, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 76, 62, 48, 174, 20, 104, 118, 200, 208, 107, 34, 217, 209, 103, 191, 65, 173, 51, 3, 74, 116, 219, 132, 99, 207, 214, 121, 160, 102, 172, 71, 32, 146, 203, 98, 166, 57, 117, 75, 101, 131, 37, 198, 70, 100, 46, 194, 18, 105, 211, 68, 96, 168, 56, 72, 112, 95, 19, 33, 108, 80, 47, 130, 97, 77, 177, 2, 60, 73, 113, 159, 67, 43, 1, 69, 110, 158, 61, 176, 54, 145, 42, 124, 58, 114, 144, 22, 45, 59, 126, 115, 63, 196, 40, 66, 12, 119, 55, 152, 122, 53, 44, 183, 109, 52, 135, 111, 14, 41, 29, 154, 39, 6, 134, 49, 171, 31, 179, 36, 28, 181, 190, 9, 138, 26, 184, 30, 213, 201, 25, 162, 140, 178, 38, 167, 27, 210, 35, 192, 120, 5, 23, 148, 50, 163, 189, 187, 149, 216, 170, 129, 165, 125, 136, 199, 4, 155, 123, 64, 128, 11, 151, 15, 17, 16, 127, 141, 157, 185, 156, 215, 169, 188, 182, 180, 206, 13, 24, 78, 106, 137, 21, 153, 164, 8, 175, 161, 133, 92, 10, 7, 147, 195, 142, 143, 150, 139]
    features_ordered2 = [220, 79, 193, 186, 218, 212, 205, 204, 202, 197, 94, 93, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 76, 62, 48, 20, 174, 104, 200, 118, 208, 107, 34, 0, 209, 217, 191, 65, 173, 103, 51, 3, 116, 219, 75, 132, 1, 207, 214, 160, 121, 102, 2, 71, 146, 203, 99, 57, 172, 32, 74, 117, 98, 37, 166, 70, 101, 4, 194, 5, 100, 6, 68, 131, 105, 96, 46, 95, 18, 108, 72, 7, 8, 9, 97, 73, 19, 77, 10, 11, 12, 80, 67, 13, 14, 15, 69, 130, 16, 198, 56, 17, 159, 112, 33, 21, 22, 23, 60, 145, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 158, 50, 52, 53, 54, 55, 58, 59, 61, 63, 144, 64, 66, 78, 92, 106, 109, 110, 111, 113, 114, 115, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 199, 201, 206, 210, 211, 213, 215, 216]
    features_ordered3 = [220, 79, 193, 0, 218, 212, 205, 204, 202, 197, 94, 93, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 76, 62, 48, 174, 20, 104, 200, 208, 118, 107, 209, 34, 217, 1, 103, 191, 65, 3, 51, 74, 116, 219, 132, 207, 102, 173, 214, 121, 160, 99, 71, 32, 203, 146, 172, 98, 57, 2, 75, 117, 101, 166, 37, 194, 100, 70, 46, 131, 105, 211, 68, 198, 96, 18, 56, 72, 95, 168, 112, 33, 77, 108, 130, 97, 47, 73, 177, 19, 67, 60, 113, 80, 159, 43, 69, 110, 158, 61, 176, 145, 54, 42, 124, 58, 144, 114, 22, 45, 59, 126, 115, 196, 63, 40, 66, 12, 119, 55, 152, 122, 53, 44, 183, 109, 135, 52, 111, 14, 41, 29, 154, 39, 6, 134, 49, 171, 31, 179, 36, 28, 181, 190, 9, 138, 26, 184, 213, 30, 201, 162, 25, 140, 178, 38, 167, 27, 210, 35, 192, 120, 5, 23, 148, 50, 163, 189, 187, 149, 216, 170, 129, 165, 125, 136, 4, 199, 155, 123, 64, 128, 11, 151, 15, 17, 16, 7, 8, 141, 10, 13, 21, 24, 78, 127, 92, 106, 133, 137, 215, 157, 188, 185, 156, 186, 169, 182, 180, 206, 153, 164, 175, 161, 147, 195, 139, 142, 143, 150]

def get_trained_lambdamart_model(x, y):
    params = {
        'objective': 'lambdarank',   # Set objective for LambdaMART
        'metric': 'ndcg',            # Evaluation metric
        'boosting_type': 'gbdt',     # Gradient boosting
        'num_leaves': 31,            # Max number of leaves per tree
        'learning_rate': 0.05,       # Learning rate
        'verbose': 0,                # Verbosity
        'max_depth': -1              # No max depth limit
    }
    train_data = lgb.Dataset(x.toarray(), label=y, group=qid_rel_count_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

for features_ordered, model_name_suffix in [
    (features_ordered1, "standard"),
    (features_ordered2, "max5"),
    (features_ordered3, "best_average"),
]:
    best_score = -9999999999999999
    best_model = None
    best_num = None
    batch_size = 64
    
    iteration_time = time.time()
    # for num_of_features in range(2,feature_count):
    for num_of_features in range(feature_count - 1 , 1, -1):
        feature_indexes = sorted(copy.deepcopy(features_ordered[:num_of_features]))
        training_X = X_train[:, features_ordered[:num_of_features]]
        testing_X = X_test[:, features_ordered[:num_of_features]]
        
        # training_X = np.array(training_X)
        # testing_X = np.array(testing_X)
    
        # model = get_model(num_of_features)
    
    
        """
        generated_dataset = tf.data.Dataset.from_generator(
            lambda: pair_data_generator(training_X, y_train, batch_size=64),
            # pair_data_generator(training_X, y_train, batch_size=64),
            output_signature=(
                tf.TensorSpec(shape=(None, num_of_features), dtype=tf.float32),  # Features shape
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)   # Labels shape
            )
        )
        generated_dataset = generated_dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
        """
    
        # model.fit(generated_dataset, epochs=100, steps_per_epoch=training_X.shape[0] // 64)
        # model.fit(generated_dataset, epochs=100, verbose=1)
        # generator = DynamicFeatureDataGenerator(training_X, y_train, batch_size)
        # model.fit(generator, epochs=10, verbose=1)
        # model.fit(training_X, y_train, epochs=100, batch_size=64, verbose=1)
        model = get_trained_lambdamart_model(training_X, y_train)
        print(f"{model_name_suffix}-{num_of_features} ### iteration_took:", time.time() - iteration_time)
        iteration_time = time.time()
            
        # test_generator = DynamicFeatureDataTESTGenerator(testing_X, batch_size)
        # y_pred = model.predict(test_generator, verbose=0)
        # y_pred = model.predict(testing_X, verbose=0)
    
        start_time = time.time()
        y_pred = model.predict(testing_X.toarray())
        time_took = time.time() - start_time
    
        ndcg_average = 0
        for start, end in qid_limits_test:
            pred_ys = y_pred[start:end]
            real_ys = y_test[start:end]
            scored_docs = list(zip(pred_ys, real_ys))
            sorted_relevances = [data[1] for data in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
            ndcg_score = ndcg_at_k(sorted_relevances)
            ndcg_average += ndcg_score
        ndcg_average /= len(qid_limits_test)
        score = ndcg_average
        print("feature_count:", num_of_features,"ndcg_score:", ndcg_average, "prediction_time:", time_took)
    
        if score > best_score:
            best_score = score
            best_model = model
            best_num = num_of_features

    print()
    print()
    print()
    best_model.save_model(f'./gas_lambdamart_{model_name_suffix}_{best_num}_model.txt')









