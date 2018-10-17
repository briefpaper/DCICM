# coding=utf-8

import pickle
from collections import namedtuple

from cqabaseline.evaluation import evaluate_mean_ndcg_and_precision
from dcicm.config import *
from gensim.models.doc2vec import Doc2Vec
from dcicm.data.network import MetaNetwork
import numpy as np

Sentence = namedtuple('NetworkSentence', 'words tags')


def train_Doc2Vec(doc_list=None, buildvoc=1, passes=20, dm=0, size=150, dm_mean=0, window=5, hs=1, negative=5,
                  min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window, hs=hs, negative=negative, min_count=min_count,
                    workers=workers)
    model.build_vocab(doc_list)
    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        model.train(doc_list, total_examples=model.corpus_count, epochs=model.iter)
    return model


train_network = MetaNetwork.load(train_network_path)
test_network = MetaNetwork.load(test_network_path)

name_network_dict = {
    # "train": train_network,
    "test": test_network
}

with open(ndcg_samples_path, "rb") as f:
    ndcg_samples = pickle.load(f)
    question_indice = set([ndcg_sample[0] for ndcg_sample in ndcg_samples])
    answer_indice = set([ndcg_sample[1] for ndcg_sample in ndcg_samples])

question_datas = test_network.type_data_dict[TYPE_QUESTION]
answer_datas = test_network.type_data_dict[TYPE_ANSWER]

docs = []
for question_index in question_indice:
    data = question_datas[question_index]
    id = "q_{}".format(question_index)
    words = [str(word_index) for word_index in data[:max_question_len]]
    tags = [id]
    doc = Sentence(words, tags)
    docs.append(doc)

for answer_index in answer_indice:
    data = answer_datas[answer_index]
    id = "a_{}".format(answer_index)
    words = [str(word_index) for word_index in data[:max_answer_len]]
    tags = [id]
    doc = Sentence(words, tags)
    docs.append(doc)

print("complete reading data")
model = train_Doc2Vec(docs, passes=2)


def cos(a, b):
    sum = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    return sum / (norm_a * norm_b)


def sim(q_index, a_index):
    q_id = "q_{}".format(q_index)
    a_id = "a_{}".format(a_index)
    q = model.docvecs[q_id]
    a = model.docvecs[a_id]
    return cos(q, a)


test_match_scores = []
for ndcg_sample in ndcg_samples:
    match_score = sim(ndcg_sample[0], ndcg_sample[1])
    test_match_scores.append(match_score)

match_scores_list = np.reshape(test_match_scores, [num_ndcg_samples, num_real_samples + num_neg_samples])
print(match_scores_list.shape)
print(match_scores_list)

evaluate_mean_ndcg_and_precision(match_scores_list)
