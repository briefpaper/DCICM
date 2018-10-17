# coding=utf-8

import numpy as np
from dcicm.config import *
from dcicm.data.network import  MetaNetwork
import random
import pickle


def pad_documents(documents, max_len, padding_value=0):
    padded_datas = np.ones([len(documents), max_len], dtype=np.int32) * padding_value
    for i, j in np.ndindex(padded_datas.shape):
        data = documents[i]
        if j < len(data):
            padded_datas[i, j] = data[j]
    return padded_datas


train_network = MetaNetwork.load(train_network_path)
test_network = MetaNetwork.load(test_network_path)
vocab_network = MetaNetwork.load(vocab_network_path)


print("train")
print("train question count: {}".format(len(train_network.type_data_dict[TYPE_QUESTION])))
print("train answer count: {}".format(len(train_network.type_data_dict[TYPE_ANSWER])))

print("test")
print("test question count: {}".format(len(test_network.type_data_dict[TYPE_QUESTION])))
print("test answer count: {}".format(len(test_network.type_data_dict[TYPE_ANSWER])))

print("==============")
print("q_len: {}".format(len(train_network.type_data_dict[TYPE_QUESTION])))


def pad_datas_in_network(network):
    mean_question_len = np.mean([len(question) for question in network.type_data_dict[TYPE_QUESTION]])
    mean_answer_len = np.mean([len(answer) for answer in network.type_data_dict[TYPE_ANSWER]])
    print("mean question len = {}".format(mean_question_len))
    print("mean answer len = {}".format(mean_answer_len))
    network.type_data_dict[TYPE_QUESTION] = pad_documents(network.type_data_dict[TYPE_QUESTION], max_question_len)
    network.type_data_dict[TYPE_ANSWER] = pad_documents(network.type_data_dict[TYPE_ANSWER], max_answer_len)


vocab_size = len(vocab_network.type_index_id_dict[TYPE_WORD])
pad_datas_in_network(train_network)
pad_datas_in_network(test_network)


def my_shuffle(datas, shuffle):
    if shuffle:
        if random.randint(0,1) == 0:
            return datas
        for i in range(datas.shape[0]):
            random_num = random.randint(0,5)
            for k in range(random_num):
                j = random.randint(0, datas.shape[1] - 1)
                datas[i, j] = random.randint(1, vocab_size - 1)
        return datas
    else:
        return datas


def create_triple_batch_generator(network, node_types, batch_size, shuffle,  data_type_suffix=""):
    question_datas = network.type_data_dict["{}{}".format(node_types[0], data_type_suffix)]
    answer_datas = network.type_data_dict["{}{}".format(node_types[1], data_type_suffix)]
    while True:
        batch_question_indice, batch_answer_indice, batch_negative_answer_indice = \
            network.sample_triples(node_types, batch_size)
        batch_question_datas = question_datas[batch_question_indice]
        batch_answer_datas = my_shuffle(answer_datas[batch_answer_indice], shuffle)
        batch_negative_answer_datas = my_shuffle(answer_datas[batch_negative_answer_indice], shuffle)
        yield batch_question_datas, batch_answer_datas, batch_negative_answer_datas


def create_eval_batch_generator(network, node_types, batch_size, shuffle, data_type_suffix=""):
    question_datas = network.type_data_dict["{}{}".format(node_types[0], data_type_suffix)]
    answer_datas = network.type_data_dict["{}{}".format(node_types[1], data_type_suffix)]
    start = 0
    while start < ndcg_samples.shape[0]:
        batch_ndcg_samples = ndcg_samples[start: start + batch_size]

        batch_question_indice = batch_ndcg_samples[:, 0]
        batch_answer_indice = batch_ndcg_samples[:, 1]
        # batch_negative_answer_indice =batch_ndcg_samples[:, 2]

        batch_question_datas = question_datas[batch_question_indice]
        batch_answer_datas = my_shuffle(answer_datas[batch_answer_indice], shuffle)
        # batch_negative_answer_datas = my_shuffle(answer_datas[batch_negative_answer_indice], shuffle)
        start += batch_size
        yield batch_question_datas, batch_answer_datas#, batch_negative_answer_datas


word_index_id_dict = vocab_network.type_index_id_dict[TYPE_WORD]
index_word_id_dict = vocab_network.type_id_index_dict[TYPE_WORD]


def indice_to_words(indice):
    result = [word_index_id_dict[index] for index in indice]
    return result


def words_to_indice(words):
    result = [index_word_id_dict[word] for word in words]
    return result


def to_sentence(indice):
    return " ".join(indice_to_words(indice))


with open(ndcg_samples_path, "rb") as f:
    ndcg_samples = pickle.load(f)
