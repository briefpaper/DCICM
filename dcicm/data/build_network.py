#coding=utf-8

from dcicm.config import *
from pymongo import MongoClient
from dcicm.data.network import MetaNetwork
import nltk
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import numpy as np

eng_regex = re.compile("[a-zA-Z]{2,}$")
punc_regex = re.compile("[\\.;,\\?!]")


def tokenize(line):
    tokens = [word.lower() for word in nltk.word_tokenize(line) if eng_regex.match(word) or punc_regex.match(word)]
    return tokens


def tokens_to_sent_tokens_list(tokens):
    sent_tokens_list = []
    sent_tokens = []
    for token in tokens:
        if punc_regex.match(token):
            if len(sent_tokens) > 0:
                sent_tokens_list.append(sent_tokens)
                sent_tokens = []
        else:
            sent_tokens.append(token)
    if len(sent_tokens) > 0:
        sent_tokens_list.append(sent_tokens)
    return sent_tokens_list


class DataGenerator(object):
    def create_qa_tokens_generator(self):
        return None

    def __iter__(self):
        self.qa_tokens_generator = self.create_qa_tokens_generator()
        return self

    def __next__(self):
        while True:
            try:
                question_tokens, answer_tokens_list = next(self.qa_tokens_generator)
            except StopIteration as e:
                self.close()
                raise e
            question_tokens = self.filter_question_tokens(question_tokens)
            if question_tokens is None:
                continue
            answer_tokens_list = [self.filter_answer_tokens(answer_tokens) for answer_tokens in answer_tokens_list]
            answer_tokens_list = [answer_tokens for answer_tokens in answer_tokens_list if answer_tokens is not None]
            if len(answer_tokens_list) == 0:
                continue
            return question_tokens, answer_tokens_list

    def filter_answer_tokens(self, answer_tokens):

        if not filter_by_length:
            return answer_tokens

        # if filter_by_length:
        if len(answer_tokens) < 3:# or len(answer_tokens) > 100:
            return None
        else:
            return answer_tokens

    def filter_question_tokens(self, question_tokens):

        if not filter_by_length:
            return question_tokens

        if len(question_tokens) < 3:
            return None
        else:
            return question_tokens

    def close(self):
        pass


class MongoDataGenerator(DataGenerator):
    def __init__(self, limit):
        super().__init__()
        self.client = MongoClient(MONGO_HOST, MONGO_PORT)
        self.qa_col = self.client[MONGO_DB]["qa"]
        self.limit = limit

    def create_qa_tokens_generator(self):
        for index, doc in enumerate(self.qa_col.find().limit(self.limit)):
            # print("index = {}".format(index))
            question = doc["question"]

            if dataset == "zhihu":
                question_tokens = question["tokens"]
            else:
                question_text = question["text"]
                question_tokens = tokenize(question_text)

            if self.filter_question_tokens(question_tokens) is None:
                continue

            answer_tokens_list = []
            answers = doc["answers"]
            for answer in answers:
                if dataset == "zhihu":
                    answer_tokens = answer["tokens"]
                else:
                    answer_text = BeautifulSoup(answer["html"], "lxml").text
                    answer_tokens = tokenize(answer_text)

                if self.filter_answer_tokens(answer_tokens) is None:
                    continue

                answer_tokens_list.append(answer_tokens)

            yield question_tokens, answer_tokens_list

    def close(self):
        self.client.close()


class SemEvalDataGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def create_qa_tokens_generator(self):
        threads = []
        for i in range(1, 3):
            fpath = os.path.join(__file__, "../../CQA_datasets/SemEval/train/SemEval2016-Task3-CQA-QL-train-part{}.xml".format(i))
            with open(fpath, "r", encoding="utf-8") as f:
                xml = f.read().lower()
            soup = BeautifulSoup(xml, "lxml")
            threads.extend(soup.select("thread"))#[:100])
        for index, thread in enumerate(threads):
            # print("index = {}".format(index))
            rel_question = thread.select("relquestion")[0]
            question_text = rel_question.text
            question_tokens = tokenize(question_text)
            answer_tokens_list = []
            for rel_comment in thread.select("relcomment[relc_relevance2orgq=good]"):
                answer_text = rel_comment.text
                answer_tokens = tokenize(answer_text)
                answer_tokens_list.append(answer_tokens)
            yield question_tokens, answer_tokens_list


class IDGenerator(object):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.last_question_id = -1

    def generate_id(self):
        self.last_question_id += 1
        return "{}_{}".format(self.prefix, self.last_question_id)


def build_networks_bak(train_network_path, test_network_path, vocab_network_path, limit):
    train_network = MetaNetwork()
    test_network = MetaNetwork()
    vocab_network = MetaNetwork()

    networks = [
        train_network,
        test_network
    ]

    network_question_tokens_list_dict = {
        train_network: [],
        test_network: []
    }
    network_answer_tokens_list_dict = {
        train_network: [],
        test_network: []
    }

    question_id_generator = IDGenerator("q")
    answer_id_generator = IDGenerator("a")

    if dataset == "sem_eval":
        data_generator = SemEvalDataGenerator()
    else:
        data_generator = MongoDataGenerator(limit)

    for i, (question_tokens, answer_tokens_list) in tqdm(enumerate(data_generator)):
        if i % 10 < 3:
            current_network = test_network
        else:
            current_network = train_network

        question_id = question_id_generator.generate_id()
        question_index = current_network.get_or_create_index(TYPE_QUESTION, question_id)
        network_question_tokens_list_dict[current_network].append(question_tokens)

        for answer_tokens in answer_tokens_list:
            answer_id = answer_id_generator.generate_id()
            answer_index = current_network.get_or_create_index(TYPE_ANSWER, answer_id)
            network_answer_tokens_list_dict[current_network].append(answer_tokens)

            current_network.add_edges([TYPE_QUESTION, TYPE_ANSWER], question_index, answer_index)

    vocab_network.get_or_create_index(TYPE_WORD, "<PAD>")
    vocab_network.get_or_create_index(TYPE_WORD, "<START>")

    def tokens_to_indice(tokens, is_train):
        if is_train:
            indices = [vocab_network.get_or_create_index(TYPE_WORD, word_id) for word_id in tokens]
        else:
            indices = [vocab_network.get_index(TYPE_WORD, word_id) for word_id in tokens if vocab_network.has_id(TYPE_WORD, word_id)]
        return indices

    for network in networks:
        is_train = (network is train_network)
        network.type_data_dict[TYPE_QUESTION] = [tokens_to_indice(tokens, is_train) for tokens in network_question_tokens_list_dict[network]]
        network.type_data_dict[TYPE_ANSWER] = [tokens_to_indice(tokens, is_train) for tokens in network_answer_tokens_list_dict[network]]
        network.build_sample_list()
        print("build sample list")

    train_network.save(train_network_path)
    test_network.save(test_network_path)
    vocab_network.save(vocab_network_path)


class QADocument(object):
    def __init__(self, id, token_indices):
        self.id = id
        self.token_indices = token_indices


class QA(object):
    def __init__(self, question, answers):
        self.question = question
        self.answers = answers


def build_qa_pickle(qa_pickle_path, vocab_network_path, limit):
    qa_list = []

    vocab_network = MetaNetwork()
    vocab_network.get_or_create_index(TYPE_WORD, "<PAD>")
    vocab_network.get_or_create_index(TYPE_WORD, "<START>")

    def tokens_to_indices(tokens):
        indices = [vocab_network.get_or_create_index(TYPE_WORD, word_id) for word_id in tokens]
        return indices

    question_id_generator = IDGenerator("q")
    answer_id_generator = IDGenerator("a")

    if dataset == "sem_eval":
        data_generator = SemEvalDataGenerator()
    else:
        data_generator = MongoDataGenerator(limit)

    for i, (question_tokens, answer_tokens_list) in tqdm(enumerate(data_generator)):

        question_id = question_id_generator.generate_id()
        question_token_indices = tokens_to_indices(question_tokens)
        question = QADocument(question_id, question_token_indices)

        answers = []
        for answer_tokens in answer_tokens_list:
            answer_id = answer_id_generator.generate_id()
            answer_token_indices = tokens_to_indices(answer_tokens)
            answer = QADocument(answer_id, answer_token_indices)
            answers.append(answer)

        qa =QA(question, answers)
        qa_list.append(qa)

    # save qa data
    with open(qa_pickle_path, "wb") as f:
        pickle.dump(qa_list, f)
    print("save qa_list")

    # save vocab data
    vocab_network.save(vocab_network_path)
    print("save vocab")


def build_network(qa_list, network_path):
    print("building network")
    network = MetaNetwork()
    network.type_data_dict[TYPE_QUESTION] = []
    network.type_data_dict[TYPE_ANSWER] = []

    for qa in tqdm(qa_list):
        question_index = network.get_or_create_index(TYPE_QUESTION, qa.question.id)
        network.type_data_dict[TYPE_QUESTION].append(qa.question.token_indices)
        for answer in qa.answers:
            answer_index = network.get_or_create_index(TYPE_ANSWER, answer.id)
            network.type_data_dict[TYPE_ANSWER].append(answer.token_indices)
            network.add_edges([TYPE_QUESTION, TYPE_ANSWER], question_index, answer_index)
    network.type_data_dict[TYPE_QUESTION] = np.array(network.type_data_dict[TYPE_QUESTION])
    network.type_data_dict[TYPE_ANSWER] = np.array(network.type_data_dict[TYPE_ANSWER])
    network.build_sample_list()
    network.save(network_path)
    return network


def int_list_to_str(l):
    return ",".join([str(v) for v in l])


def build_test_samples(test_network):
    test_samples = test_network.sample_ndcgs([TYPE_QUESTION, TYPE_ANSWER], num_ndcg_samples, num_real_samples, num_neg_samples)
    with open(ndcg_samples_path, "wb") as f:
        pickle.dump(test_samples, f)
    return test_samples


def build_train_and_test(qa_pickle_path, train_network_path, test_network_path, training_rate):
    with open(qa_pickle_path, "rb") as f:
        qa_list = pickle.load(f)
    training_size = int(len(qa_list) * training_rate)
    random_indices = np.random.permutation(len(qa_list))
    train_indices = random_indices[:training_size]
    test_indices = random_indices[training_size:]

    train_qa_list = [qa_list[index] for index in train_indices]
    test_qa_list = [qa_list[index] for index in test_indices]

    print("building training network")
    train_network = build_network(train_qa_list, train_network_path)
    print("building testing network")
    test_network = build_network(test_qa_list, test_network_path)
    print("building test samples")
    test_samples = build_test_samples(test_network)
    return train_network, test_network, test_samples


def build_knrm_test_data(knrm_test_data_path, test_samples, test_network):
    with open(knrm_test_data_path, "w", encoding="utf-8") as f:
        for ndcg_sample in test_samples:
            q = test_network.type_data_dict[TYPE_QUESTION][ndcg_sample[0]]
            a = test_network.type_data_dict[TYPE_ANSWER][ndcg_sample[1]]
            line = "{}\t{}\t{}\t1\n".format(int_list_to_str(q), int_list_to_str(a), int_list_to_str(a))
            f.write(line)


def build_knrm_train_data(path, network, batch_count, batch_size):
    node_types = [TYPE_QUESTION, TYPE_ANSWER]
    batch_generator = network.create_triple_batch_generator(node_types, batch_size)
    with open(path, "w", encoding="utf-8") as f:
        for batch_index in range(batch_count):
            for a, b, c in zip(*next(batch_generator)):
                line = "{}\t{}\t{}".format(int_list_to_str(a), int_list_to_str(b), int_list_to_str(c))
                line = "{}\t{}\n".format(line, 1)
                f.write(line)


def build_knrm_data(train_network, test_network, test_samples):
    if not os.path.exists(knrm_dir):
        os.mkdir(knrm_dir)
    build_knrm_train_data(knrm_train_data_path, train_network, 5000, 200)
    build_knrm_test_data(knrm_test_data_path, test_samples, test_network)
    print("build data for knrm")


if __name__ == "__main__":
    build_qa_pickle(qa_pickle_path, vocab_network_path, 40000)
    train_network, test_network, test_samples = build_train_and_test(qa_pickle_path, train_network_path, test_network_path, 0.3)
    build_knrm_data(train_network, test_network, test_samples)