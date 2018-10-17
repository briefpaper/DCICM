# coding=utf-8

import math
import numpy as np
from cqabaseline.evaluation import evaluate_mean_ndcg_and_precision
from dcicm.config import num_ndcg_samples, knrm_test_data_path, num_real_samples, num_neg_samples

test_data_path = knrm_test_data_path


class Bow(object):
    def __init__(self, words):
        self.data = {}
        for word in words:
            if word in self.data:
                self.data[word] += 1
            else:
                self.data[word] = 1

    def vector_length(self):
        sum = 0.0
        for value in self.data.values():
            sum += value * value
        return math.sqrt(sum)

    def sim(self, other_bow):
        sum = 0.0
        for word in self.data:
            if word in other_bow.data:
                sum += self.data[word] * other_bow.data[word]
        return sum / (self.vector_length() * other_bow.vector_length())


def compute_match_scores_list():
    test_match_scores = []

    with open(test_data_path, "r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            items = line.split()
            words_list = [item.split(",") for item in items]
            bows = [Bow(words) for words in words_list]
            match_score = bows[0].sim(bows[1])
            test_match_scores.append(match_score)
            # match_scores.append(match_score)

    match_scores_list = np.reshape(test_match_scores, [num_ndcg_samples, num_real_samples + num_neg_samples])
    return match_scores_list


def compute_correct_list():
    correct_list = []
    p_scores = []
    n_scores = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.split()
            words_list = [item.split(",") for item in items]
            bows = [Bow(words) for words in words_list]
            positive_sim = bows[0].sim(bows[1])
            negative_sim = bows[0].sim(bows[2])
            p_scores.append(positive_sim)
            n_scores.append(negative_sim)
            if positive_sim > negative_sim:
                correct_list.append(1)
            else:
                correct_list.append(0)
    return correct_list


match_scores_list = compute_match_scores_list()
evaluate_mean_ndcg_and_precision(match_scores_list)

