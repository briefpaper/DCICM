# coding=utf-8

import numpy as np
from dcicm.config import num_real_samples, num_neg_samples
from cqabaseline.measures import find_ndcg


def evaluate(correct_list):
    correct_matrix = np.array(correct_list).reshape(-1, num_neg_samples)
    precision = np.mean((np.sum(correct_matrix, axis=1) == num_neg_samples))
    error_counts = (num_neg_samples - correct_matrix.sum(1)).astype(np.int32)
    ndcg_scores = []
    for i, error_count in enumerate(error_counts):
        hypo = [0] * (1 + num_neg_samples)
        hypo[error_count] = 1
        ndcg_score = find_ndcg([1] + [0] * num_neg_samples, hypo)
        ndcg_scores.append(ndcg_score)
    ave_ndcg_score = np.mean(ndcg_scores)
    print("ave_ndcg = {}\tprecision = {}".format(ave_ndcg_score, precision))
    return ave_ndcg_score, precision


def evaluate_ndcg_and_precision(match_scores):
    ndcg_gold = np.array([1] * num_real_samples + [0] * num_neg_samples)
    sort_index = np.argsort(match_scores)[::-1]
    hypo = ndcg_gold[sort_index]
    ndcg_score = find_ndcg(ndcg_gold, hypo)
    precision = np.mean(hypo[:num_real_samples])
    return ndcg_score, precision


def evaluate_mean_ndcg_and_precision(match_scores_list):
    score_tuples = [evaluate_ndcg_and_precision(match_scores) for match_scores in match_scores_list]
    score_tuples = np.array(score_tuples)
    ave_ndcg_score, precision = np.mean(score_tuples, axis=0)
    num_samples = num_real_samples + num_neg_samples
    print("ave_ndcg_{}@{} = {}\tprecision = {}".format(num_real_samples, num_samples, ave_ndcg_score, precision))
    return ave_ndcg_score, precision
