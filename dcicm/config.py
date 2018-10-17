# coding=utf-8
import os

dataset = "quora"  # quora/zhihu/sem_eval
data_dir = "/YOUR_DATA_DIR/{}".format(dataset)
knrm_dir = os.path.join(data_dir, "knrm")

qa_pickle_path = os.path.join(data_dir, "qa_list.p")
train_network_path = os.path.join(data_dir, "train_network.p")
test_network_path = os.path.join(data_dir, "test_network.p")
vocab_network_path = os.path.join(data_dir, "vocab_network.p")
ndcg_samples_path = os.path.join(data_dir, "ndcg_samples.p")
knrm_train_data_path = os.path.join(knrm_dir, "train.txt")
knrm_test_data_path = os.path.join(knrm_dir, "test.txt")

training_rate = 0.3

# MongoDB Information
MONGO_HOST = "127.0.0.1"
MONGO_PORT = 27017
if dataset == "quora":
    MONGO_DB = "quora"
else:
    MONGO_DB = "qa_zhihu"

embedding_size = 150
attention_num_units = embedding_size

if dataset == "quora":
    max_question_len = 28
    max_answer_len = 100
elif dataset == "zhihu":
    max_question_len = 20
    max_answer_len = 60
elif dataset == "sem_eval":
    max_question_len = 50
    max_answer_len = 50

filter_by_length = True
train_batch_size = 250
test_batch_size = 250
learning_rate = 2e-3
l2_coe = 1e-3
margin = 1.0
test_interval = 400

# a set of candicate answers consists of 1 real answer and 5 fake answers
num_real_samples = 1
num_neg_samples = 5

num_ndcg_samples = 5000#10000

# use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# some constant
TYPE_QUESTION = "QUESTION"
TYPE_ANSWER = "ANSWER"
TYPE_WORD = "WORD"
TYPE_USER = "USER"
TYPE_TAG = "TAG"








