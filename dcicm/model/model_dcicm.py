# coding=utf-8

from cqabaseline.evaluation import evaluate_mean_ndcg_and_precision
from dcicm.model.modules import *
from dcicm.data.load_data import *
import math
import tensorflow as tf

question_datas = train_network.type_data_dict[TYPE_QUESTION]
answer_datas = train_network.type_data_dict[TYPE_ANSWER]

drop_rate_placeholder = tf.placeholder(tf.float32)
word_embeddings = tf.get_variable("word_embeddings", [vocab_size, embedding_size],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(embedding_size)))
word_embeddings = tf.concat((tf.zeros((1, embedding_size)), word_embeddings[1:]), 0)

question_placeholder = tf.placeholder(tf.int32, [None, max_question_len])
positive_answer_placeholder = tf.placeholder(tf.int32, [None, max_answer_len])
negative_answer_placeholder = tf.placeholder(tf.int32, [None, max_answer_len])

embedded_question = embed(word_embeddings, question_placeholder, drop_rate_placeholder)
embedded_positive_answer = embed(word_embeddings, positive_answer_placeholder, drop_rate_placeholder)
embedded_negative_answer = embed(word_embeddings, negative_answer_placeholder, drop_rate_placeholder)

match_network = dcicm_match_network
positive_match_score = match_network(embedded_question, embedded_positive_answer, attention_num_units, drop_rate_placeholder, reuse=False)
negative_match_score = match_network(embedded_question, embedded_negative_answer, attention_num_units, drop_rate_placeholder, reuse=True)

positive_losses = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(positive_match_score),
    logits=positive_match_score
)

negative_losses = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(negative_match_score),
    logits=negative_match_score
)

losses = positive_losses + negative_losses
# losses = tf.maximum(margin - (positive_match_score - negative_match_score), 0)

correct = tf.cast(tf.greater(positive_match_score, negative_match_score), tf.float32)
accuracy = tf.reduce_mean(correct)

kernel_vars = [var for var in tf.global_variables() if "kernel" in var.name]
l2_losses = sum([tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]) * l2_coe
mean_loss = tf.reduce_mean(losses)# + l2_losses
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

train_batch_generator = create_triple_batch_generator(train_network, [TYPE_QUESTION, TYPE_ANSWER], train_batch_size, True)
test_batch_generator = create_triple_batch_generator(test_network, [TYPE_QUESTION, TYPE_ANSWER], test_batch_size, False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(200000):
        if step % test_interval == 0:
            print("eval step={}".format(step))
            test_match_scores = []
            eval_batch_generator = create_eval_batch_generator(test_network, [TYPE_QUESTION, TYPE_ANSWER], test_batch_size, False)
            correct_list = []
            for test_step, batch_data in enumerate(eval_batch_generator):
                test_batch_question_datas, test_batch_answer_datas = batch_data
                test_feed_dict = {
                    question_placeholder: test_batch_question_datas,
                    positive_answer_placeholder: test_batch_answer_datas,
                    drop_rate_placeholder: 0.0
                }
                shape = (max_question_len, max_answer_len)
                positive_match_score_val = sess.run(positive_match_score, feed_dict=test_feed_dict).flatten()
                test_match_scores.extend(positive_match_score_val)

            num_trained_sample = train_batch_size * step
            print("step = {}\t samples = {}".format(step, num_trained_sample))

            match_scores_list = np.reshape(test_match_scores, [num_ndcg_samples, num_real_samples + num_neg_samples])
            evaluate_mean_ndcg_and_precision(match_scores_list)

        if step % 1 == 0:
            train_batch_question_datas, train_batch_answer_datas, train_batch_negative_answer_datas = next(
                train_batch_generator)
            main_feed_dict = {
                question_placeholder: train_batch_question_datas,
                positive_answer_placeholder: train_batch_answer_datas,
                negative_answer_placeholder: train_batch_negative_answer_datas,
                drop_rate_placeholder: 0.1
            }
            for _ in range(1):
                _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=main_feed_dict)
            if step % (test_interval // 2) == 0:
                print("step = {}\tloss = {}".format(step, mean_loss_val))
