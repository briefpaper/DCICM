# coding=utf-8

import tensorflow as tf
import numpy as np


def co_attention_layer(questions,
                       answers,
                       num_units,
                       drop_rate,
                       scope,
                       residual=True,
                       reuse=False):
    current_questions = questions
    current_answers = answers
    for i in range(2):
        current_questions, q_a_attention_matrix = attention_layer(queries=current_questions,
                                                                  keys=current_answers,
                                                                  num_units=num_units,
                                                                  drop_rate=drop_rate,
                                                                  scope="{}/attention_q_a_{}".format(scope, i),
                                                                  residual=residual,
                                                                  reuse=reuse)
        current_answers, a_q_attention_matrix = attention_layer(queries=current_answers,
                                                                keys=current_questions,
                                                                num_units=num_units,
                                                                drop_rate=drop_rate,
                                                                scope="{}/attention_a_q_{}".format(scope, i),
                                                                residual=residual,
                                                                reuse=reuse)
    return current_questions, current_answers, q_a_attention_matrix, a_q_attention_matrix


def layer_normalize(inputs,
                    epsilon=1e-8,
                    scope="ln",
                    reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        # params_shape = [inputs.shape[-1]]
        # params_shape = [num_units]

        epsilon = tf.constant(np.ones(params_shape, dtype=np.float32) * epsilon)
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", initializer=tf.zeros(params_shape))
        gamma = tf.get_variable("gamma", initializer=tf.ones(params_shape))
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)  # ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        # normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
    return outputs

    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))

    beta = tf.get_variable("beta", shape=params_shape, initializer=0.0)
    gamma = tf.get_variable("gamma", shape=params_shape, initializer=1.0)
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
    return outputs


residual_rate = tf.placeholder(tf.float32, name="residual_rate")


def attention_layer(queries,
                    keys,
                    num_units,
                    drop_rate,
                    scope,
                    residual=True,
                    num_heads=1,
                    reuse=False,
                    bias=None,
                    causality=False,
                    conv_window=3,
                    layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name='Q')
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name='K')
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name='V')

        # Split and concat
        Q = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # (h*N, T_q, C/h)
        K = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)
        V = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)

        attention_matrixes = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        attention_matrixes = attention_matrixes / (K.get_shape().as_list()[-1] ** 0.5)

        # attention_matrixes = tf.ones_like(attention_matrixes, dtype=tf.float32)

        if bias is not None:
            attention_matrixes += bias

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(attention_matrixes) * (-2 ** 32 + 1)
        attention_matrixes = tf.where(tf.equal(key_masks, 0), paddings, attention_matrixes)

        # Activation
        attention_matrixes = tf.nn.softmax(attention_matrixes)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        head_query_masks = tf.tile(query_masks, [num_heads, 1])
        attention_matrixes *= tf.expand_dims(head_query_masks, -1)

        attention_matrixes = tf.layers.dropout(attention_matrixes, drop_rate)

        outputs = tf.matmul(attention_matrixes, V)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # (N, T_q, C)

        if residual:
            outputs = outputs * 0.5 + queries * 0.5
        if layer_norm:
            outputs = layer_normalize(outputs, reuse=reuse)

        return outputs, attention_matrixes


def coattention_block(inputs, context, num_units, drop_rate, scope, residual=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        _, f_attention_matrix = attention_layer(inputs, context, num_units, drop_rate, "f_attention", False,
                                                reuse=reuse)
        attention_matrix = f_attention_matrix
        V = tf.layers.dense(context, num_units, name="V", activation=tf.nn.relu)
        V = tf.layers.dropout(V, rate=drop_rate)
        outputs = tf.matmul(attention_matrix, V)
        if residual:
            outputs = 0.5 * outputs + 0.5 * inputs
    return outputs


def dcicm_match_network(embedded_question, embedded_answer, num_units, drop_rate, reuse):
    with tf.variable_scope("dcicm_network", reuse=reuse):
        current_q = embedded_question
        current_a = embedded_answer
        for i in range(2):
            current_q = coattention_block(current_q, current_a, num_units, drop_rate, "coattention_q_{}".format(i), residual=True, reuse=reuse)
            current_a = coattention_block(current_a, current_q, num_units, drop_rate, "coattention_a_{}".format(i), residual=True, reuse=reuse)
        sim_matrix = sim_matrix_layer(current_q, current_a, drop_rate, norm=False)
        sim_matrix = tf.transpose(sim_matrix, [0, 2, 1])

        current_layers = []
        with tf.variable_scope("sim_conv_1d", reuse=reuse):
            for gram in [2, 3]:
                current_layer = tf.layers.conv1d(sim_matrix, 30, gram, activation=tf.nn.relu, padding="same",
                                                 name="sim_conv_gram_{}".format(gram))
                current_layer = tf.reduce_max(current_layer, axis=-2)
                current_layers.append(current_layer)
            current_layer = tf.concat(current_layers, axis=-1)
            current_layer = tf.layers.dropout(current_layer, rate=drop_rate)

        with tf.variable_scope("pool_w", reuse=reuse):
            current_layer = tf.layers.dense(current_layer, 1, name="pool_W1")

        return current_layer




def sparsity_match_network(embedded_question, embedded_answer, num_units, drop_rate, reuse):
    with tf.variable_scope("dcicm_network", reuse=reuse):
        current_q = embedded_question
        current_a = embedded_answer

        pre_sim_matrix = sim_matrix_layer(embedded_question, embedded_answer, drop_prob=0.0, norm=False)


        for i in range(2):
            current_q = coattention_block(current_q, current_a, num_units, drop_rate, "coattention_q_{}".format(i), residual=True, reuse=reuse)
            current_a = coattention_block(current_a, current_q, num_units, drop_rate, "coattention_a_{}".format(i), residual=True, reuse=reuse)
        sim_matrix = sim_matrix_layer(current_q, current_a, drop_rate, norm=False)
        post_sim_matrix = sim_matrix_layer(current_q, current_a, drop_prob=0.0, norm=False)
        sim_matrix = tf.transpose(sim_matrix, [0, 2, 1])

        # pre_mean = tf.reduce_sum(tf.cast(tf.greater(pre_sim_matrix, 0.0), tf.float32))
        # post_mean = tf.reduce_sum(tf.cast(tf.greater(post_sim_matrix, 0.0), tf.float32))
        pre_mean = tf.reduce_mean(tf.reduce_max(pre_sim_matrix, axis=-1))
        post_mean = tf.reduce_mean(tf.reduce_max(post_sim_matrix, axis=-1))

        current_layers = []
        with tf.variable_scope("sim_conv_1d", reuse=reuse):
            for gram in [2, 3]:
                current_layer = tf.layers.conv1d(sim_matrix, 30, gram, activation=tf.nn.relu, padding="same",
                                                 name="sim_conv_gram_{}".format(gram))
                current_layer = tf.reduce_max(current_layer, axis=-2)
                current_layers.append(current_layer)
            current_layer = tf.concat(current_layers, axis=-1)
            current_layer = tf.layers.dropout(current_layer, rate=drop_rate)

        with tf.variable_scope("pool_w", reuse=reuse):
            current_layer = tf.layers.dense(current_layer, 1, name="pool_W1")

        return current_layer, pre_mean, post_mean


def smatrix(embedded_question, embedded_answer, num_units, drop_rate, reuse, q_depth=1, a_depth=1, recurrent=False,
            shared=False):
    sim_matrix = sim_matrix_layer(embedded_question, embedded_answer, drop_rate, norm=True)
    sim_matrix = tf.expand_dims(sim_matrix, axis=-1)
    match_score = smatrix_conv_score(sim_matrix, drop_rate, "sim_matrix", reuse=reuse)
    return match_score


def smatrix_conv_score(sim_matrix, drop_rate, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        padding = "same"
        current_layer = tf.layers.conv2d(sim_matrix, 20, 5, [1, 1], padding=padding, activation=tf.nn.tanh,
                                         name='conv_0')
        current_layer = tf.layers.max_pooling2d(current_layer, [2, 2], [2, 2], padding=padding)
        current_layer = tf.layers.conv2d(current_layer, 50, 5, [1, 1], padding=padding, activation=tf.nn.tanh,
                                         name='conv_1')
        current_layer = tf.layers.max_pooling2d(current_layer, [2, 2], [2, 2], padding=padding)

        current_layer = tf.layers.flatten(current_layer)

        current_layer = tf.layers.dense(current_layer, 500, activation=tf.nn.tanh, name='W_1')
        current_layer = tf.layers.dropout(current_layer, drop_rate)
        current_layer = tf.layers.dense(current_layer, 1, activation=tf.nn.tanh, name='W_2')
    return current_layer


def sim_matrix_layer(transformed_question, transformed_answer, drop_prob, expand=False, norm=True, reuse=False,
                     activation=None):
    def normalize(input):
        if norm:
            return tf.nn.l2_normalize(input, axis=-1)
        else:
            return input

    transformed_question = normalize(transformed_question)
    transformed_answer = normalize(transformed_answer)
    result = tf.matmul(transformed_question, tf.transpose(transformed_answer, [0, 2, 1]))
    if activation is not None:
        result = activation(result)
    result = tf.layers.dropout(result, drop_prob)
    if expand:
        result = tf.expand_dims(result, -1)
    return result


l2_loss = None  # tf.nn.l2_loss


def multi_filter_conv_score(sim_matrix, drop_rate, scope, reuse):
    num_conv_attention_filters = 20
    num_filters_list = [20]  # [10, 10]#, 15]  # , 20]# 60, 80]
    multi_kernels = [5]  # ,4]

    def conv(sim_matrix, kernel, num_filters):
        padding = "same"
        current_layer = tf.layers.conv2d(sim_matrix, num_filters, kernel, [1, 1], padding=padding,
                                         activation=tf.nn.relu, name='conv_0_gram_{}'.format(kernel))
        current_layer = tf.layers.max_pooling2d(current_layer, [2, 2], [2, 2], padding=padding)
        # current_layer = tf.layers.batch_normalization(current_layer, axis=-1, name="gram_{}_0".format(kernel))
        current_layer = tf.layers.conv2d(current_layer, 2, 2, [1, 1], padding=padding,
                                         activation=tf.nn.relu, name='conv_1_gram_{}'.format(kernel))
        current_layer = tf.layers.max_pooling2d(current_layer, [2, 2], [2, 2], padding=padding)
        current_layer = tf.layers.flatten(current_layer)
        return current_layer

    with tf.variable_scope(scope, reuse=reuse):
        layers = [conv(sim_matrix, kernel, num_filters) for kernel, num_filters in zip(multi_kernels, num_filters_list)]
        current_layer = tf.concat(layers, axis=-1)
        current_layer = tf.layers.dense(current_layer, 20, activation=tf.nn.relu, name='W_1')
        current_layer = tf.layers.dropout(current_layer, drop_rate)
        current_layer = tf.layers.dense(current_layer, 1, activation=None, name='W_2')
    return current_layer


def embed(embeddings, indice, drop_prob):
    result = tf.nn.embedding_lookup(embeddings, indice)
    result = tf.layers.dropout(result, drop_prob)
    return result
