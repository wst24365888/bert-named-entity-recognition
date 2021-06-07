# -*- coding: utf-8 -*-
# @Time : 2021/4/6 22:54
# @Author : luff543
# @Email : luff543@gmail.com
# @File : bert_model.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import bert


class BiLSTM_CRFModel(tf.keras.Model):
    def __init__(self, configs, vocab_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        max_seq_length = configs.max_sequence_length
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

        hub_url = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2"
        # hub_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2"

        self.l_bert = hub.KerasLayer(hub_url, trainable=True)
        pooled_output, sequence_output = self.l_bert(
            [input_word_ids, input_mask, segment_ids])
        self.model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[
                           pooled_output, sequence_output])
        self.model.build(input_shape=[
                         (None, max_seq_length), (None, max_seq_length), (None, max_seq_length)])

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, configs.embedding_dim, mask_zero=True)
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(num_classes, num_classes)))

        vocab_file = self.l_bert.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.l_bert.resolved_object.do_lower_case.numpy()
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)

    @tf.function
    def call(self, X_batch, attention_mask_batch, segment_batch, inputs_length, targets, training=None):
        X_batch = tf.cast(X_batch, tf.int32)
        attention_mask_batch = tf.cast(attention_mask_batch, tf.int32)
        segment_batch = tf.cast(segment_batch, tf.int32)
        embedding_inputs = self.l_bert(
            [X_batch, attention_mask_batch, segment_batch], training=training)[1]

        dropout_inputs = self.dropout(embedding_inputs, training=training)
        bilstm_outputs = self.bilstm(dropout_inputs)
        logits = self.dense(bilstm_outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=tf.int32)
        log_likelihood, self.transition_params = crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params

    def get_tokenizer(self):
        return self.tokenizer
