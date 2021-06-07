# -*- coding: utf-8 -*-
# @Time : 2021/4/6 22:55
# @Author : luff543
# @Email : luff543@gmail.com
# @File : predict.py
# @Software: PyCharm

import tensorflow as tf
from engines.bert_model import BiLSTM_CRFModel
from tensorflow_addons.text.crf import crf_decode
import math
from tqdm import tqdm
import codecs


class Predictor:
    def __init__(self, configs, data_manager, logger, bilstm_crf_model):
        self.data_manager = data_manager
        vocab_size = data_manager.max_token_number
        num_classes = data_manager.max_label_number
        self.logger = logger
        self.configs = configs
        logger.info('loading model')
        self.bilstm_crf_model = bilstm_crf_model

        if (bilstm_crf_model == None):
            self.bilstm_crf_model = BiLSTM_CRFModel(
                configs, vocab_size, num_classes)
            checkpoint = tf.train.Checkpoint(model=self.bilstm_crf_model)
            checkpoint.restore(tf.train.latest_checkpoint(
                configs.checkpoints_dir))  # 從文件恢復模型參數
            logger.info('loading model successfully')
        self.tokenizer = self.bilstm_crf_model.get_tokenizer()

    def metrics(self, X, y_true, y_pred, configs, data_manager, tokenizer):
        ner_results = []
        # tensor向量不能直接索引，需要轉成numpy
        y_pred = y_pred.numpy()
        for i in range(len(y_true)):
            x = tokenizer.convert_ids_to_tokens(X[i].tolist())
            x[0] = '[CLS]'

            y = [str(data_manager.id2label[val]) for val in y_true[i] if
                 val != data_manager.label2id[data_manager.PADDING]]
            y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                     val != data_manager.label2id[data_manager.PADDING]]
            y_hat_crop = y_hat[0:len(y)]

            ner_results.append((x, y, y_hat_crop))
        return ner_results

    def evaluate_test(self, data_manager, epoch=None):
        batch_size = self.configs.batch_size
        X_test, y_test, att_mask_test, segment_test = self.data_manager.get_testing_set()

        all_ner_results = []
        num_test_iterations = int(math.ceil(1.0 * len(X_test) / batch_size))
        for iteration in tqdm(range(num_test_iterations)):
            X_test_batch, y_test_batch, att_mask_batch, segement_batch = data_manager.next_batch(
                X_test, y_test, att_mask_test, segment=segment_test, start_index=iteration * batch_size)
            # 計算沒有加入pad之前的句子的長度
            inputs_length_test = tf.math.count_nonzero(X_test_batch, 1)

            logits_test, log_likelihood_test, transition_params_test = self.bilstm_crf_model.call(
                X_batch=X_test_batch, attention_mask_batch=att_mask_batch, segment_batch=segement_batch,
                inputs_length=inputs_length_test, targets=y_test_batch, training=False)
            test_loss = -tf.reduce_mean(log_likelihood_test)
            batch_pred_sequence_test, _ = crf_decode(
                logits_test, transition_params_test, inputs_length_test)
            ner_results = self.metrics(
                X_test_batch, y_test_batch, batch_pred_sequence_test, self.configs, self.data_manager, self.tokenizer)
            all_ner_results.extend(ner_results)

        def result_to_pair(writer):
            for ner_result in all_ner_results:
                tokens = ner_result[0]
                labels = ner_result[1]
                preds = ner_result[2]

                line = ''
                for current_token, current_label, current_pred in zip(tokens, labels, preds):
                    try:
                        line += current_token + ' ' + current_label + ' ' + current_pred + '\n'
                    except Exception as e:
                        self.logger.info(e)
                        self.logger.info(str(tokens))
                        self.logger.info(str(labels))
                        line = ''

                writer.write(line + '\n')

        predit_file = self.configs.datasets_fold + \
            '/' + 'label_test_' + str(epoch) + '.txt'

        if epoch == None:
            predit_file = self.configs.datasets_fold + '/' + 'label_test_output' + '.txt'
        with codecs.open(predit_file, 'w',
                         encoding='utf-8') as writer:
            result_to_pair(writer)
