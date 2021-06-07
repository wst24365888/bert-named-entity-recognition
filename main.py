# -*- coding: utf-8 -*-
# @Time : 2021/4/7 14:51
# @Author : luff543
# @Email : luff543@gmail.com
# @File : main.py
# @Software: PyCharm

import argparse
import random
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TFHUB_CACHE_DIR"] = 'tfhub/'
import tensorflow as tf
from engines.train import train
from engines.data import BertDataManager
from engines.configure import Configure
from engines.utils.io import fold_check
from engines.utils.logger import get_logger
from engines.predict import Predictor


def set_env(configures):
    random.seed(configures.seed)
    np.random.seed(configures.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with BiLSTM+CRF')
    parser.add_argument(
        '--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(
        devices=gpus[configs.CUDA_VISIBLE_DEVICES], device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    set_env(configs)
    mode = configs.mode.lower()

    if mode == 'train':
        data_manager = BertDataManager(configs, logger)
        logger.info('mode: train')
        train(configs, data_manager, logger)
    elif mode == 'evaluate_test':
        configs.batch_size = 1
        data_manager = BertDataManager(configs, logger)
        logger.info('evaluate_test')
        predictor = Predictor(
            configs=configs, data_manager=data_manager, logger=logger, bilstm_crf_model=None)
        predictor.evaluate_test(data_manager=data_manager, epoch=None)
