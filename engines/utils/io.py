# -*- coding: utf-8 -*-
# @Time : 2021/3/19 19:27
# @Author : luff543
# @Email : luff543@gmail.com
# @File : io.py
# @Software: PyCharm

import os


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configures.checkpoints_dir) or not hasattr(configures, checkpoints_dir):
        print('checkpoints fold not found, creating...')
        paths = configures.checkpoints_dir.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(configures.checkpoints_dir):
            os.mkdir(configures.checkpoints_dir)
        else:
            os.mkdir('checkpoints')

    vocabs_dir = 'vocabs_dir'
    if not os.path.exists(configures.vocabs_dir):
        print('vocabs fold not found, creating...')
        if hasattr(configures, vocabs_dir):
            os.mkdir(configures.vocabs_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(configures.log_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')

    tfhub_dir = 'tfhub/'
    if not os.path.exists(tfhub_dir):
        print('tfhub fold not found, creating...')
        os.mkdir(tfhub_dir)