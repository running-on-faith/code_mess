#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/7/1 21:10
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from logging.config import dictConfig

from ibats_utils.mess import is_windows_os

DATA_FOLDER_PATH = r'D:\WSPych\IBATSCommon\ibats_common\example\data' if is_windows_os() else r'/home/mg/github/IBATS_Common/ibats_common/example/data'
MODEL_SAVED_FOLDER = 'models'
MODEL_ANALYSIS_IMAGES_FOLDER = 'images'
MODEL_REWARDS_FOLDER = 'rewards'
TENSORBOARD_LOG_FOLDER = 'tb_log'
# evn configuration
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(filename)s.%(funcName)s:%(lineno)d|%(message)s'

# log settings
logging_config = dict(
    version=1,
    formatters={
        'simple': {
            'format': LOG_FORMAT}
    },
    handlers={
        'file_handler':
            {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logger.log',
                'maxBytes': 1024 * 1024 * 10,
                'backupCount': 5,
                'level': 'DEBUG',
                'formatter': 'simple',
                'encoding': 'utf8'
            },
        'console_handler':
            {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'simple'
            }
    },

    root={
        'handlers': ['console_handler', 'file_handler'],
        'level': logging.DEBUG,
    }
)
dictConfig(logging_config)
logging.info('set logging finished')
