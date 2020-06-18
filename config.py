# -*- coding: utf-8 -*-
"""
Created on 2017/6/9
@author: MG
"""
import logging

from ibats_common.config import ConfigBase, update_config

logger = logging.getLogger(__name__)


class Config(ConfigBase):
    DB_SCHEMA_IBATS = 'ibats'
    DB_URL_DIC = {
        DB_SCHEMA_IBATS: 'mysql://mg:Abcd1234@localhost:3307/' + DB_SCHEMA_IBATS,
    }

    BACKTEST_UPDATE_OR_INSERT_PER_ACTION = False
    ORM_UPDATE_OR_INSERT_PER_ACTION = True
    UPDATE_STG_RUN_STATUS_DETAIL_PERIOD = 1  # 1 每一个最小行情周期，2 每天


# 开发配置（SIMNOW MD + Trade）
config = Config()
update_config(config)
