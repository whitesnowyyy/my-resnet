# '''
# Author: error: git config user.name && git config user.email & please set dead value or install git
# Date: 2022-11-28 09:11:52
# LastEditors: error: git config user.name && git config user.email & please set dead value or install git
# LastEditTime: 2022-11-28 16:09:57
# FilePath: \my\utils.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''

import logging
from logging.handlers import TimedRotatingFileHandler
import datetime
import os
import torch
import random
import numpy as np
import shutil #??


def mkdirs(directory):
    try:
        os.makedirs(directory)
    except Exception as e:
        ...#??


def mkdirsWithFullPath(path):
    path = path.replace("\\", "/")
    p0 = path.rfind('/')
    if p0 != -1 and not os.path.exists(path[:p0]):
        mkdirs(path[:p0])


def getLogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    mkdirsWithFullPath(path)

    rf_handler = logging.handlers.TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s')
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def copy_code_to(src,dst):
    if len(dst)==0 or dst =='.':
        print("invalid opertate,copy to current dirctory")
        return

    for file in os.listdir(src):
        if file.endswith('.py'):
            source = f'{src}/{file}'
            dest = f'{dst}/{file}'
            mkdirsWithFullPath(dest)
            shutil.copy(source,dest)#复制文件