from __future__ import print_function
import sys
import subprocess
import os
import logging

LOG_LEVEL_INFO = 0
LOG_LEVEL_WARNING = 1
LOG_LEVEL_ERROR = 2


def print_fn(log, level=0):
    if level == LOG_LEVEL_INFO:
        logging.info(log)
    elif level == LOG_LEVEL_WARNING:
        logging.warning(log)
    elif level == LOG_LEVEL_ERROR:
        logging.error(log)
        exit()


def make_dir_if_not_exist(local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        logging.info('Created directory %s', local_path)


def mkdir(folder_path):
    cmd = 'mkdir -p ' + folder_path
    ret = subprocess.check_call(cmd, shell=True)
    print_fn(ret)


def search_dict_list(dict_list, key, value):
    '''
    Search the targeted <key, value> in the dict_list
    Return:
        list entry, or just None 
    '''
    for e in dict_list:
        # if e.has_key(key) == True:
        if key in e:
            if e[key] == value:
                return e

    return None
