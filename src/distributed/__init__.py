# coding=utf-8
import os
DEFAULT_SHARED_DIR = '/tmp'
def distributed_init(shared_fs_path=None):
    global DEFAULT_SHARED_DIR
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    init_method = 'tcp://{address}:{port}'.format(address=master_addr, port=master_port)
