import json
import os
import shutil
import numpy as np

from split_generator import SplitGenerator

if __name__ == "__main__":

    jsonfile = open(os.path.join('config/seqdataset.json'))
    config = json.loads(jsonfile.read())

    split_dir = os.path.join(os.path.join(config['data_dir'], 'cifar_fs', config['split_folder']))
    dest_dir = os.path.join(config['data_dir'], 'cifar_fs')

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)

    metatrain = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
                 for line in open('data/cifar100_raw/splits/bertinetto/train.txt', 'r')]
    metaval = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
               for line in open('data/cifar100_raw/splits/bertinetto/val.txt', 'r')]
    metatest = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
                for line in open('data/cifar100_raw/splits/bertinetto/test.txt', 'r')]

    
    np.save(os.path.join(split_dir, 'metatrain.npy'), metatrain)
    np.save(os.path.join(split_dir, 'metaval.npy'), metaval)
    np.save(os.path.join(split_dir, 'metatest.npy'), metatest)
