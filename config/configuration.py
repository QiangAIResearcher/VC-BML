import os
import json
from warnings import warn


def get_run_name(dataset_ls):
    run_name = ''
    for task in dataset_ls:
        run_name += '{}-'.format(task)
    run_name = run_name[:-1]
    return run_name


class config_object(object):
    def __init__(self, config):
        self.config = config
        # for key, value in config.items():
        #     setattr(self, key, value)

    def save_config(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            warn('Experiment directory exists. Earlier files might be overwritten.')
        with open(os.path.join(directory, '{}.json'.format(filename)), 'w') as outfile:
            outfile.write(json.dumps(self.config, indent=4))

    def eval_func(self, rtn_config_dict=True):
        for config_key, config_val in self.config.items():
            if 'dict' in str(type(config_val)):
                for config_val_key, config_val_val in config_val.items():
                    if 'str' in str(type(config_val_val)) and 'lambda' in config_val_val:
                        self.config[config_key][config_val_key] = eval(config_val_val)
        if rtn_config_dict:
            return self.config
