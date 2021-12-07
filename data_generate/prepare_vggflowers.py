import os
import json
from scipy.io import loadmat

from data_generate.split_generator import SplitGenerator

def organise_class_folders(image_folder_dir, label_mat_dir):
    image_file_ls = [
        'image_{}.jpg'.format('0' * (5 - len(str(i))) + str(i)) for i in range(1, len(os.listdir(image_folder_dir)) + 1)
    ]
    label_ls = list(loadmat(label_mat_dir)['labels'][0])
    image_label_dict = dict(zip(image_file_ls, label_ls))

    # create class folders
    for label_int in list(set(label_ls)):
        os.makedirs(os.path.join(image_folder_dir, str(label_int)))

    # move files into respective folder
    for image, label in image_label_dict.items():
        os.rename(
            src=os.path.join(image_folder_dir, image),
            dst=os.path.join(os.path.join(image_folder_dir, str(label)), image)
        )


if __name__ == "__main__":

    # load config file
    jsonfile = open(os.path.join('config/seqdataset.json'))
    config = json.loads(jsonfile.read())

    split_dir = os.path.join(os.path.join(config['data_dir'], 'vggflowers', config['split_folder']))
    dest_dir = os.path.join(config['data_dir'], 'vggflowers')

    # organise images into class folders
    img_folder = os.path.join(config['data_dir'], 'vggflowers_raw', "data")
    organise_class_folders(image_folder_dir=img_folder, label_mat_dir='../data/vggflowers_raw/imagelabels.mat')

    split_vggflowers = SplitGenerator(
        data_dir=img_folder, dest_dir=dest_dir, split_dir=split_dir,
        back_eval_raw=False, supercls_raw=False, supercls_split=config['vggflowers']['supercls']
    )
    split_vggflowers.split_train_val_test(nclass_train=66, nclass_val=16, save_split_npy=True, csv_save_form=None)
