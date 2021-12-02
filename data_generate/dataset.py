from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import data_generate.transformations as transfm

from PIL import Image
from tqdm import tqdm
from itertools import chain
import pandas as pd
import glob
import random

from training.util import split_path

class FewShotImageDataset(Dataset):
    def __init__(self, task_list, supercls=True, img_lvl=1, transform=None, relabel=None, device=None,
                 cuda_img_tensor=True, verbose='dataset'):
        self.task_list = task_list
        self.supercls = supercls # true if data has superclasses
        self.img_lvl = img_lvl # num of level below task dirs where imgs are located
        self.transform = transform
        self._relabel = relabel # tuple (colname, [val1, val2, ...])
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self.verbose = verbose

        self.df = self.generate_task_df()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cuda_img_tensor:
            image = self.df.loc[idx, 'cuda_tensor']
        else:
            image = Image.open(self.df.loc[idx, 'img_path'])
            # check if non-grayscale images are in RGB mode. There's one image in mini_imagenet that has 4 channels.
            # convert grayscale to rgb if image mode is already grayscale but no grayscale transformation in transform list
            if ((image.mode == '1' or image.mode == 'L')
                and sum(transfm.__class__.__name__ == 'Grayscale' for transfm in self.transform.transforms) == 0) \
                    or (image.mode not in ('RGB', '1', 'L')):
                image = image.convert(mode='RGB')
            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(self.df.loc[idx, 'cls_lbl'], device=self.device)
        return image, label

    @property
    def relabel(self):
        return self._relabel

    @relabel.setter
    def relabel(self, labels):
        self._relabel = labels

    def generate_task_df(self):
        img_dict_list = []
        for idx, classdir in \
                (tqdm(enumerate(self.task_list), desc='Generating {}'.format(self.verbose),
                         total=len(self.task_list)) if self.verbose is not None
                else enumerate(self.task_list)):
            # list all img paths in this class
            img_path_list = glob.glob(classdir + '/*' * self.img_lvl)
            for img_path in img_path_list:
                # split img path per folders
                path_split = split_path(img_path)
                # append gpu_tensor (after transformation)
                if self.cuda_img_tensor:
                    image = Image.open(img_path)
                    # check if non-grayscale images are in RGB mode. There's one image in mini_imagenet that has 4 channels.
                    # convert grayscale to rgb if image mode is already grayscale but no grayscale transformation in transform list
                    if ((image.mode == '1' or image.mode == 'L')
                        and sum(transf.__class__.__name__ == 'Grayscale' for transf in self.transform.transforms) == 0) \
                            or (image.mode not in ('RGB', '1', 'L')):
                        image = image.convert(mode='RGB')
                    if self.transform is not None:
                        image = self.transform(image)
                else:
                    image = None
                # append info dict to list
                img_dict_list.append({
                    'supercls': path_split[-3],
                    'cls_name': '{}.{}'.format(path_split[-3], path_split[-2]),
                    'cls_lbl': idx,
                    'img_path': img_path,
                    'cuda_tensor': image
                } if self.supercls else {
                    'cls_name': path_split[-2],
                    'cls_lbl': idx,
                    'img_path': img_path,
                    'cuda_tensor': image
                })
        df_task = pd.DataFrame(img_dict_list)
        return df_task

    def relbl_df(self):
        for ind, value in enumerate(self.relabel[1]):
            self.df.loc[self.df[self.relabel[0]] == value, 'cls_lbl'] = ind


def get_df_inds_per_col_value(df, col, shuffle=True):
    inds_per_val = []
    for colval in df[col].unique():
        inds = df.loc[df[col] == colval].index.tolist()
        if shuffle:
            random.shuffle(inds)
        inds_per_val.append((colval, inds))
    return inds_per_val

def split_traintest_inds_per_cls(indices_per_class, num_test_per_class, rtn_chained=True):
    test_inds_per_cls = []
    train_inds_per_cls = []

    for cls, inds in indices_per_class:
        random.shuffle(inds)
        test_inds_per_cls.append((cls, inds[:num_test_per_class]))
        train_inds_per_cls.append((cls, inds[num_test_per_class:]))

    if rtn_chained:
        train_inds_chained = list(chain.from_iterable(list(zip(*train_inds_per_cls))[1]))
        test_inds_chained = list(chain.from_iterable(list(zip(*test_inds_per_cls))[1]))

        random.shuffle(train_inds_chained)
        random.shuffle(test_inds_chained)
        return train_inds_per_cls, train_inds_chained, test_inds_per_cls, test_inds_chained
    else:
        return train_inds_per_cls, test_inds_per_cls
