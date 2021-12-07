import os
import glob
import random
import csv
import numpy as np
import shutil
from distutils.dir_util import copy_tree
import cv2
from tqdm import tqdm

class SplitGenerator(object):
    def __init__(self, data_dir='../data/omniglot_raw', dest_dir='./data/omniglot', split_dir='./data/omniglot/split',
                 back_eval_raw=True, supercls_raw=True, supercls_split=False):
        self.data_dir = data_dir
        self.dest_dir = dest_dir
        self.split_dir = split_dir
        self.back_eval_raw = back_eval_raw
        self.supercls_raw = supercls_raw
        self.supercls_split = supercls_split

    def split_train_val_test(self, nclass_train=1150, nclass_val=50, save_split_npy=False, csv_save_form=None):
        
        classdir_all = glob.glob(self.data_dir + '/*' * (self.supercls_raw + (not self.supercls_split)))

        self.classdir_train = random.sample(classdir_all, nclass_train)
        classdir_excl_train = set(classdir_all) - set(self.classdir_train)

        self.classdir_val = random.sample(list(classdir_excl_train), nclass_val)
        self.classdir_test = list(classdir_excl_train - set(self.classdir_val))

        if save_split_npy:
            if os.path.exists(self.split_dir):
                # print('Split-in-npy destination folder not empty. Deleting...')
                shutil.rmtree(self.split_dir)
            os.makedirs(self.split_dir)

            # print('Saving split-in-npy...')
            np.save(os.path.join(self.split_dir, 'metatrain.npy'), self.classdir_train)
            np.save(os.path.join(self.split_dir, 'metaval.npy'), self.classdir_val)
            np.save(os.path.join(self.split_dir, 'metatest.npy'), self.classdir_test)

        if csv_save_form is not None:
            split_csv_dir = os.path.join(self.dest_dir, 'csv')
            if os.path.exists(split_csv_dir):
                print('Split-in-csv destination folder not empty. Deleting...')
                shutil.rmtree(split_csv_dir)
            os.makedirs(split_csv_dir)

            if csv_save_form == 'classonly':
                with open(os.path.join(split_csv_dir, 'metatrain.csv'), 'w', newline='') as csvtrain:
                    train_writer = csv.writer(csvtrain, delimiter=',')
                    train_writer.writerow(self.classdir_train)
                with open(os.path.join(split_csv_dir, 'metaval.csv'), 'w', newline='') as csvval:
                    val_writer = csv.writer(csvval, delimiter=',')
                    val_writer.writerow(self.classdir_val)
                with open(os.path.join(split_csv_dir, 'metatest.csv'), 'w', newline='') as csvtest:
                    test_writer = csv.writer(csvtest, delimiter=',')
                    test_writer.writerow(self.classdir_test)
            else:
                raise NotImplementedError('csv_save_form can only be "classonly".')

    def generate_foldersplit(self, save_split_npy=True):
        self.train_dir = os.path.join(self.dest_dir, 'metatrain')
        self.val_dir = os.path.join(self.dest_dir, 'metaval')
        self.test_dir = os.path.join(self.dest_dir, 'metatest')

        # remove all classes if train, val, test non empty
        if os.path.exists(self.train_dir):
            print('Train destination folder not empty. Deleting...')
            shutil.rmtree(self.train_dir)
        if os.path.exists(self.val_dir):
            print('Val destination folder not empty. Deleting...')
            shutil.rmtree(self.val_dir)
        if os.path.exists(self.test_dir):
            print('Test destination folder not empty. Deleting...')
            shutil.rmtree(self.test_dir)

        os.makedirs(self.train_dir)
        os.makedirs(self.val_dir)
        os.makedirs(self.test_dir)

        classdir_train = self.classdir_train
        classdir_val = self.classdir_val
        classdir_test = self.classdir_test

        len_classdir_train = len(classdir_train)
        len_classdir_val = len(classdir_val)
        len_classdir_test = len(classdir_test)

        for idx, classdir in enumerate(classdir_train):
            if self.supercls_raw and not self.supercls_split:
                language_dir, character_name = os.path.split(os.path.normpath(classdir))
                _, language_name = os.path.split(language_dir)
                language_character_name = language_name + '_' + character_name
                print('Copying train %s of %s class %s' % (idx + 1, len_classdir_train, language_character_name))
                copy_tree(classdir, os.path.join(self.train_dir, language_character_name))
            else:
                classname = os.path.basename(os.path.normpath(classdir))
                print('Copying train %s of %s class %s' % (idx + 1, len_classdir_train, classname))
                copy_tree(classdir, os.path.join(self.train_dir, classname))

        for idx, classdir in enumerate(classdir_val):
            if self.supercls_raw and not self.supercls_split:
                language_dir, character_name = os.path.split(os.path.normpath(classdir))
                _, language_name = os.path.split(language_dir)
                language_character_name = language_name + '_' + character_name
                print('Copying val %s of %s class %s' % (idx + 1, len_classdir_val, language_character_name))
                copy_tree(classdir, os.path.join(self.val_dir, language_character_name))
            else:
                classname = os.path.basename(os.path.normpath(classdir))
                print('Copying val %s of %s class %s' % (idx + 1, len_classdir_val, classname))
                copy_tree(classdir, os.path.join(self.val_dir, classname))

        for idx, classdir in enumerate(classdir_test):
            if self.supercls_raw and not self.supercls_split:
                language_dir, character_name = os.path.split(os.path.normpath(classdir))
                _, language_name = os.path.split(language_dir)
                language_character_name = language_name + '_' + character_name
                print('Copying test %s of %s class %s' % (idx + 1, len_classdir_test, language_character_name))
                copy_tree(classdir, os.path.join(self.test_dir, language_character_name))
            else:
                classname = os.path.basename(os.path.normpath(classdir))
                print('Copying test %s of %s class %s' % (idx + 1, len_classdir_test, classname))
                copy_tree(classdir, os.path.join(self.test_dir, classname))

        if save_split_npy:
            if os.path.exists(self.split_dir):
                print('Split-in-npy destination folder not empty. Deleting...')
                shutil.rmtree(self.split_dir)
            os.makedirs(self.split_dir)

            print('Saving split-in-npy...')
            np.save(os.path.join(self.split_dir, 'metatrain.npy'), glob.glob(self.train_dir + '/*'))
            np.save(os.path.join(self.split_dir, 'metaval.npy'), glob.glob(self.val_dir + '/*'))
            np.save(os.path.join(self.split_dir, 'metatest.npy'), glob.glob(self.test_dir + '/*'))

    def augment_img(self, subset='train', num_aug_proc_batch=1, num_aug=500, iaa_seq=None):
        '''
        Augmentations happens after splitting meta-train, -val and -test
        :param subset: subset to augment, str or tuple of str, any of 'train', 'val, 'test, 'background', 'eval'
        :param max_proc_batchsz:
        :param iaa_seq:
        :param iaa_seq_kwargs:
        :return:
        '''
        # list all classes in subset dirs
        if isinstance(subset, str):
            subset = (subset, )
        allsubset_clsls = []
        for name in subset:
            if name in ('train', 'val', 'test'):
                allsubset_clsls.append(glob.glob(getattr(self, name + '_dir') + '/*' * (self.supercls_split + 1)))
            elif name in ('background', 'eval'):
                bgeval_subset_dir = glob.glob(self.data_dir + '/*/')
                subset_dir = list(filter(lambda x: name in x, bgeval_subset_dir))[0]
                allsubset_clsls.append(glob.glob(subset_dir + '/*' * (self.supercls_raw + 1)))
            else:
                raise ValueError("subset must be str or tuple of str, any of 'train', 'val, 'test, 'background', 'eval'")
        # augment for each subset
        for clsls in allsubset_clsls:
            for clsdir in tqdm(clsls):
                # list all images in the dir
                imgdir_ls = glob.glob(clsdir + '/*')
                # split imgdir ls into num_proc_batch arrays
                imgdir_ls_split = np.array_split(imgdir_ls, indices_or_sections=num_aug_proc_batch)
                # augment by batch
                for imgdir_array in imgdir_ls_split:
                    imgdir_ls = list(imgdir_array) # change array to list
                    img_ls = [cv2.imread(imgdir) for imgdir in imgdir_ls]
                    imgs_np = np.stack(img_ls, axis=0).astype('uint8')
                    for idx in range(num_aug):
                        imgs_aug = iaa_seq(images=imgs_np)
                        # write images
                        for imgdir, img_aug in zip(imgdir_ls, list(imgs_aug)):
                            imgdir_root, imgdir_ext = os.path.splitext(imgdir)
                            cv2.imwrite(imgdir_root + '_aug' + str(idx) + imgdir_ext, img_aug)

    def augment_cls(self, type='rotation', subset_to_aug=('train', 'val', 'test'), save_split_npy=True):
        # only supporting rotation atm
        if type == 'rotation':
            # list all classes in train val test dir
            train_classlist = glob.glob(self.train_dir + '/*' * (self.supercls_split + 1))
            val_classlist = glob.glob(self.val_dir + '/*' * (self.supercls_split + 1))
            test_classlist = glob.glob(self.test_dir + '/*' * (self.supercls_split + 1))

            len_train_classlist = len(train_classlist)
            len_val_classlist = len(val_classlist)
            len_test_classlist = len(test_classlist)

            # augment for train
            if 'train' in subset_to_aug:
                for idx, classdir in enumerate(train_classlist):
                    print('Augmenting training folder %s of %s' % (idx + 1, len_train_classlist))
                    # make new folder for rotated 90, 180, 270
                    imgdir_rot90 = classdir + '_rotate090'
                    imgdir_rot180 = classdir + '_rotate180'
                    imgdir_rot270 = classdir + '_rotate270'
                    os.makedirs(imgdir_rot90)
                    os.makedirs(imgdir_rot180)
                    os.makedirs(imgdir_rot270)
                    # list all imgs in this dir
                    train_imglist = os.listdir(classdir)
                    for imgname in train_imglist:
                        img = cv2.imread(os.path.join(classdir, imgname))
                        (height, width) = img.shape[:-1]
                        center = (height / 2, width / 2)

                        rotmat90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                        rotmat180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                        rotmat270 = cv2.getRotationMatrix2D(center, 270, 1.0)

                        img_rot90 = cv2.warpAffine(img, rotmat90, (height, width))
                        img_rot180 = cv2.warpAffine(img, rotmat180, (width, height))
                        img_rot270 = cv2.warpAffine(img, rotmat270, (height, width))

                        cv2.imwrite(os.path.join(imgdir_rot90, 'rot090_' + imgname), img_rot90)
                        cv2.imwrite(os.path.join(imgdir_rot180, 'rot180_' + imgname), img_rot180)
                        cv2.imwrite(os.path.join(imgdir_rot270, 'rot270_' + imgname), img_rot270)

            if 'val' in subset_to_aug:
                for idx, classdir in enumerate(val_classlist):
                    print('Augmenting val folder %s of %s' % (idx + 1, len_val_classlist))
                    # make new folder for rotated 90, 180, 270
                    imgdir_rot90 = classdir + '_rotate090'
                    imgdir_rot180 = classdir + '_rotate180'
                    imgdir_rot270 = classdir + '_rotate270'
                    os.makedirs(imgdir_rot90)
                    os.makedirs(imgdir_rot180)
                    os.makedirs(imgdir_rot270)
                    # list all imgs in this dir
                    val_imglist = os.listdir(classdir)
                    for imgname in val_imglist:
                        img = cv2.imread(os.path.join(classdir, imgname))
                        (height, width) = img.shape[:-1]
                        center = (height / 2, width / 2)

                        rotmat90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                        rotmat180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                        rotmat270 = cv2.getRotationMatrix2D(center, 270, 1.0)

                        img_rot90 = cv2.warpAffine(img, rotmat90, (height, width))
                        img_rot180 = cv2.warpAffine(img, rotmat180, (width, height))
                        img_rot270 = cv2.warpAffine(img, rotmat270, (height, width))

                        cv2.imwrite(os.path.join(imgdir_rot90, 'rot090_' + imgname), img_rot90)
                        cv2.imwrite(os.path.join(imgdir_rot180, 'rot180_' + imgname), img_rot180)
                        cv2.imwrite(os.path.join(imgdir_rot270, 'rot270_' + imgname), img_rot270)

            if 'test' in subset_to_aug:
                for idx, classdir in enumerate(test_classlist):
                    print('Augmenting test folder %s of %s' % (idx + 1, len_test_classlist))
                    # make new folder for rotated 90, 180, 270
                    imgdir_rot90 = classdir + '_rotate090'
                    imgdir_rot180 = classdir + '_rotate180'
                    imgdir_rot270 = classdir + '_rotate270'
                    os.makedirs(imgdir_rot90)
                    os.makedirs(imgdir_rot180)
                    os.makedirs(imgdir_rot270)
                    # list all imgs in this dir
                    test_imglist = os.listdir(classdir)
                    for imgname in test_imglist:
                        img = cv2.imread(os.path.join(classdir, imgname))
                        (height, width) = img.shape[:-1]
                        center = (height / 2, width / 2)

                        rotmat90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                        rotmat180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                        rotmat270 = cv2.getRotationMatrix2D(center, 270, 1.0)

                        img_rot90 = cv2.warpAffine(img, rotmat90, (height, width))
                        img_rot180 = cv2.warpAffine(img, rotmat180, (width, height))
                        img_rot270 = cv2.warpAffine(img, rotmat270, (height, width))

                        cv2.imwrite(os.path.join(imgdir_rot90, 'rot090_' + imgname), img_rot90)
                        cv2.imwrite(os.path.join(imgdir_rot180, 'rot180_' + imgname), img_rot180)
                        cv2.imwrite(os.path.join(imgdir_rot270, 'rot270_' + imgname), img_rot270)

        if save_split_npy:
            if os.path.exists(self.split_dir):
                print('Split-in-npy destination folder not empty. Deleting...')
                shutil.rmtree(self.split_dir)
            os.makedirs(self.split_dir)
            # save train, val, test classes after augmentation
            print('Saving split-in-npy...')
            np.save(os.path.join(self.split_dir, 'metatrain.npy'), glob.glob(self.train_dir + '/*'))
            np.save(os.path.join(self.split_dir, 'metaval.npy'), glob.glob(self.val_dir + '/*'))
            np.save(os.path.join(self.split_dir, 'metatest.npy'), glob.glob(self.test_dir + '/*'))