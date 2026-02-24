import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from lib.data.dataset import CustomDataset


class LoadDataSplit:
    """
    Manages the loading, parsing, and organization of image-caption datasets into train, validation, and test splits.

    Args:
        src_id (dict): Paths for train, val, and test ID files.
        src_cap (str): Path to the raw caption file.
        src_img (str): Directory path containing the images.
        train_transforms (callable, optional): Image augmentation/preprocessing for training.
        val_transforms (callable, optional): Image preprocessing for validation/testing.
        caption_mode (str): Method for handling captions (default: 'augmentation').
    """
    def __init__(
            self, src_id, 
            src_cap, src_img, 
            train_transforms=None, 
            val_transforms=None, 
            caption_mode='augmentation'
    ):
        self.captions = LoadDataSplit.read_captions(src_cap=src_cap)
        self.split_ids = LoadDataSplit.read_split_id(src_id=src_id)
        self.src_img = src_img
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.caption_mode = caption_mode

    @staticmethod
    def read_split_id(src_id):
        """
        Reads text files containing image IDs and categorizes them by dataset split.

        Args:
            src_id (dict): Dictionary containing file paths for 'train', 'val', and 'test'.

        Returns:
            dict: A dictionary containing lists of IDs for 'train_ids', 'val_ids', and 'test_ids'.
        """
        train_ids = open(src_id['train']).read().splitlines()
        val_ids = open(src_id['val']).read().splitlines()
        test_ids = open(src_id['test']).read().splitlines()
        split_ids = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
        return split_ids

    @staticmethod
    def read_captions(src_cap):
        """
        Parses a tab-delimited caption file into a dictionary mapped by image ID.

        Args:
            src_cap (str): Path to the caption source file.

        Returns:
            dict: A dictionary where keys are image IDs and values are lists of (caption_id, text) tuples.
        """
        file = open(src_cap, 'r')
        text = file.read()
        file.close()

        data_dict = defaultdict(list)
        lines = text.split('\n')
        for line in lines:
            if len(line.split('\t')) == 1:
                continue
            ids, img_cap = line.split('\t')
            img_id, cap_id = ids.split('#')
            data_dict[img_id].append((cap_id, img_cap.lower()))
        return data_dict
    
    def build(self):
        """
        Filters captions by split and instantiates CustomDataset objects for training, validation, and testing.

        Returns:
            tuple: (train_data, val_data, test_data) as CustomDataset instances.
        """
        train_ids = self.split_ids['train_ids']
        train_captions = {img_id: self.captions[img_id] for img_id in train_ids}

        val_ids = self.split_ids['val_ids']
        val_captions = {img_id: self.captions[img_id] for img_id in val_ids}

        test_ids = self.split_ids['test_ids']
        test_captions = {img_id: self.captions[img_id] for img_id in test_ids}

        train_data = CustomDataset(
            self.src_img, train_captions, 
            transforms=self.train_transforms, 
            caption_mode=self.caption_mode
        )
        val_data = CustomDataset(
            self.src_img, val_captions, 
            caption_mode="flatten", 
            transforms=self.val_transforms
        )
        test_data = CustomDataset(
            self.src_img, test_captions, 
            caption_mode="flatten", 
            transforms=self.val_transforms
        )
        return train_data, val_data, test_data
