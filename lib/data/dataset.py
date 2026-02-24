import os
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.io import decode_image


def caption_selection(captions, mode):
    """
    Processes a caption dictionary into a list format based on the specified selection mode.

    Args:
        captions (dict): Dictionary where keys are image IDs and values are lists of (cap_id, text).
        mode (str): Selection strategy ('random', 'augmentation', 'flatten', 'dictionary', or 'fixed').

    Returns:
        list|dict: Formatted captions, typically a list of [img_id, caption(s)] or the original dict.
    """
    if mode == 'random':
        new_captions = []
        for img_id, caps in captions.items():
            sample_idx = torch.randint(len(caps), size=(1,)).item()
            new_captions.append([img_id, caps[sample_idx][1]])
        return new_captions
    elif mode == 'augmentation':
        new_captions = []
        for img_id, caps in captions.items():
            for cap in caps:
                new_captions.append([img_id, cap[1]])
        return new_captions
    elif mode == 'flatten':
        new_captions = []
        for img_id, caps in captions.items():
            flat_caps = []
            for cap in caps:
                flat_caps.append(cap[1])
            new_captions.append([img_id, *flat_caps])
        return new_captions
    elif mode == 'dictionary':
        return captions
    elif mode == 'fixed':
        new_captions = []
        for img_id, caps in captions.items():
            new_captions.append([img_id, caps[0][1]])
        return new_captions
    else:
        raise NotImplementedError(f'{mode} is not implemented. \
                         Please set mode to: "random", "augmentation", "flatten", "fixed",\
                          or "dictionary".')


class CustomDataset(Dataset):
    """
    A PyTorch Dataset for loading images and their associated captions.

    Args:
        src_img (str): Directory path containing the image files.
        captions (dict): Dictionary of image IDs and their corresponding captions.
        transforms (callable, optional): Transformations to apply to the images.
        caption_mode (str): Strategy for selecting captions (default: 'augmentation').
    """
    def __init__(self, src_img, captions, transforms=None, caption_mode='augmentation'):
        super().__init__()
        self.captions = caption_selection(captions, caption_mode)
        self.src_img = src_img
        self.transforms = transforms
        self.caption_mode = caption_mode

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of caption entries.
        """
        return len(self.captions)
    
    def __getitem__(self, index):
        """
        Fetches the image and its associated caption(s) at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: (img, caption) where img is the processed image and caption is a list of strings.
        """
        img_path = os.path.join(self.src_img, self.captions[index][0])
        img = decode_image(img_path)
        if self.transforms:
            img = self.transforms(img)
        caption = self.captions[index][1:]
        return img, caption
    