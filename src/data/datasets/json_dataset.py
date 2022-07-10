#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter

from ..transforms import get_transforms
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("visual_prompt")


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self.data_dir = cfg.DATA.DATAPATH
        self.data_percentage = cfg.DATA.PERCENTAGE
        self._construct_imdb(cfg)
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            # "id": index
        }
        return sample

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        super(CarsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

