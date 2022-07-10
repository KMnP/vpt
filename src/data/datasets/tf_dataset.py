#!/usr/bin/env python3

"""a dataset that handles output of tf.data: support datasets from VTAB"""
import functools
import tensorflow.compat.v1 as tf
import torch
import torch.utils.data
import numpy as np

from collections import Counter
from torch import Tensor

from ..vtab_datasets import base
# pylint: disable=unused-import
from ..vtab_datasets import caltech
from ..vtab_datasets import cifar
from ..vtab_datasets import clevr
from ..vtab_datasets import diabetic_retinopathy
from ..vtab_datasets import dmlab
from ..vtab_datasets import dsprites
from ..vtab_datasets import dtd
from ..vtab_datasets import eurosat
from ..vtab_datasets import kitti
from ..vtab_datasets import oxford_flowers102
from ..vtab_datasets import oxford_iiit_pet
from ..vtab_datasets import patch_camelyon
from ..vtab_datasets import resisc45
from ..vtab_datasets import smallnorb
from ..vtab_datasets import sun397
from ..vtab_datasets import svhn
from ..vtab_datasets.registry import Registry

from ...utils import logging
logger = logging.get_logger("visual_prompt")
tf.config.experimental.set_visible_devices([], 'GPU')  # set tensorflow to not use gpu  # noqa
DATASETS = [
    'caltech101',
    'cifar(num_classes=100)',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'patch_camelyon',
    'sun397',
    'svhn',
    'resisc45',
    'eurosat',
    'dmlab',
    'kitti(task="closest_vehicle_distance")',
    'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")',
    'clevr(task="count_all")',
    'diabetic_retinopathy(config="btgraham-300")'
]


class TFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
            "trainval"
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME

        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.get_data(cfg, split)

    def get_data(self, cfg, split):
        tf_data = build_tf_dataset(cfg, split)
        data_list = list(tf_data)  # a list of tuples

        self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
        self._targets = [int(t[1].numpy()[0]) for t in data_list]
        self._class_ids = sorted(list(set(self._targets)))

        logger.info("Number of images: {}".format(len(self._image_tensor_list)))
        logger.info("Number of classes: {} / {}".format(
            len(self._class_ids), self.get_class_num()))

        del data_list
        del tf_data

    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

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
        label = self._targets[index]
        im = to_torch_imgs(
            self._image_tensor_list[index], self.img_mean, self.img_std)

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
        return len(self._targets)


def preprocess_fn(data, size=224, input_range=(0.0, 1.0)):
    image = data["image"]
    image = tf.image.resize(image, [size, size])

    image = tf.cast(image, tf.float32) / 255.0
    image = image * (input_range[1] - input_range[0]) + input_range[0]

    data["image"] = image
    return data


def build_tf_dataset(cfg, mode):
    """
    Builds a tf data instance, then transform to a list of tensors and labels
    """

    if mode not in ["train", "val", "test", "trainval"]:
        raise ValueError("The input pipeline supports `train`, `val`, `test`."
                         "Provided mode is {}".format(mode))

    vtab_dataname = cfg.DATA.NAME.split("vtab-")[-1]
    data_dir = cfg.DATA.DATAPATH
    if vtab_dataname in DATASETS:
        data_cls = Registry.lookup("data." + vtab_dataname)
        vtab_tf_dataloader = data_cls(data_dir=data_dir)
    else:
        raise ValueError("Unknown type for \"dataset\" field: {}".format(
            type(vtab_dataname)))

    split_name_dict = {
        "dataset_train_split_name": "train800",
        "dataset_val_split_name": "val200",
        "dataset_trainval_split_name": "train800val200",
        "dataset_test_split_name": "test",
    }

    def _dict_to_tuple(batch):
        return batch['image'], batch['label']

    return vtab_tf_dataloader.get_tf_data(
        batch_size=1,  # data_params["batch_size"],
        drop_remainder=False,
        split_name=split_name_dict[f"dataset_{mode}_split_name"],
        preprocess_fn=functools.partial(
            preprocess_fn,
            input_range=(0.0, 1.0),
            size=cfg.DATA.CROPSIZE,
            ),
        for_eval=mode != "train",  # handles shuffling
        shuffle_buffer_size=1000,
        prefetch=1,
        train_examples=None,
        epochs=1  # setting epochs to 1 make sure it returns one copy of the dataset
    ).map(_dict_to_tuple)  # return a PrefetchDataset object. (which does not have much documentation to go on)


def to_torch_imgs(img: np.ndarray, mean: Tensor, std: Tensor) -> Tensor:
    t_img: Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    t_img -= mean
    t_img /= std

    return t_img
