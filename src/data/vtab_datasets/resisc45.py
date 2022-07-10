# coding=utf-8
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements RESISC-45 data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry
TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20


@Registry.register("data.resisc45", "class")
class Resisc45Data(base.ImageTfdsData):
  """Provides RESISC-45 dataset.

  RESISC45 dataset is a publicly available benchmark for Remote Sensing Image
  Scene Classification (RESISC), created by Northwestern Polytechnical
  University (NWPU). This dataset contains 31,500 images, covering 45 scene
  classes with 700 images in each class.

  URL: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
  """

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("resisc45:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Example counts are retrieved from the tensorflow dataset info.
    num_examples = dataset_builder.info.splits["train"].num_examples
    train_count = num_examples * TRAIN_SPLIT_PERCENT // 100
    val_count = num_examples * VALIDATION_SPLIT_PERCENT // 100
    test_count = num_examples * TEST_SPLIT_PERCENT // 100

    tfds_splits = {
        "train":
            "train[:{}]".format(train_count),
        "val":
            "train[{}:{}]".format(train_count, train_count + val_count),
        "trainval":
            "train[:{}]".format(train_count + val_count),
        "test":
            "train[{}:]".format(train_count + val_count),
        "train800":
            "train[:800]",
        "val200":
            "train[{}:{}]".format(train_count, train_count+200),
        "train800val200":
            "train[:800]+train[{}:{}]".format(train_count, train_count+200),
    }

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": train_count + val_count,
        "test": test_count,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    super(Resisc45Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Rename tensors but keep their original types.
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": ("image", None),
            "label": ("label", None),
        }),
        num_classes=dataset_builder.info.features["label"].num_classes)
