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

"""Implements OxfordIIITPet data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry
# This constant specifies the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the official training
# split is used as a new training split and the rest is used for validation.
TRAIN_SPLIT_PERCENT = 80


@Registry.register("data.oxford_iiit_pet", "class")
class OxfordIIITPetData(base.ImageTfdsData):
  """Provides OxfordIIITPet data.

  The OxfordIIITPet dataset comes only with a training and test set.
  Therefore, the validation set is split out of the original training set, and
  the remaining examples are used as the "train" split. The "trainval" split
  corresponds to the original training set.

  For additional details and usage, see the base class.
  """

  def __init__(self, data_dir=None, train_split_percent=None):

    dataset_builder = tfds.builder("oxford_iiit_pet:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()
    train_split_percent = train_split_percent or TRAIN_SPLIT_PERCENT

    # Creates a dict with example counts for each split.
    trainval_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples
    num_samples_splits = {
        "train": (train_split_percent * trainval_count) // 100,
        "val": trainval_count - (train_split_percent * trainval_count) // 100,
        "trainval": trainval_count,
        "test": test_count,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "train": "train[:{}]".format(num_samples_splits["train"]),
        "val": "train[{}:]".format(num_samples_splits["train"]),
        "trainval": tfds.Split.TRAIN,
        "test": tfds.Split.TEST,
        "train800": "train[:800]",
        "val200": "train[{}:{}]".format(
            num_samples_splits["train"], num_samples_splits["train"]+200),
        "train800val200": "train[:800]+train[{}:{}]".format(
            num_samples_splits["train"], num_samples_splits["train"]+200),
    }

    super(OxfordIIITPetData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
        num_classes=dataset_builder.info.features["label"].num_classes)
