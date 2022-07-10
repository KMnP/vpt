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

"""Imports the Caltech images dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import base as base
from .registry import Registry
import tensorflow_datasets as tfds


# Percentage of the original training set retained for training, the rest is
# used as a validation set.
_TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.caltech101", "class")
class Caltech101(base.ImageTfdsData):
  """Provides the Caltech101 dataset.

  See the base class for additional details on the class.

  See TFDS dataset for details on the dataset:
  third_party/py/tensorflow_datasets/image/caltech.py

  The original (TFDS) dataset contains only a train and test split. We randomly
  sample _TRAIN_SPLIT_PERCENT% of the train split for our "train" set. The
  remainder of the TFDS train split becomes our "val" set. The full TFDS train
  split is called "trainval". The TFDS test split is used as our test set.

  Note that, in the TFDS dataset, the training split is class-balanced, but not
  the test split. Therefore, a significant difference between performance on the
  "val" and "test" sets should be expected.
  """

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("caltech101:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Creates a dict with example counts for each split.
    trainval_count = dataset_builder.info.splits["train"].num_examples
    train_count = (_TRAIN_SPLIT_PERCENT * trainval_count) // 100
    test_count = dataset_builder.info.splits["test"].num_examples
    num_samples_splits = dict(
        train=train_count,
        val=trainval_count - train_count,
        trainval=trainval_count,
        test=test_count,
        train800=800,
        val200=200,
        train800val200=1000)

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "train": "train[:{}]".format(train_count),
        "val": "train[{}:]".format(train_count),
        "trainval": "train",
        "test": "test",
        "train800": "train[:800]",
        "val200": "train[{}:{}]".format(train_count, train_count+200),
        "train800val200": (
            "train[:800]+train[{}:{}]".format(train_count, train_count+200)),
    }

    super(Caltech101, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=3000,
        base_preprocess_fn=base.make_get_tensors_fn(("image", "label")),
        num_classes=dataset_builder.info.features["label"].num_classes)
