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

"""Implements the SmallNORB data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry
# This constant specifies the percentage of data that is used to create custom
# val/test splits. Specifically, VAL_SPLIT_PERCENT% of the official testing
# split is used as a new validation split and the rest is used for testing.
VAL_SPLIT_PERCENT = 50


@Registry.register("data.smallnorb", "class")
class SmallNORBData(base.ImageTfdsData):
  """Provides the SmallNORB data set.

  SmallNORB comes only with a training and test set. Therefore, the validation
  set is split out of the original training set, and the remaining examples are
  used as the "train" split. The "trainval" split corresponds to the original
  training set.

  For additional details and usage, see the base class.

  The data set page is https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/.
  """

  def __init__(self, predicted_attribute, data_dir=None):
    dataset_builder = tfds.builder("smallnorb:2.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    if predicted_attribute not in dataset_builder.info.features:
      raise ValueError(
          "{} is not a valid attribute to predict.".format(predicted_attribute))

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "train": "train",
        "val": "test[:{}%]".format(VAL_SPLIT_PERCENT),
        "trainval": "train+test[:{}%]".format(VAL_SPLIT_PERCENT),
        "test": "test[{}%:]".format(VAL_SPLIT_PERCENT),
        "train800": "train[:800]",
        "val200": "test[:200]",
        "train800val200": "train[:800]+test[:200]",
    }

    # Creates a dict with example counts for each split.
    train_count = dataset_builder.info.splits["train"].num_examples
    test_count = dataset_builder.info.splits["test"].num_examples
    num_samples_validation = VAL_SPLIT_PERCENT * test_count // 100
    num_samples_splits = {
        "train": train_count,
        "val": num_samples_validation,
        "trainval": train_count + num_samples_validation,
        "test": test_count - num_samples_validation,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    def preprocess_fn(tensors):
      # For consistency with other datasets, image needs to have three channels.
      image = tf.tile(tensors["image"], [1, 1, 3])
      return dict(image=image, label=tensors[predicted_attribute])

    info = dataset_builder.info
    super(SmallNORBData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # We extract the attribute we want to predict in the preprocessing.
        base_preprocess_fn=preprocess_fn,
        num_classes=info.features[predicted_attribute].num_classes)
