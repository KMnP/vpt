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

"""Implements the DSprites data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry


# These constants specify the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the data set is used
# as a new training split and VAL_SPLIT_PERCENT% is used for validation.
# The rest is used for testing.
TRAIN_SPLIT_PERCENT = 80
VAL_SPLIT_PERCENT = 10


@Registry.register("data.dsprites", "class")
class DSpritesData(base.ImageTfdsData):
  """Provides the DSprites data set.

  DSprites only comes with a training set. Therefore, the training, validation,
  and test set are split out of the original training set.

  For additional details and usage, see the base class.

  The data set page is https://github.com/deepmind/dsprites-dataset/.
  """

  def __init__(self, predicted_attribute, num_classes=None, data_dir=None):
    dataset_builder = tfds.builder("dsprites:2.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()
    info = dataset_builder.info

    if predicted_attribute not in dataset_builder.info.features:
      raise ValueError(
          "{} is not a valid attribute to predict.".format(predicted_attribute))

    # If num_classes is set, we group together nearby integer values to arrive
    # at the desired number of classes. This is useful for example for grouping
    # together different spatial positions.
    num_original_classes = info.features[predicted_attribute].num_classes
    if num_classes is None:
      num_classes = num_original_classes
    if not isinstance(num_classes, int) or num_classes <= 1 or (
        num_classes > num_original_classes):
      raise ValueError(
          "The number of classes should be None or in [2, ..., num_classes].")
    class_division_factor = float(num_original_classes) / num_classes

    # Creates a dict with example counts for each split.
    num_total = dataset_builder.info.splits["train"].num_examples
    num_samples_train = TRAIN_SPLIT_PERCENT * num_total // 100
    num_samples_val = VAL_SPLIT_PERCENT * num_total // 100
    num_samples_splits = {
        "train": num_samples_train,
        "val": num_samples_val,
        "trainval": num_samples_val + num_samples_train,
        "test": num_total - num_samples_val - num_samples_train,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "train": "train[:{}]".format(num_samples_splits["train"]),
        "val": "train[{}:{}]".format(num_samples_splits["train"],
                                     num_samples_splits["trainval"]),
        "trainval": "train[:{}]".format(num_samples_splits["trainval"]),
        "test": "train[{}:]".format(num_samples_splits["trainval"]),
        "train800": "train[:800]",
        "val200": "train[{}:{}]".format(num_samples_splits["train"],
                                        num_samples_splits["train"]+200),
        "train800val200": "train[:800]+train[{}:{}]".format(
            num_samples_splits["train"], num_samples_splits["train"]+200),
    }

    def preprocess_fn(tensors):
      # For consistency with other datasets, image needs to have three channels
      # and be in [0, 255).
      images = tf.tile(tensors["image"], [1, 1, 3]) * 255
      label = tf.cast(
          tf.math.floordiv(
              tf.cast(tensors[predicted_attribute], tf.float32),
              class_division_factor), info.features[predicted_attribute].dtype)
      return dict(image=images, label=label)

    super(DSpritesData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # We extract the attribute we want to predict in the preprocessing.
        base_preprocess_fn=preprocess_fn,
        num_classes=num_classes)
