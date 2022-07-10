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

"""Implements Svhn data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry
# This constant specifies the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the official training
# split is used as a new training split and the rest is used for validation.
TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.svhn", "class")
class SvhnData(base.ImageTfdsData):
  """Provides SVHN data.

  The Street View House Numbers (SVHN) Dataset is an image digit recognition
  dataset of over 600,000 color digit images coming from real world data.
  Split size:
    - Training set: 73,257 images
    - Testing set: 26,032 images
    - Extra training set: 531,131 images
  Following the common setup on SVHN, we only use the official training and
  testing data. Images are cropped to 32x32.

  URL: http://ufldl.stanford.edu/housenumbers/
  """

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("svhn_cropped:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Example counts are retrieved from the tensorflow dataset info.
    trainval_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        # Calculates the train/val split example count based on percent.
        "train": TRAIN_SPLIT_PERCENT * trainval_count // 100,
        "val": trainval_count - TRAIN_SPLIT_PERCENT * trainval_count // 100,
        "trainval": trainval_count,
        "test": test_count,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    # Defines dataset specific train/val/trainval/test splits.
    # The validation set is split out of the original training set, and the
    # remaining examples are used as the "train" split. The "trainval" split
    # corresponds to the original training set.
    tfds_splits = {
        "train":
            "train[:{}]".format(num_samples_splits["train"]),
        "val":
            "train[{}:]".format(num_samples_splits["train"]),
        "trainval":
            "train",
        "test":
            "test",
        "train800":
            "train[:800]",
        "val200":
            "train[{}:{}]".format(num_samples_splits["train"],
                                  num_samples_splits["train"] + 200),
        "train800val200":
            "train[:800]+train[{}:{}]".format(
                num_samples_splits["train"], num_samples_splits["train"] + 200),
    }

    super(SvhnData, self).__init__(
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
        num_classes=dataset_builder.info.features["label"]
        .num_classes)
