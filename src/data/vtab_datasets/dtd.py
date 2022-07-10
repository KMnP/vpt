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

"""Implements the Describable Textures Dataset (DTD) data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry


@Registry.register("data.dtd", "class")
class DTDData(base.ImageTfdsData):
  """Provides Describable Textures Dataset (DTD) data.

  As of version 1.0.0, the train/val/test splits correspond to those of the
  1st fold of the official cross-validation partition.

  For additional details and usage, see the base class.
  """

  def __init__(self, data_dir=None):

    dataset_builder = tfds.builder("dtd:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "train": "train",
        "val": "validation",
        "trainval": "train+validation",
        "test": "test",
        "train800": "train[:800]",
        "val200": "validation[:200]",
        "train800val200": "train[:800]+validation[:200]",
    }

    # Creates a dict with example counts for each split.
    train_count = dataset_builder.info.splits["train"].num_examples
    val_count = dataset_builder.info.splits["validation"].num_examples
    test_count = dataset_builder.info.splits["test"].num_examples
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": train_count + val_count,
        "test": test_count,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    super(DTDData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
        num_classes=dataset_builder.info.features["label"].num_classes)
