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

"""Implements CLEVR data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry

TRAIN_SPLIT_PERCENT = 90


def _count_preprocess_fn(x):
  return {"image": x["image"],
          "label": tf.size(x["objects"]["size"]) - 3}


def _count_cylinders_preprocess_fn(x):
  # Class distribution:

  num_cylinders = tf.reduce_sum(
      tf.cast(tf.equal(x["objects"]["shape"], 2), tf.int32))
  return {"image": x["image"], "label": num_cylinders}


def _closest_object_preprocess_fn(x):
  dist = tf.reduce_min(x["objects"]["pixel_coords"][:, 2])
  # These thresholds are uniformly spaced and result in more or less balanced
  # distribution of classes, see the resulting histogram:

  thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
  label = tf.reduce_max(tf.where((thrs - dist) < 0))
  return {"image": x["image"],
          "label": label}


_TASK_DICT = {
    "count_all": {
        "preprocess_fn": _count_preprocess_fn,
        "num_classes": 8
    },
    "count_cylinders": {
        "preprocess_fn": _count_cylinders_preprocess_fn,
        "num_classes": 11
    },
    "closest_object_distance": {
        "preprocess_fn": _closest_object_preprocess_fn,
        "num_classes": 6
    },
}


@Registry.register("data.clevr", "class")
class CLEVRData(base.ImageTfdsData):
  """Provides CLEVR dataset.

  Currently, two tasks are supported:
    1. Predict number of objects.
    2. Predict distnace to the closest object.
  """

  def __init__(self, task, data_dir=None):

    if task not in _TASK_DICT:
      raise ValueError("Unknown task: %s" % task)

    dataset_builder = tfds.builder("clevr:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Creates a dict with example counts for each split.
    trainval_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples
    num_samples_splits = {
        "train": (TRAIN_SPLIT_PERCENT * trainval_count) // 100,
        "val": trainval_count - (TRAIN_SPLIT_PERCENT * trainval_count) // 100,
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
        "trainval": "train",
        "test": "validation",
        "train800": "train[:800]",
        "val200": "train[{}:{}]".format(
            num_samples_splits["train"], num_samples_splits["train"]+200),
        "train800val200": "train[:800]+train[{}:{}]".format(
            num_samples_splits["train"], num_samples_splits["train"]+200),
    }

    task = _TASK_DICT[task]
    base_preprocess_fn = task["preprocess_fn"]

    super(CLEVRData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base_preprocess_fn,
        num_classes=task["num_classes"])
