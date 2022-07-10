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

"""Implements Diabetic Retinopathy data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_addons.image as tfa_image
import tensorflow_datasets as tfds

from . import base as base
from .registry import Registry


@Registry.register("data.diabetic_retinopathy", "class")
class RetinopathyData(base.ImageTfdsData):
  """Provides Diabetic Retinopathy classification data.

  Retinopathy comes only with a training and test set. Therefore, the validation
  set is split out of the original training set, and the remaining examples are
  used as the "train" split. The "trainval" split corresponds to the original
  training set.

  For additional details and usage, see the base class.
  """

  _CONFIGS_WITH_GREY_BACKGROUND = ["btgraham-300"]

  def __init__(self, config="btgraham-300", heavy_train_augmentation=False,
               data_dir=None):
    """Initializer for Diabetic Retinopathy dataset.

    Args:
      config: Name of the TFDS config to use for this dataset.
      heavy_train_augmentation: If True, use heavy data augmentation on the
        training data. Recommended to achieve SOTA.
      data_dir: directory for downloading and storing the data.
    """
    config_and_version = config + ":3.*.*"
    dataset_builder = tfds.builder("diabetic_retinopathy_detection/{}".format(
        config_and_version), data_dir=data_dir)
    self._config = config
    self._heavy_train_augmentation = heavy_train_augmentation

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

    super(RetinopathyData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
        num_classes=dataset_builder.info.features["label"].num_classes)

  @property
  def config(self):
    return self._config

  @property
  def heavy_train_augmentation(self):
    return self._heavy_train_augmentation

  def get_tf_data(self,
                  split_name,
                  batch_size,
                  preprocess_fn=None,
                  for_eval=False,
                  **kwargs):
    if self._heavy_train_augmentation and not for_eval:
      preprocess_fn = base.compose_preprocess_fn(
          self._heavy_train_augmentation, preprocess_fn)

    return super(RetinopathyData, self).get_tf_data(
        split_name=split_name,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        for_eval=for_eval,
        **kwargs)

  def _sample_heavy_data_augmentation_parameters(self):
    # Scale image +/- 10%.
    s = tf.random.uniform(shape=(), minval=-0.1, maxval=0.1)
    # Rotate image [0, 2pi).
    a = tf.random.uniform(shape=(), minval=0.0, maxval=2.0 * 3.1415926535)
    # Vertically shear image +/- 20%.
    b = tf.random.uniform(shape=(), minval=-0.2, maxval=0.2) + a
    # Horizontal and vertial flipping.
    hf = tf.random.shuffle([-1.0, 1.0])[0]
    vf = tf.random.shuffle([-1.0, 1.0])[0]
    # Relative x,y translation.
    dx = tf.random.uniform(shape=(), minval=-0.1, maxval=0.1)
    dy = tf.random.uniform(shape=(), minval=-0.1, maxval=0.1)
    return s, a, b, hf, vf, dx, dy

  def _heavy_data_augmentation_fn(self, example):
    """Perform heavy augmentation on a given input data example.

    This is the same data augmentation as the one done by Ben Graham, the winner
    of the 2015 Kaggle competition. See:
    https://github.com/btgraham/SparseConvNet/blob/a6bdb0c938b3556c1e6c23d5a014db9f404502b9/kaggleDiabetes1.cpp#L12

    Args:
      example: A dictionary containing an "image" key with the image to
        augment.

    Returns:
      The input dictionary with the key "image" containing the augmented image.
    """
    image = example["image"]
    image_shape = tf.shape(image)
    if len(image.get_shape().as_list()) not in [2, 3]:
      raise ValueError(
          "Input image must be a rank-2 or rank-3 tensor, but rank-{} "
          "was given".format(len(image.get_shape().as_list())))
    height = tf.cast(image_shape[0], dtype=tf.float32)
    width = tf.cast(image_shape[1], dtype=tf.float32)
    # Sample data augmentation parameters.
    s, a, b, hf, vf, dx, dy = self._sample_heavy_data_augmentation_parameters()
    # Rotation + scale.
    c00 = (1 + s) * tf.cos(a)
    c01 = (1 + s) * tf.sin(a)
    c10 = (s - 1) * tf.sin(b)
    c11 = (1 - s) * tf.cos(b)
    # Horizontal and vertial flipping.
    c00 = c00 * hf
    c01 = c01 * hf
    c10 = c10 * vf
    c11 = c11 * vf
    # Convert x,y translation to absolute values.
    dx = width * dx
    dy = height * dy
    # Convert affine matrix to TF's transform. Matrix is applied w.r.t. the
    # center of the image.
    cy = height / 2.0
    cx = width / 2.0
    affine_matrix = [[c00, c01, (1.0 - c00) * cx - c01 * cy + dx],
                     [c10, c11, (1.0 - c11) * cy - c10 * cx + dy],
                     [0.0, 0.0, 1.0]]
    affine_matrix = tf.convert_to_tensor(affine_matrix, dtype=tf.float32)
    transform = tfa_image.transform_ops.matrices_to_flat_transforms(
        tf.linalg.inv(affine_matrix))
    if self._config in self._CONFIGS_WITH_GREY_BACKGROUND:
      # Since background is grey in these configs, put in pixels in [-1, 1]
      # range to avoid artifacts from the affine transformation.
      image = tf.cast(image, dtype=tf.float32)
      image = (image / 127.5) - 1.0
    # Apply the affine transformation.
    image = tfa_image.transform(images=image, transforms=transform)
    if self._config in self._CONFIGS_WITH_GREY_BACKGROUND:
      # Put pixels back to [0, 255] range and cast to uint8, since this is what
      # our preprocessing pipeline usually expects.
      image = (1.0 + image) * 127.5
      image = tf.cast(image, dtype=tf.uint8)
    example["image"] = image
    return example
