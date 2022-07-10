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

"""Abstract class for reading the data using tfds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def make_get_tensors_fn(output_tensors):
  """Create a function that outputs a collection of tensors from the dataset."""

  def _get_fn(data):
    """Get tensors by name."""
    return {tensor_name: data[tensor_name] for tensor_name in output_tensors}

  return _get_fn


def make_get_and_cast_tensors_fn(output_tensors):
  """Create a function that gets and casts a set of tensors from the dataset.

  Optionally, you can also rename the tensors.

  Examples:
    # This simply gets "image" and "label" tensors without any casting.
    # Note that this is equivalent to make_get_tensors_fn(["image", "label"]).
    make_get_and_cast_tensors_fn({
      "image": None,
      "label": None,
    })

    # This gets the "image" tensor without any type conversion, casts the
    # "heatmap" tensor to tf.float32, and renames the tensor "class/label" to
    # "label" and casts it to tf.int64.
    make_get_and_cast_tensors_fn({
      "image": None,
      "heatmap": tf.float32,
      "class/label": ("label", tf.int64),
    })

  Args:
    output_tensors: dictionary specifying the set of tensors to get and cast
      from the dataset.

  Returns:
    The function performing the operation.
  """

  def _tensors_to_cast():
    tensors_to_cast = []  # AutoGraph does not support generators.
    for tensor_name, tensor_dtype in output_tensors.items():
      if isinstance(tensor_dtype, tuple) and len(tensor_dtype) == 2:
        tensors_to_cast.append((tensor_name, tensor_dtype[0], tensor_dtype[1]))
      elif tensor_dtype is None or isinstance(tensor_dtype, tf.dtypes.DType):
        tensors_to_cast.append((tensor_name, tensor_name, tensor_dtype))
      else:
        raise ValueError('Values of the output_tensors dictionary must be '
                         'None, tf.dtypes.DType or 2-tuples.')
    return tensors_to_cast

  def _get_and_cast_fn(data):
    """Get and cast tensors by name, optionally changing the name too."""

    return {
        new_name:
        data[name] if new_dtype is None else tf.cast(data[name], new_dtype)
        for name, new_name, new_dtype in _tensors_to_cast()
    }

  return _get_and_cast_fn


def compose_preprocess_fn(*functions):
  """Compose two or more preprocessing functions.

  Args:
    *functions: Sequence of preprocess functions to compose.

  Returns:
    The composed function.
  """

  def _composed_fn(x):
    for fn in functions:
      if fn is not None:  # Note: If one function is None, equiv. to identity.
        x = fn(x)
    return x

  return _composed_fn


# Note: DO NOT implement any method in this abstract class.
@six.add_metaclass(abc.ABCMeta)
class ImageDataInterface(object):
  """Interface to the image data classes."""

  @property
  @abc.abstractmethod
  def default_label_key(self):
    """Returns the default label key of the dataset."""

  @property
  @abc.abstractmethod
  def label_keys(self):
    """Returns a tuple with the available label keys of the dataset."""

  @property
  @abc.abstractmethod
  def num_channels(self):
    """Returns the number of channels of the images in the dataset."""

  @property
  @abc.abstractmethod
  def splits(self):
    """Returns the splits defined in the dataset."""

  @abc.abstractmethod
  def get_num_samples(self, split_name):
    """Returns the number of images in the given split name."""

  @abc.abstractmethod
  def get_num_classes(self, label_key=None):
    """Returns the number of classes of the given label_key."""

  @abc.abstractmethod
  def get_tf_data(self,
                  split_name,
                  batch_size,
                  pairwise_mix_fn=None,
                  preprocess_fn=None,
                  preprocess_before_filter=None,
                  epochs=None,
                  drop_remainder=True,
                  for_eval=False,
                  shuffle_buffer_size=None,
                  prefetch=1,
                  train_examples=None,
                  filtered_num_samples=None,
                  filter_fn=None,
                  batch_preprocess_fn=None,
                  ignore_errors=False,
                  shuffle_files=False):
    """Provides preprocessed and batched data.

    Args:
      split_name: name of a data split to provide. Can be "train", "val",
          "trainval" or "test".
      batch_size: batch size.
      pairwise_mix_fn: a function for mixing each data with another random one.
      preprocess_fn: a function for preprocessing input data. It expects a
          dictionary with a key "image" associated with a 3D image tensor.
      preprocess_before_filter: a function for preprocessing input data,
          before filter_fn. It is only designed for light preprocessing,
          i.e. augment with image id. For heavy preprocessing, it's more
          efficient to do it after filter_fn.
      epochs: number of full passes through the data. If None, the data is
          provided indefinitely.
      drop_remainder: if True, the last incomplete batch of data is dropped.
          Normally, this parameter should be True, otherwise it leads to
          the unknown batch dimension, which is not compatible with training
          or evaluation on TPUs.
      for_eval: get data for evaluation. Disables shuffling.
      shuffle_buffer_size: overrides default shuffle buffer size.
      prefetch: number of batches to prefetch.
      train_examples: optional number of examples to take for training.
        If greater than available number of examples, equivalent to None (all).
        Ignored with for_eval is True.
      filtered_num_samples: required when filter_fn is set, number of
        samples after applying filter_fn.
      filter_fn: filter function for generating training subset.
      batch_preprocess_fn: optional function for preprocessing a full batch of
        input data. Analoguous to preprocess_fn with an extra batch-dimension
        on all tensors.
      ignore_errors: whether to skip images that encountered an error in
        decoding *or pre-processing*, the latter is why it is False by default.
      shuffle_files: whether to shuffle the dataset files or not.

    Returns:
      A tf.data.Dataset object as a dictionary containing the output tensors.
    """


class ImageData(ImageDataInterface):
  """Abstract data provider class.

  IMPORTANT: You should use ImageTfdsData below whenever is posible. We want
  to use as many datasets in TFDS as possible to ensure reproducibility of our
  experiments. Your data class should only inherit directly from this if you
  are doing experiments while creating a TFDS dataset.
  """

  @abc.abstractmethod
  def __init__(self,
               num_samples_splits,
               shuffle_buffer_size,
               num_preprocessing_threads,
               num_classes,
               default_label_key='label',
               base_preprocess_fn=None,
               filter_fn=None,
               image_decoder=None,
               num_channels=3):
    """Initializer for the base ImageData class.

    Args:
      num_samples_splits: a dictionary, that maps splits ("train", "trainval",
          "val", and "test") to the corresponding number of samples.
      shuffle_buffer_size: size of a buffer used for shuffling.
      num_preprocessing_threads: the number of parallel threads for data
          preprocessing.
      num_classes: int/dict, number of classes in this dataset for the
        `default_label_key` tensor, or dictionary with the number of classes in
        each label tensor.
      default_label_key: optional, string with the name of the tensor to use
        as label. Default is "label".
      base_preprocess_fn: optional, base preprocess function to apply in all
        cases for this dataset.
      filter_fn: optional, function to filter the examples to use in the
        dataset. DEPRECATED, soon to be removed.
      image_decoder: a function to decode image.
      num_channels: number of channels in the dataset image.
    """
    self._log_warning_if_direct_inheritance()
    self._num_samples_splits = num_samples_splits
    self._shuffle_buffer_size = shuffle_buffer_size
    self._num_preprocessing_threads = num_preprocessing_threads
    self._base_preprocess_fn = base_preprocess_fn
    self._default_label_key = default_label_key
    self._filter_fn = filter_fn
    if self._filter_fn:
      tf.logging.warning('Using deprecated filtering mechanism.')
    self._image_decoder = image_decoder
    self._num_channels = num_channels

    if isinstance(num_classes, dict):
      self._num_classes = num_classes
      if default_label_key not in num_classes:
        raise ValueError(
            'No num_classes was specified for the default_label_key %r' %
            default_label_key)
    elif isinstance(num_classes, int):
      self._num_classes = {default_label_key: num_classes}
    else:
      raise ValueError(
          '"num_classes" must be a int or a dict, but type %r was given' %
          type(num_classes))

  @property
  def default_label_key(self):
    return self._default_label_key

  @property
  def label_keys(self):
    return tuple(self._num_classes.keys())

  @property
  def num_channels(self):
    return self._num_channels

  @property
  def splits(self):
    return tuple(self._num_samples_splits.keys())

  def get_num_samples(self, split_name):
    return self._num_samples_splits[split_name]

  def get_num_classes(self, label_key=None):
    if label_key is None:
      label_key = self._default_label_key
    return self._num_classes[label_key]

  def get_version(self):
    return NotImplementedError('Version is not supported outside TFDS.')

  def get_tf_data(self,
                  split_name,
                  batch_size,
                  pairwise_mix_fn=None,
                  preprocess_fn=None,
                  preprocess_before_filter=None,
                  epochs=None,
                  drop_remainder=True,
                  for_eval=False,
                  shuffle_buffer_size=None,
                  prefetch=1,
                  train_examples=None,
                  filtered_num_samples=None,
                  filter_fn=None,
                  batch_preprocess_fn=None,
                  ignore_errors=False,
                  shuffle_files=False):
    # Obtains tf.data object.
    # We shuffle later when not for eval, it's important to not shuffle before
    # a subset of data is retrieved.
    data = self._get_dataset_split(
        split_name=split_name,
        shuffle_files=shuffle_files)

    if preprocess_before_filter is not None:
      data = preprocess_before_filter(data)


    if self._filter_fn and (filter_fn is None):
      filter_fn = self._filter_fn

    # Dataset filtering priority: (1) filter_fn; (2) train_examples.
    if filter_fn and train_examples:
      raise ValueError('You must not set both filter_fn and train_examples.')

    if filter_fn:
      tf.logging.warning(
          'You are filtering the dataset. Notice that this may hurt your '
          'throughput, since examples still need to be decoded, and may '
          'make the result of get_num_samples() inacurate. '
          'train_examples is ignored for filtering, but only used for '
          'calculating training steps.')
      data = data.filter(filter_fn)
      num_samples = filtered_num_samples
      assert num_samples is not None, (
          'You must set filtered_num_samples if filter_fn is set.')

    elif not for_eval and train_examples:
      # Deterministic for same dataset version.
      data = data.take(train_examples)
      num_samples = train_examples

    else:
      num_samples = self.get_num_samples(split_name)

    data = self._cache_data_if_possible(
        data, split_name=split_name, num_samples=num_samples, for_eval=for_eval)

    def print_filtered_subset(ex):
      """Print filtered subset for debug purpose."""
      if isinstance(ex, dict) and 'id' in ex and 'label' in ex:
        print_op = tf.print(
            'filtered_example:',
            ex['id'],
            ex['label'],
            output_stream=tf.logging.error)
        with tf.control_dependencies([print_op]):
          ex['id'] = tf.identity(ex['id'])
      return ex
    if not for_eval and filter_fn:
      data = data.map(print_filtered_subset)

    # Repeats data `epochs` time or indefinitely if `epochs` is None.
    if epochs is None or epochs > 1:
      data = data.repeat(epochs)

    shuffle_buffer_size = shuffle_buffer_size or self._shuffle_buffer_size
    if not for_eval and shuffle_buffer_size > 1:
      data = data.shuffle(shuffle_buffer_size)

    data = self._preprocess_and_batch_data(data, batch_size, drop_remainder,
                                           pairwise_mix_fn, preprocess_fn,
                                           ignore_errors)

    if batch_preprocess_fn is not None:
      data = data.map(batch_preprocess_fn, self._num_preprocessing_threads)

    if prefetch != 0:
      data = data.prefetch(prefetch)

    return data

  @abc.abstractmethod
  def _get_dataset_split(self, split_name, shuffle_files=False):
    """Return the Dataset object for the given split name.

    Args:
      split_name: Name of the dataset split to get.
      shuffle_files: Whether or not to shuffle files in the dataset.

    Returns:
      A tf.data.Dataset object containing the data for the given split.
    """

  def _log_warning_if_direct_inheritance(self):
    tf.logging.warning(
        'You are directly inheriting from ImageData. Please, consider porting '
        'your dataset to TFDS (go/tfds) and inheriting from ImageTfdsData '
        'instead.')

  def _preprocess_and_batch_data(self,
                                 data,
                                 batch_size,
                                 drop_remainder=True,
                                 pairwise_mix_fn=None,
                                 preprocess_fn=None,
                                 ignore_errors=False):
    """Preprocesses and batches a given tf.Dataset."""
    # Preprocess with basic preprocess functions (e.g. decoding images, parsing
    # features etc.).
    base_preprocess_fn = compose_preprocess_fn(self._image_decoder,
                                               self._base_preprocess_fn)
    # Note: `map_and_batch` is deprecated, and at least when nothing happens
    # in-between, automatically gets merged for efficiency. Same below.
    data = data.map(base_preprocess_fn, self._num_preprocessing_threads)

    # Mix images pair-wise before other element-wise preprocessing.
    # Note: The pairing is implemented by shifting `data` by 1, so the last
    # element of `data` will be dropped.
    if pairwise_mix_fn is not None:
      data = tf.data.Dataset.zip(
          (data, data.skip(1))).map(pairwise_mix_fn,
                                    self._num_preprocessing_threads)

    # Preprocess with customized preprocess functions.
    if preprocess_fn is not None:
      data = data.map(preprocess_fn, self._num_preprocessing_threads)

    if ignore_errors:
      tf.logging.info('Ignoring any image with errors.')
      data = data.apply(tf.data.experimental.ignore_errors())

    return data.batch(batch_size, drop_remainder)

  def _cache_data_if_possible(self, data, split_name, num_samples, for_eval):
    del split_name

    if not for_eval and num_samples <= 150000:
      # Cache the whole dataset if it's smaller than 150K examples.
      data = data.cache()
    return data


class ImageTfdsData(ImageData):
  """Abstract data provider class for datasets available in Tensorflow Datasets.

  To add new datasets inherit from this class. This class implements a simple
  API that is used throughout the project and provides standardized way of data
  preprocessing and batching.
  """

  @abc.abstractmethod
  def __init__(self, dataset_builder, tfds_splits, image_key='image', **kwargs):
    """Initializer for the base ImageData class.

    Args:
      dataset_builder: tfds dataset builder object.
      tfds_splits: a dictionary, that maps splits ("train", "trainval", "val",
          and "test") to the corresponding tfds `Split` objects.
      image_key: image key.
      **kwargs: Additional keyword arguments for the ImageData class.
    """
    self._dataset_builder = dataset_builder
    self._tfds_splits = tfds_splits
    self._image_key = image_key

    # Overwrite image decoder
    def _image_decoder(data):
      decoder = dataset_builder.info.features[image_key].decode_example
      data[image_key] = decoder(data[image_key])
      return data
    self._image_decoder = _image_decoder

    kwargs.update({'image_decoder': _image_decoder})

    super(ImageTfdsData, self).__init__(**kwargs)

  def get_version(self):
    return self._dataset_builder.version.__str__()

  def _get_dataset_split(self, split_name, shuffle_files):
    dummy_decoder = tfds.decode.SkipDecoding()
    return self._dataset_builder.as_dataset(
        split=self._tfds_splits[split_name], shuffle_files=shuffle_files,
        decoders={self._image_key: dummy_decoder})

  def _log_warning_if_direct_inheritance(self):
    pass
