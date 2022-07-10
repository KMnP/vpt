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

"""Global Registry for the task adaptation framework.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import functools


def partialclass(cls, *base_args, **base_kwargs):
  """Builds a subclass with partial application of the given args and keywords.

  Equivalent to functools.partial performance, base_args are preprended to the
  positional arguments given during object initialization and base_kwargs are
  updated with the kwargs given later.

  Args:
    cls: The base class.
    *base_args: Positional arguments to be applied to the subclass.
    **base_kwargs: Keyword arguments to be applied to the subclass.

  Returns:
    A subclass of the input class.
  """

  class _NewClass(cls):

    def __init__(self, *args, **kwargs):
      bound_args = base_args + args
      bound_kwargs = base_kwargs.copy()
      bound_kwargs.update(kwargs)
      super(_NewClass, self).__init__(*bound_args, **bound_kwargs)

  return _NewClass


def parse_name(string_to_parse):
  """Parses input to the registry's lookup function.

  Args:
    string_to_parse: can be either an arbitrary name or function call
      (optionally with positional and keyword arguments).
      e.g. "multiclass", "resnet50_v2(filters_factor=8)".

  Returns:
    A tuple of input name and a dctinary with arguments. Examples:
      "multiclass" -> ("multiclass", (), {})
      "resnet50_v2(9, filters_factor=4)" ->
          ("resnet50_v2", (9,), {"filters_factor": 4})
  """
  expr = ast.parse(string_to_parse, mode="eval").body  # pytype: disable=attribute-error
  if not isinstance(expr, (ast.Attribute, ast.Call, ast.Name)):
    raise ValueError(
        "The given string should be a name or a call, but a {} was parsed from "
        "the string {!r}".format(type(expr), string_to_parse))

  # Notes:
  # name="some_name" -> type(expr) = ast.Name
  # name="module.some_name" -> type(expr) = ast.Attribute
  # name="some_name()" -> type(expr) = ast.Call
  # name="module.some_name()" -> type(expr) = ast.Call

  if isinstance(expr, ast.Name):
    return string_to_parse, {}
  elif isinstance(expr, ast.Attribute):
    return string_to_parse, {}

  def _get_func_name(expr):
    if isinstance(expr, ast.Attribute):
      return _get_func_name(expr.value) + "." + expr.attr
    elif isinstance(expr, ast.Name):
      return expr.id
    else:
      raise ValueError(
          "Type {!r} is not supported in a function name, the string to parse "
          "was {!r}".format(type(expr), string_to_parse))

  def _get_func_args_and_kwargs(call):
    args = tuple([ast.literal_eval(arg) for arg in call.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
    }
    return args, kwargs

  func_name = _get_func_name(expr.func)
  func_args, func_kwargs = _get_func_args_and_kwargs(expr)
  if func_args:
    raise ValueError("Positional arguments are not supported here, but these "
                     "were found: {!r}".format(func_args))

  return func_name, func_kwargs


class Registry(object):
  """Implements global Registry."""

  _GLOBAL_REGISTRY = {}

  @staticmethod
  def global_registry():
    return Registry._GLOBAL_REGISTRY

  @staticmethod
  def register(name, item_type):
    """Creates a function that registers its input."""
    if item_type not in ["function", "class"]:
      raise ValueError("Unknown item type: %s" % item_type)

    def _register(item):
      if name in Registry.global_registry():
        raise KeyError(
            "The name {!r} was already registered in with type {!r}".format(
                name, item_type))

      Registry.global_registry()[name] = (item, item_type)
      return item

    return _register

  @staticmethod
  def lookup(lookup_string, kwargs_extra=None):
    """Lookup a name in the registry."""

    name, kwargs = parse_name(lookup_string)
    if kwargs_extra:
      kwargs.update(kwargs_extra)
    item, item_type = Registry.global_registry()[name]
    if item_type == "function":
      return functools.partial(item, **kwargs)
    elif item_type == "class":
      return partialclass(item, **kwargs)
