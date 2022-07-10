#!/usr/bin/env python3

"""Logging."""

import builtins
import decimal
import functools
import logging
import simplejson
import sys
import os
from termcolor import colored

from .distributed import is_master_process
from .file_io import PathManager

# Show filename and line number in logs
_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"


def _suppress_print():
    """Suppresses printing from the current process."""

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers  # noqa
def setup_logging(
    num_gpu, num_shards, output="", name="visual_prompt", color=True):
    """Sets up the logging."""
    # Enable logging only for the master process
    if is_master_process(num_gpu):
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format=_FORMAT, stream=sys.stdout
        )
    else:
        _suppress_print()

    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    # remove any lingering handler
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(name),
        )
    else:
        formatter = plain_formatter

    if is_master_process(num_gpu):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if is_master_process(num_gpu * num_shards):
        if len(output) > 0:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "logs.txt")

            PathManager.mkdirs(os.path.dirname(filename))

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)
    return logger


def setup_single_logging(name, output=""):
    """Sets up the logging."""
    # Enable logging only for the master process
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if len(name) == 0:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=name,
        abbrev_name=str(name),
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if len(output) > 0:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs.txt")

        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats, sort_keys=True):
    """Logs json stats."""
    # It seems that in Python >= 3.6 json.encoder.FLOAT_REPR has no effect
    # Use decimal+string as a workaround for having fixed length values in logs
    logger = get_logger(__name__)
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    if stats["_type"] == "test_epoch" or stats["_type"] == "train_epoch":
        logger.info("json_stats: {:s}".format(json_stats))
    else:
        logger.info("{:s}".format(json_stats))


class _ColorfulFormatter(logging.Formatter):
    # from detectron2
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
