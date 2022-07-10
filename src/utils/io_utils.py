#!/usr/bin/env python3
"""
a bunch of helper functions for read and write data
"""
import os
import json
import numpy as np
import time
import pandas as pd

from typing import List, Union
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None


def save_or_append_df(out_path, df):
    if os.path.exists(out_path):
        previous_df = pd.read_pickle(out_path)
        df = pd.concat([previous_df, df], ignore_index=True)
    df.to_pickle(out_path)
    print(f"Saved output at {out_path}")


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # return super(MyEncoder, self).default(obj)

            raise TypeError(
                "Unserializable object {} of type {}".format(obj, type(obj))
            )


def write_json(data: Union[list, dict], outfile: str) -> None:
    json_dir, _ = os.path.split(outfile)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(outfile, 'w') as f:
        json.dump(data, f, cls=JSONEncoder, ensure_ascii=False, indent=2)


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data


def pil_loader(path: str) -> Image.Image:
    """load an image from path, and suppress warning"""
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
