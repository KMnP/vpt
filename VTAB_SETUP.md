# VTAB Preperation

## Download and prepare

It is recommended to download the data before the experiments, to avoid duplicated effort if submitting experiments for multiple tuning protocols. Here are the collective command to set up the vtab data. 

```python
import tensorflow_datasets as tfds
data_dir = ""  # TODO: setup the data_dir to put the the data to, the DATA.DATAPATH value in config

# caltech101
dataset_builder = tfds.builder("caltech101:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# cifar100
dataset_builder = tfds.builder("cifar100:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# clevr
dataset_builder = tfds.builder("clevr:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dmlab
dataset_builder = tfds.builder("dmlab:2.0.1", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dsprites
dataset_builder = tfds.builder("dsprites:2.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dtd
dataset_builder = tfds.builder("dtd:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# eurosat
subset="rgb"
dataset_name = "eurosat/{}:2.*.*".format(subset)
dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
dataset_builder.download_and_prepare()

# oxford_flowers102
dataset_builder = tfds.builder("oxford_flowers102:2.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# oxford_iiit_pet
dataset_builder = tfds.builder("oxford_iiit_pet:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# patch_camelyon
dataset_builder = tfds.builder("patch_camelyon:2.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# smallnorb
dataset_builder = tfds.builder("smallnorb:2.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# svhn
dataset_builder = tfds.builder("svhn_cropped:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()
```

There are 4 datasets need special care:

```python
# sun397 --> need cv2
# cannot load one image, similar to issue here: https://github.com/tensorflow/datasets/issues/2889
# "Image /t/track/outdoor/sun_aophkoiosslinihb.jpg could not be decoded by Tensorflow.""
# sol: modify the file: "/fsx/menglin/conda/envs/prompt_tf/lib/python3.7/site-packages/tensorflow_datasets/image_classification/sun.py" to ignore those images
dataset_builder = tfds.builder("sun397/tfds:4.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# kitti version is wrong from vtab repo, try 3.2.0 (https://github.com/google-research/task_adaptation/issues/18)
dataset_builder = tfds.builder("kitti:3.2.0", data_dir=data_dir)
dataset_builder.download_and_prepare()


# diabetic_retinopathy
"""
Download this dataset from Kaggle.
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
After downloading, 
- unpack the test.zip file into <data_dir>/manual_dir/.
- unpack the sample.zip to sample/. 
- unpack the sampleSubmissions.csv and trainLabels.csv.

# ==== important! ====
# 1. make sure to check that there are 5 train.zip files instead of 4 (somehow if you chose to download all from kaggle, the train.zip.005 file is missing)
# 2. if unzip train.zip ran into issues, try to use jar xvf train.zip to handle huge zip file
cat test.zip.* > test.zip
cat train.zip.* > train.zip
"""

config_and_version = "btgraham-300" + ":3.*.*"
dataset_builder = tfds.builder("diabetic_retinopathy_detection/{}".format(config_and_version), data_dir=data_dir)
dataset_builder.download_and_prepare()


# resisc45
"""
download/extract dataset artifacts manually: 
Dataset can be downloaded from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
After downloading the rar file, please extract it to the manual_dir.
"""

dataset_builder = tfds.builder("resisc45:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()
```



## Notes

### TFDS version
Note that the experimental results may be different with different API and/or dataset generation code versions. See more from [tfds documentation](https://www.tensorflow.org/datasets/datasets_versioning). Here are what we used for VPT:

```bash
tfds: 4.4.0+nightly

# Natural:
cifar100: 3.0.2
caltech101: 3.0.1
dtd: 3.0.1
oxford_flowers102: 2.1.1
oxford_iiit_pet: 3.2.0
svhn_cropped: 3.0.0
sun397: 4.0.0

# Specialized:
patch_camelyon: 2.0.0
eurosat: 2.0.0
resisc45: 3.0.0
diabetic_retinopathy_detection: 3.0.0


# Structured
clevr: 3.1.0
dmlab: 2.0.1
kitti: 3.2.0
dsprites: 2.0.0
smallnorb: 2.0.0
```

### Train split
As in issue https://github.com/KMnP/vpt/issues/1, we also uploaded the vtab train split info to the vtab data release [Google Drive](https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM)/[Dropbox](https://cornell.app.box.com/v/vptfgvcsplits). In the file `vtab_trainval_splits.json`, for each dataset, you can find the filenames of the randomly selected 1k training examples used in our experiment. We got them by extracting the ‘filename’ attribute from the tensorflow dataset feature dict. Unfortunately, because there’s no such info for [dsprite](https://www.tensorflow.org/datasets/catalog/dsprites), [smallnorb](https://www.tensorflow.org/datasets/catalog/smallnorb) and [svhn](https://www.tensorflow.org/datasets/catalog/svhn_cropped) in the tensorflow dataset format, we cannot provide the splits for these 3 datasets.
