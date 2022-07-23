# Visual Prompt Tuning 

https://arxiv.org/abs/2203.12119 

------

This repository contains the official PyTorch implementation for Visual Prompt Tuning.

![vpt_teaser](https://github.com/KMnP/vpt/blob/main/imgs/teaser.png)

## Environment settings

See `env_setup.sh`

## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.
  
  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

    * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VPT. This folder should contain the same file names as those in  `vit_backbones`

    * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT, Swin and ViT with mae, moco-v3

    * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  
- `src/utils`: helper functions for io, loggings, training, visualizations. 
- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job.

## Experiments

### Key configs:

- 🔥VPT related:
  - MODEL.PROMPT.NUM_TOKENS: prompt length
  - MODEL.PROMPT.DEEP: deep or shallow prompt
- Fine-tuning method specification:
  - MODEL.TRANSFER_TYPE
- Vision backbones:
  - DATA.FEATURE: specify which representation to use
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - RUN_N_TIMES: ensure only run once in case for duplicated submision, not used during vtab runs
  - OUTPUT_DIR: output dir of the final model and logs
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set

### Datasets preperation:

See Table 8 in the Appendix for dataset details. 

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Dropbox](https://cornell.box.com/v/vptfgvcsplits), [Google Drive](https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM?usp=sharing). 

  - [CUB200 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- [Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/) (VTAB): see [`VTAB_SETUP.md`](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for detailed instructions and tips.

### Pre-trained model preperation

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT` (ConvNeXt-Base and ResNet50 would be automatically downloaded via the links in the code). Note that you also need to rename the downloaded ViT-B/16 ckpt from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.

See Table 9 in the Appendix for more details about pre-trained backbones.


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">link</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
<tr><td align="left">ConvNeXt-Base</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://pytorch.org/vision/stable/models.html">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>

### Examples for training and aggregating results

See [`demo.ipynb`](https://github.com/KMnP/vpt/blob/main/demo.ipynb) for how to use this repo.

### Hyperparameters for experiments in paper

The hyperparameter values used (prompt length for VPT / reduction rate for Adapters, base learning rate, weight decay values) in Table 1-2, Fig. 3-4, Table 4-5 can be found here: [Dropbox](https://cornell.box.com/s/lv10kptgyrm8uxb6v6ctugrhao24rs2z) / [Google Drive](https://drive.google.com/drive/folders/1ldhqkXelHDXq4bG7qpKn5YEfU6sRehJH?usp=sharing). 

## Citation

If you find our work helpful in your research, please cite it as:

```
@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.
