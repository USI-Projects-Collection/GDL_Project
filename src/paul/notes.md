# 4.2 VITS: GRID GRAPHS AND IMAGE DATA

## Model Vision Transformer

1. Possible links
   - https://github.com/google-research/vision_transformer
   - https://github.com/huggingface/pytorch-image-models

## Data

### Imagenet 1k

1. Possible links
   - https://huggingface.co/datasets/ILSVRC/imagenet-1k
   - https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
   - https://www.image-net.org/download.php

2. Size
   - Training images:       1,281,167 
   - Validation images:     50,000
   - Test images:           100,000
   - Raw:                   **167 GB**

### iNaturalist2021

1. Possible links
   - [tensorflow](https://www.tensorflow.org/datasets/catalog/i_naturalist2021)
   - [pytorch](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.INaturalist.html)
   - [github](https://github.com/visipedia/inat_comp/tree/master/2021)

2. Size
   - Training images:         2,686,843
   - Validation images:       100,000
   - Test images:             500,000
   - Raw:                   **224 - 316.54 GB**

### Place365 (Small)

1. Possible links
   - [tensorflow](https://www.tensorflow.org/datasets/catalog/places365_small)

### Summary

|                | Imagenet-1k | iNaturalist2021 | Place365 (Small) |
| -------------- | ----------- | --------------- | ---------------- |
| **Train imgs** | 1,281,167   | 2,686,843       | 1,803,460        |
| **Val imgs**   | 50,000      | 100,000         | 36,500           |
| **Test imgs**  | 100,000     | 500,000         | 328,500          |
| **Raw**        | 167 GB      | 224 - 316.54 GB | 27.85 GB         |


## Challenges

### Datasets

Imagenet-1k and iNatuarlist2021 are quite large. Might lead to problems

### Architecture, hyperparameters and training details

**ImageNet**: Batch size 4096. This might not be feasible for us. Also running it for 90 epochs probably takes an eternity.