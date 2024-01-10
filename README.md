## Context-Aware Adaptive Network for UDA Semantic Segmentation(CAAN)

## Overview
![UDA over time](sources/F1g1.png)
![UDA over time](sources/Fig22222.png)
![UDA over time](sources/F2g.png)

Code was tested on an NVIDIA 3090Ti with 24G Memory.

## Comparison with State-of-the-Art UDA
Our method significantly outperforms previous works on several UDA benchmarks.
This includes synthetic-to-real adaptation on GTA→Cityscapes and
Synthia→Cityscapes.

|              | GTA→CS(val) | Synthia→CS(val) | 
|--------------|-------------|-----------------|
| ADVENT [1]   | 45.5        | 41.2            | 
| BDL [2]      | 48.5        | --              |
| FDA [3]      | 50.5        | --              |
| DACS [4]     | 52.1        | 48.3            |
| ProDA [5]    | 57.5        | 55.5            |
| MGCDA [6]    | --          | --              |
| DANNet [7]   | --          | --              |
| DAFormer [8] | 68.3        | 60.9            |
| Cross [9]    | 69.5        | 57.5            |
| IDM [10]     | 69.5        | 60.9            |
| **Ours**     | **69.8**    | **63.2**        |
References:

1. Vu et al. "Advent: Adversarial entropy minimization for domain adaptation in semantic segmentation" in CVPR 2019.
2. Li et al. "Bidirectional learning for domain adaptation of semantic segmentation" in CVPR 2019.
3. Yang et al. "Fda: Fourier domain adaptation for semantic segmentation" in CVPR 2020.
4. Tranheden et al. "Dacs: Domain adaptation via crossdomain mixed sampling" in WACV 2021.
5. Zhang et al. "Prototypical pseudo label denoising and target structure learning for domain adaptive semantic segmentation" in CVPR 2021.
6. Sakaridis et al. "Map-guided curriculum domain adaptation and uncertaintyaware evaluation for semantic nighttime image segmentation" in TPAMI, 2020.
7. Wu et al. "DANNet: A one-stage domain adaptation network for unsupervised nighttime semantic segmentation" in CVPR, 2021.
8. Hoyer L, Dai D, Van Gool L. Daformer: Improving network architectures and training strategies for domain-adaptive semantic segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 9924-9935.
9. Yin Y, Hu W, Liu Z, et al. CrossMatch: Source-Free Domain Adaptive Semantic Segmentation via Cross-Modal Consistency Training[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 21786-21796.
10. Wang Y, Liang J, Xiao J, et al. Informative Data Mining for One-Shot Cross-Domain Semantic Segmentation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 1064-1074.
## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
conda create -n CAAN 
conda activate CAAN
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT ImageNet weights (b3-b5) provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

All experiments were executed on an NVIDIA 3090Ti with 24G Memory.


## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia :** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.



The final folder structure should look like this:

```none
Main dir
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia (optional)
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

## Training

```shell
Coming soon
```

## Checkpoints

Below, we provide checkpoints of CAAN for different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [CAAN for GTA→Cityscapes](https://drive.google.com/file/d/1AQRr1Z9-rxKad-JCoSctzh7Mbl2NvR-n/view?usp=sharing)
* [CAAN for Synthia→Cityscapes](https://drive.google.com/file/d/1-E1nO95b21rjvm0NLxds_nsnx8SWrSIY/view?usp=sharing)

## Test the trained  model
```shell
sh test.sh path/to/checkpoint_directory
```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
