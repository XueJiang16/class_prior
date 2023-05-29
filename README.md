# Detecting Out-of-distribution Data through In-distribution Class Prior

This is the source code for our paper:
[Detecting Out-of-distribution Data through In-distribution Class Prior](https://github.com/XueJiang16/class_prior)
by Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han.
Code is modified from [GradNorm](https://github.com/deeplearning-wisc/gradnorm_ood).


## Usage

### 1. Install

```bash
conda create -n class_prior python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scikit-learn
```

### 2. Dataset Preparation

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data (not necessary) and validation data in
`./dataset/id_data/imagenet_train` and  `./dataset/id_data/imagenet_val`, respectively.

The meta file for ImageNet-LT-a8 is in  `./meta`.

#### Out-of-distribution dataset
Following [MOS](https://arxiv.org/pdf/2105.01879.pdf), we use the following 4 OOD datasets for evaluation:
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

Please refer to [MOS](https://github.com/deeplearning-wisc/large_scale_ood), download OOD datasets and put them into `./dataset/ood_data/`.


### 3. Pre-trained Model Preparation

We use mmclassification to train ResNet101 on ImageNet-LT-a8 dataset.

Put the downloaded model in `./checkpoints/`.


### 4. OOD Detection Evaluation

To reproduce our results, please run:
```bash
bash ./run.sh
```

## Citation

If you find our codebase useful, please cite our work:
```
@inproceedings{jiang2023detecting,
        title={Detecting Out-of-distribution Data through In-distribution Class Prior}, 
        author={Xue Jiang and Feng Liu and Zhen Fang and Hong Chen and Tongliang Liu and Feng Zheng and 				Bo Han},
        booktitle = {ICML},
        year = {2023}
}
```
