# Seeing Wake Words: Audio-visual Keyword Spotting

[Liliane Momeni](http://www.robots.ox.ac.uk/~liliane/), [Triantafyllos Afouras](http://www.robots.ox.ac.uk/~afourast/), [Themos Stafylakis](http://github.com/tstafylakis), [Samuel Albanie](http://www.robots.ox.ac.uk/~albanie/) and [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/),
*Seeing Wake Words: Audio-visual Keyword Spotting*, BMVC 2020.

![alt text](media/teaser/teaser_fig.gif )

[[Project page]](http://www.robots.ox.ac.uk/~vgg/research/kws-net/) 

TODO: Add visualisation

## Contents
* [1. Preparation](https://github.com/lilianemomeni/KWS-Net#1-preparation)
* [2. Training](https://github.com/lilianemomeni/KWS-Net#2-training)
* [3. Testing](https://github.com/lilianemomeni/KWS-Net#3-testing)
* [Citation](https://github.com/lilianemomeni/KWS-Net#citation)


## 1. Preparation

### 1.1. Dependencies

#### System 
* ffmpeg

#### Python 
* Torch
* NumPy

###### Optional for visualization
* Matplotlib
* TensorBoard

Install python dependencies by creating a new virtual environment and then running 

```
pip install -r requirements.txt
```

### 1.2. Datasets

The models have been trained on the [LRW and LRS2 datasets](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/) and evaluated on the LRS2 dataset. More details can be found in the paper.

TODO: Add details on preprocessing

### 1.3. Pre-trained models

Download the pre-trained models by running

```
./download_models.sh
```
We provide several pre-trained models used in the paper:

* [Stafylakis & Tzimiropoulos G2P](https://arxiv.org/pdf/1807.08469.pdf) implementation - G2P_baseline.pth
* Stafylakis & Tzimiropoulos P2G - P2G_baseline.pth
* KWS-Net - KWS_Net.pth

The above models are explained in more detail in the [training section](https://github.com/lilianemomeni/KWS-Net#2-training).

TODO: make download_models.sh

## 2. Training

TODO: explain two stages of training

TODO: show expected outputs

## 2. Testing

TODO: make demo

## Citation
If you use this code, please cite the following:
```
@INPROCEEDINGS{momeni20_kwsnet,
  title     = {Seeing Wake Words: Audio-visual Keyword Spotting },
  author    = {Momeni, Liliane and Afouras, Triantafyllos and Stafylakis, Themos and Albanie, Samuel and Zisserman, Andrew},
  booktitle = {BMVC},
  year      = {2020}
}
