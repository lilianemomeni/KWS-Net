# Seeing Wake Words: Audio-visual Keyword Spotting
This repository contains code for training and evaluating the best performing visual keyword spotting model described in the paper [Liliane Momeni](http://www.robots.ox.ac.uk/~liliane/), [Triantafyllos Afouras](http://www.robots.ox.ac.uk/~afourast/), [Themos Stafylakis](http://github.com/tstafylakis), [Samuel Albanie](http://www.robots.ox.ac.uk/~albanie/) and [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/),
*Seeing Wake Words: Audio-visual Keyword Spotting*, BMVC 2020. Two baseline keyword spotting models are also included.

![alt text](media/teaser/teaser_fig.gif )

[[Project page]](http://www.robots.ox.ac.uk/~vgg/research/kws-net/) 


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

### 1.2. Datasets & Pre-processing

* Download [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) datasets for training; [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) dataset for testing
* Extract talking faces from clips using [metadata](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) available
* Pre-compute features for clips of talking faces using pre-trained [lip reading model](https://github.com/afourast/deep_lip_reading) and save LRS2 features at ```data/lrs2/features/main``` and ```data/lrs2/features/pretrain``` and LRW features at ```data/lrw/features/main```
* Find word-level timings for [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) transcriptions using [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)
* Download [CMU phonetic dictionary](https://github.com/cmusphinx/cmudict): ```data/vocab/cmudict.dict```
* Build CMU phoneme and grapheme vocabulary files: ```data/vocab/phoneme_field_vocab.json``` and ```data/vocab/grapheme_field_vocab.json```
* Build dataset split json files: ```data/lrs2/DsplitsLRS2.json``` and ```data/lrw/DsplitsLRW.json``` using ```misc/xxx``` and ```misc/xxxx``` respectively


### 1.3. Pre-trained models

Download the pre-trained models by running

```
./download_models.sh
```
We provide several pre-trained models used in the paper:

* [Stafylakis & Tzimiropoulos G2P](https://arxiv.org/pdf/1807.08469.pdf) implementation: G2P_baseline.pth
* Stafylakis & Tzimiropoulos P2G, a variant of the above model where the grapheme-to-phoneme keyword encoder-decoder has been switched to a phoneme-to-grapheme architecture: P2G_baseline.pth
* KWS-Net, the novel convolutional architecture we propose: KWS_Net.pth

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
