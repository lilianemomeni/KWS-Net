import argparse
import os
import h5py
import numpy as np
import torch
import torchtext.data as data
import torch.utils.data
import sys
from tqdm import tqdm
import json

def get_CMU_words(CMU_path):
    words = []
    with open(CMU_path) as f:
        lines = f.readlines()
    for wcnt, line in enumerate(lines):
        grapheme, phoneme = line.split(" ",1)
        words.append(grapheme)
    return words

def get_LRW_split(args, split, CMUwords):
    lst_path = args.LRW_words_path
    word_indices = []
    with open(lst_path) as f:
        lines = f.readlines() #list of words
    for word in tqdm(lines):
      word = word.strip()
      widx_array = []
      widx = CMUwords.index(word)    
      widx_array.append(widx)
      for filename in os.listdir(os.path.join(args.LRW_path, word.upper(), split)):
          Fwidx = {}
          L = filename.strip()
          path = os.path.join(word.upper(), split, L).replace(".mp4.npy", "")
          Fwidx['widx']=widx_array
          Fwidx['fn']=path
          word_indices.append(Fwidx)
    return word_indices

def get_LRW_splits():
    parser = argparse.ArgumentParser(description='Script for creating main splits of LRW.')
    parser.add_argument('--CMUdict_path', default='../data/vocab/cmudict.dict')
    parser.add_argument('--LRW_path', default='../data/lrw/features/main/') 
    parser.add_argument('--LRW_words_path', default='../data/lrw/LRWwords.lst')
    args = parser.parse_args()
    CMUwords = get_CMU_words(args.CMUdict_path)    
    S = ['train', 'val', 'test']
    Dsplits = {}
    for i,s in enumerate(S):
        Dsplits[s] = get_LRW_split(args, s, CMUwords)
    with open("../data/lrw/DsplitsLRW.json", "w") as fp:
      json.dump(Dsplits, fp)

if __name__=='__main__':
  get_LRW_splits()
