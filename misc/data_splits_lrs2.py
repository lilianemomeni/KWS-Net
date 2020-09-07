import argparse
import glob
import os
import h5py
import numpy as np
import torch
import torchtext.data as data
import torch.utils.data
import json
from tqdm import tqdm

def get_CMU_words(CMU_path):
    words = []
    phonemes = []
    with open(CMU_path) as f:
        lines = f.readlines()
    for wcnt, line in enumerate(lines):
        grapheme, phoneme = line.split(" ",1)
        words.append(grapheme)
        phonemes.append(phoneme.split(" "))
    return words, phonemes

def get_word_indices_pretrain(file_path, CMUwords, max_dur):
    with open(file_path) as f:
        lines = f.readlines()
    Word_Idx = []
    Start_time = []
    End_time = []
    for i in range(0,len(lines)):
        L = lines[i].strip().split(" ")
        if float(L[2])<max_dur:
            w = L[0].lower()
            Start_time.append(round(float(L[1])*25,2))
            End_time.append(round(float(L[2])*25,2))
            try:
                widx = CMUwords.index(w)
            except ValueError:
                widx = -1
            Word_Idx.append(widx)
        else:
            break
    return Word_Idx, Start_time, End_time

def get_word_indices(file_path, CMUwords):
    with open(file_path) as f:
        lines = f.readlines()
    Word_Idx = []
    Start_time = []
    End_time = []
    for i in range(0, len(lines)):
      L = lines[i].strip().split(" ")
      w = L[0].lower()
      Start_time.append(round(float(L[1])*25,2))
      End_time.append(round(float(L[2])*25,2))
      try:
        widx = CMUwords.index(w)
      except ValueError:
        widx = -1
      Word_Idx.append(widx)   
    return Word_Idx, Start_time, End_time 

def get_LRS_split(args, split):
  count=0
  lst_path = args.LRS_splits_path + split + ".lst"
  with open(lst_path) as f:
    lines = f.readlines()
  word_indices = []
  for l in tqdm(lines):  
    Fwidx = {}
    Fwidx['view'] = 'UK'
    L = l.strip().split(" ")
    fn = L[0]
    Fwidx['fn'] = fn
    if split == "pretrain":
      if os.path.isfile(args.LRS_pretrain_path+fn+'.txt'):    
        widx, start, end = get_word_indices_pretrain(args.LRS_pretrain_path+fn+'.txt', args.CMUwords, args.pretrain_maxdur) 
        Fwidx['widx'] = widx
        Fwidx['start_word'] = start
        Fwidx['end_word']= end
        word_indices.append(Fwidx)
    else:
      if os.path.isfile(args.LRS_main_path+fn+'.txt'):
        widx, start, end = get_word_indices(args.LRS_main_path+fn+'.txt', args.CMUwords)
        Fwidx['widx'] = widx
        Fwidx['start_word'] = start
        Fwidx['end_word']= end
        word_indices.append(Fwidx)
  return word_indices

def get_LRS_splits():
    parser = argparse.ArgumentParser(description='Script for creating main splits of LRS.')
    parser.add_argument('--LRS_pretrain_path',
        default='../data/lrs2/word_alignments/pretrain/')
    parser.add_argument('--LRS_main_path',
        default='../data/lrs2/word_alignments/main/')
    parser.add_argument('--CMUdict_path', default='../data/vocab/cmudict.dict')
    parser.add_argument('--LRS_splits_path', default='../data/lrs2/file_lists/')
    parser.add_argument('--pretrain_maxdur', default=500) # it was 3.6!!! I increased it to 500
    parser.add_argument('--seed', default=5)
    args = parser.parse_args()
    args.CMUwords, args.phonemes = get_CMU_words(args.CMUdict_path)
    S = ['val', 'test', 'pretrain', 'train']
    Dsplits = {}
    for i,s in enumerate(S):
        Dsplits[s] = get_LRS_split(args, s)
    with open('../data/lrs2/DsplitsLRS2.json', 'w') as fp:
      json.dump(Dsplits, fp)
    
 
if __name__ == '__main__':
  get_LRS_splits() 
