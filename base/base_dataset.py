import os
import json
import argparse
import tqdm
import torch
import torchtext
import numpy as np

class BaseDataset(torch.utils.data.Dataset):

   def __init__(self, num_words, Wstruct, num_phoneme_thr):
     self.num_words = num_words
     self.Wstruct = Wstruct
     self.num_phoneme_thr = num_phoneme_thr
     self.word_indices = 0
     self.word_mask = 0
     self.length = 0
   
   def set_word_mask(self):
        """Restrict to words that meet the minimum number of phonemes threshold.
        """
        WM = [False] * self.num_words
        WI = [-1] * self.num_words
        for k in range(0, len(self.Wstruct)):
            WI[self.Wstruct[k].idx] = k
            if len(self.Wstruct[k].phoneme) >= self.num_phoneme_thr:
                WM[self.Wstruct[k].idx] = True
        return WM, WI

   def get_word_mask(self):
        return self.word_mask

   def get_word_indices(self):
        return self.word_indices

   def get_GP(self, widx):
        
        if widx < self.num_words and self.word_indices[widx] != -1:
            example = self.Wstruct[self.word_indices[widx]]
            return (example.grapheme, example.phoneme)
        return -1
   
   def __len__(self):
     return self.length



