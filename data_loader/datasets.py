import os
import json
import argparse
import tqdm
import torch
import torchtext
import numpy as np
import h5py
from pathlib import Path
import torch.utils.data
import torchtext.data as data
import re
from torchvision import datasets, transforms
from base import BaseDataLoader
from base import BaseDataset
from torch.utils.data.dataloader import default_collate


class CMUDict(torchtext.data.Dataset):
    def __init__(self, data_lines, word_indices, i_field, g_field, p_field):
        fields = [('idx', i_field), ('grapheme', g_field), ('phoneme', p_field)]
        examples = []  
        wcnt = 0        
        for wcnt, line in enumerate(data_lines):
            grapheme, phoneme = line.split(" ",1)
            examples.append(data.Example.fromlist([word_indices[wcnt], grapheme, phoneme], fields))
        super().__init__(examples, fields)


    @classmethod
    def splits_datasetv(cls, cmu_dict_path, i_field, g_field, p_field):
        with open(cmu_dict_path) as f:
            lines = f.readlines()
        with open('../data/lrs2/LRS2_test_words.json', "r") as fp:
          widx_object = json.load(fp)
          test_words = widx_object['widx']
        with h5py.File("./permute_word_splits.hdf5",'r') as f:
            I = f['permute'][:]       
        linesP = []
        for i in range(0,len(lines)):
            linesP.append(lines[I[i]])
        train_lines, val_lines, test_lines = [], [], []
        train_widx, val_widx, test_widx = [], [], []
        for i, line in enumerate(linesP):
            if i % 20 == 0 and I[i] not in test_words:
                val_lines.append(line)
                val_widx.append(I[i])
            elif i % 20 < 6 and I[i] not in test_words:
                val_lines.append(line)
                val_widx.append(I[i])
            else:
              if I[i] not in test_words:
                train_lines.append(line)
                train_widx.append(I[i])
        train_data = cls(
            data_lines=train_lines,
            word_indices=train_widx,
            i_field=i_field,
            g_field=g_field,
            p_field=p_field
        )
        val_data = train_data
        return (train_data, val_data)


    @classmethod
    def splits_datasetv_test644(cls, cmu_dict_path, i_field, g_field, p_field):
      with open(cmu_dict_path) as f:
        lines = f.readlines()
      with open('data/lrs2/LRS2_test_words.json', "r") as fp:
        widx_object = json.load(fp)
        test_words = widx_object['widx']
      test_lines = []
      for i in test_words:
        test_lines.append(lines[i])
      test_data = cls(
         data_lines = test_lines,
         word_indices = test_words,
         i_field = i_field,
         g_field = g_field,
         p_field = p_field
      )
      return test_data
    
    
    @classmethod
    def splits_dataset_lrw(cls, cmu_dict_path, i_field, g_field, p_field): 
      with open(cmu_dict_path) as f:
        lines = f.readlines()
      with open('../data/lrw/LRW_train_words.json', "r") as fp:
        widx_object = json.load(fp)
        widx = widx_object['widx'] 
      data_lines = []
      for i in widx:
        data_lines.append(lines[i])
      data = cls(
          data_lines = data_lines,
          word_indices = widx,
          i_field = i_field,
          g_field = g_field,
          p_field = p_field,
      )
      return data
  
def get_splits_datasetv(cmu_dict_path, data_struct_path, splitname):
    i_field = data.Field(lambda x: x)
    g_field = data.Field(init_token='<s>',
                     tokenize=(lambda x: list(x.split('(')[0]))) #sequence reversing removed                                                                   
    p_field = data.Field(init_token='<os>', eos_token='</os>',
                     tokenize=(lambda x: x.split('#')[0].split()))
                    
    if data_struct_path == "../data/lrw/DsplitsLRW.json":
      Wstruct = CMUDict.splits_dataset_lrw(cmu_dict_path, i_field, g_field, p_field)
    elif splitname == "test":
      Wstruct = CMUDict.splits_datasetv_test644(cmu_dict_path, i_field, g_field, p_field)
      return Wstruct
    else:
      WsplitsLst = CMUDict.splits_datasetv(cmu_dict_path, i_field, g_field, p_field)
      S = ['train', 'val']
      Wsplits = {}
      for i,s in enumerate(S):
        Wsplits[s] = WsplitsLst[i]
      if splitname=='val':
        Wstruct=Wsplits['val']
      else:
        Wstruct = Wsplits[splitname]
    return Wstruct

def merge_train_pretrain(Dstruct):
    for s in Dstruct['pretrain']:
      Dstruct['train'].append(s)
    return Dstruct
 
class DatasetV(BaseDataset):
 
    def __init__(self, num_words, num_phoneme_thr, cmu_dict_path, Vpath,
        splitname, data_struct_path, p_field_vocab_path, g_field_vocab_path,merge):
        self.data_struct_path = data_struct_path
        with open(data_struct_path, 'r') as f:
          Dstruct = json.load(f)
        if "train" in Dstruct.keys(): 
          self.Ntrain = len(Dstruct["train"]) 
        if merge == True:         
          Dstruct = merge_train_pretrain(Dstruct) 
        self.Dstruct = Dstruct[splitname]
        self.splitname = splitname
        self.Vpath = Vpath  
        self.wstruct = get_splits_datasetv(cmu_dict_path,data_struct_path, splitname) #need to change
        self.num_phoneme_thr = num_phoneme_thr
        self.num_words = num_words
        super().__init__(self.num_words, self.wstruct, self.num_phoneme_thr)
        self.word_mask, self.word_indices = self.set_word_mask()
        self.length = len(self.Dstruct)

        with open(g_field_vocab_path, 'r') as f:
            self.g_field_vocab = json.load(f)
        with open(p_field_vocab_path, 'r') as f:
            self.p_field_vocab = json.load(f)
        msg = "g_size: {}, p_size: {}"
        self.g_size = len(self.g_field_vocab)
        self.p_size = len(self.p_field_vocab)
        print(msg.format(self.g_size, self.p_size))  

    def __getitem__(self, index):
        Didx = 0
        if self.splitname == 'train' and index>=self.Ntrain: #add later
          Didx = 1  
        if Didx < len(self.Vpath) and index < len(self.Dstruct):
          fpath = os.path.join(self.Vpath[Didx],self.Dstruct[index]['fn']+'.mp4.npy')
        else:
          return self.__getitem__(index+1)
        if not os.path.isfile(fpath):
          return self.__getitem__(index+1)
        V = np.load(fpath)
        V = torch.from_numpy(V).float()
        if V.size()[0]>500:
          return self.__getitem__(index+1)
        widx = self.Dstruct[index]['widx']
        if 'start_word' in self.Dstruct[index].keys():
          start_times = self.Dstruct[index]['start_word'] 
          end_times = self.Dstruct[index]['end_word']
        for w in range(0,len(widx)):
            if widx[w] == -1 or self.word_mask[widx[w]]==False:
                widx[w] = -1
        if 'start_word' in self.Dstruct[index].keys(): 
            self.Dstruct[index]['view'] = 'UK'
            return V, widx, self.Dstruct[index]['fn'], self.Dstruct[index]['view'], start_times, end_times
        else:
            return V, widx, self.Dstruct[index]['fn']
 
    def grapheme2tensor(self, grapheme):
        mlen = 0
        for i, w in enumerate(grapheme):
            if mlen < len(w):
                mlen = len(w)
        G = np.ones((mlen + 2, len(grapheme)), dtype='int64')
        G[:, :] = self.g_field_vocab.index('<pad>')
        gs = self.g_field_vocab.index('<s>')
        ge = self.g_field_vocab.index("</s>")
        for i, w in enumerate(grapheme):
            G[0, i] = gs
            for j, g in enumerate(w):
                G[j + 1, i] = self.g_field_vocab.index(g)
                G[j + 2, i] = ge
        return torch.from_numpy(G)

    def grapheme2tensor_g2p(self, grapheme):
        mlen = 0
        for i,w in enumerate(grapheme):
            if mlen<len(w):
                mlen = len(w)
        G = np.ones((mlen+1,len(grapheme)),dtype='int64')
        G[:,:] = self.g_field_vocab.index('<pad>')
        gs = self.g_field_vocab.index('<s>')
        for i,w in enumerate(grapheme):
            for j,g in enumerate(w):
                G[j,i] = self.g_field_vocab.index(g)
                G[j+1,i] = gs
        G = np.flip(G,0).copy()
        return torch.from_numpy(G)

    def phoneme2tensor(self, phoneme):
        mlen = 0
        for i, w in enumerate(phoneme):
            if mlen < len(w):
                mlen = len(w)
        P = np.ones((mlen + 1, len(phoneme)), dtype='int64')
        P[:, :] = self.p_field_vocab.index('<pad>')
        ps = self.p_field_vocab.index('<os>')
        for i, w in enumerate(phoneme):
            for j, p in enumerate(w):
                P[j, i] = self.p_field_vocab.index(p)
                P[j + 1, i] = ps
        P = np.flip(P, 0).copy()
        return torch.from_numpy(P)

    def phoneme2tensor_g2p(self, phoneme):
        mlen = 0
        for i,w in enumerate(phoneme):
            if mlen<len(w):
                mlen = len(w)
        P = np.ones((mlen+2,len(phoneme)),dtype='int64')
        P[:,:] = self.p_field_vocab.index('<pad>')
        ps = self.p_field_vocab.index('<os>')
        pe = self.p_field_vocab.index('</os>')
        for i,w in enumerate(phoneme):
            P[0,i] = ps
            for j,p in enumerate(w):
                P[j+1,i] = self.p_field_vocab.index(p)
                P[j+2,i] = pe
        return torch.from_numpy(P)

if  __name__ == "__main__":
    main()