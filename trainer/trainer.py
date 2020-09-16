import numpy as np
import torch
import math
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
import model.loss as module_loss
import model.metric as module_met
from tqdm import tqdm
from model.metric import AverageMeter
from data_loader.datasets import DatasetV
from parse_config import ConfigParser
from torch.utils.data import DataLoader
import time
from torch.autograd import  Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F

def transform_batch(lstV_widx_sent, word_mask, Nw, config, sample_new_neg=0.0):
    vidx = []
    widx_sent = []
    widx = []
    target = []
    lens = []
    poswidx = []
    batch_size = len(lstV_widx_sent)
    start_times = []
    end_times = []
    lstV_widx_sent_real = []
    for k in range(0,batch_size):
        if lstV_widx_sent[k][0].size(0)>1:
            lstV_widx_sent_real.append(lstV_widx_sent[k])
    batch_size = len(lstV_widx_sent_real)
    for k in range(0,batch_size):
        for i, l in enumerate(lstV_widx_sent_real[k][1]):
            if l != -1:
                vidx.append(k)
                widx.append(l)
                lens.append(lstV_widx_sent_real[k][0].size(0))
                target.append(1)                
                poswidx.append(l)
                if len(lstV_widx_sent_real[k])>3:
                  start_times.append(lstV_widx_sent_real[k][4][i])
                  end_times.append(lstV_widx_sent_real[k][5][i])

    for k in range(0,batch_size):
        for i, l in enumerate(lstV_widx_sent_real[k][1]):
            if l != -1:
                sample_new_word = np.random.binomial(size=1,n=1,p=sample_new_neg)
                if sample_new_word==0:
                    tries = 0
                    nidx = poswidx[np.random.randint(len(poswidx))]
                    while any(x == nidx for x in lstV_widx_sent_real[k][1]):
                        nidx = poswidx[np.random.randint(len(poswidx))]
                        tries +=1
                        if tries >10:
                          nidx = np.random.randint(Nw,dtype='int64')
                          while word_mask[nidx]==False or any(x == nidx for x in lstV_widx_sent_real[k][1]):
                              nidx = np.random.randint(Nw,dtype='int64')
                vidx.append(k)
                widx.append(nidx)
                lens.append(lstV_widx_sent_real[k][0].size(0))
                target.append(0)
                if len(lstV_widx_sent_real[k])>3:
                  start_times.append(lstV_widx_sent_real[k][4][i])
                  end_times.append(lstV_widx_sent_real[k][5][i])

    batch_size = len(vidx)
    vidx = np.asarray(vidx)
    widx = np.asarray(widx)
    lens = np.asarray(lens)
    target = np.asarray(target)
    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)
    Ilens = np.argsort(-lens)
    lens = lens[Ilens]
    widx = widx[Ilens]
    target = target[Ilens]
    vidx = vidx[Ilens]
    if len(lstV_widx_sent_real[k])>3: 
      start_times = start_times[Ilens]
      end_times = end_times[Ilens]
    
    if len(lens)>0:
        if config["arch"]["args"]["rnn2"]==True:
          max_len = lens[0]
          localization_mask_boundaries =np.ones((batch_size,max_len))
          localization_mask = np.zeros((batch_size,max_len))
          batchV = np.zeros((batch_size,max_len,lstV_widx_sent_real[0][0].size(1))).astype('float')
          for i in range(0, batch_size):
            batchV[i,:lens[i],:] = lstV_widx_sent_real[vidx[i]][0].clone()

        else:
          max_len = lens[0]
          batchV = np.zeros((batch_size,max_len,lstV_widx_sent_real[0][0].size(1))).astype('float')
          max_out_len, max_out_len_start_out = in2out_idx(max_len)
          out_lens = np.array([ in2out_idx(ll)[0] for ll in lens ]).astype(lens.dtype)
          localization_mask_boundaries =np.ones((batch_size,max_out_len))
          localization_mask = np.ones((batch_size,max_out_len))
  
          for i in range(0, batch_size):
            if config["localisation"]["loc_loss"]:
              if target[i]==1:
                w_st_orig = w_st = math.ceil(start_times[i])
                w_end_orig = w_end = math.floor(end_times[i])
                w_st, w_st_start_out = in2out_idx(w_st)
                w_end, w_end_start_out = in2out_idx(w_end)
                assert w_end >= w_st
                w_st = max(0, w_st)
                w_end = max(w_end, w_st+1)
                if w_end > max_out_len:
                  w_end = max_out_len
                  if w_st == max_out_len and w_end == max_out_len:
                    w_st = w_st-1
              if target[i] == 0:
                w_st = 0
                w_end = max_out_len
              if len( localization_mask[i, w_st: w_end]) == w_end - w_st and w_end != w_st:
                localization_mask[i, w_st:w_end] = np.zeros(w_end-w_st)
              else:
                print("mismatch localization mask shapes")
            batchV[i,:lens[i],:] = lstV_widx_sent_real[vidx[i]][0].clone()

    return batchV, lens, widx, target, localization_mask, localization_mask_boundaries


def in2out_idx(idx_in):
  layers = [
    { 'type': 'conv3d', 'n_channels': 32,  'kernel_size': (1,5,5), 'stride': (1,2,2), 'padding': (0,2,2)  ,
          'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)} },

    { 'type': 'conv3d', 'n_channels': 64, 'kernel_size': (1,5,5), 'stride': (1,2,2), 'padding': (0,2,2),
      # 'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)}
      },

  ]
  layer_names = None
  from misc.compute_receptive_field import calc_receptive_field
  idx_out, _, _, offset = calc_receptive_field(layers, idx_in, layer_names)
  return idx_out, offset

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, optimizer, config, lr_scheduler, num_words, logger, train_dataset, 
                 train_dataloader, val_dataset, val_dataloader, len_epoch=None):
        super().__init__(model, optimizer, config, lr_scheduler)
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if len_epoch is None:
          self.len_epoch = len(self.train_dataloader)
        else:
          self.train_dataloader = inf_loop(self.train_dataloader)
          self.len_epoch = len_epoch
        
        self.BCE_loss = torch.nn.BCEWithLogitsLoss() 
        self.dec_weight_loss = config["loss"]["dec_weight_loss"]
        self.loc_weight_loss= config["loss"]["loc_weight_loss"]
        self.kws_weight_loss = config["loss"]["kws_weight_loss"]
        self.g2p =  config["arch"]["args"]["g2p"]
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_words = num_words
        self.start_BEloc_epoch = config["data_loader"]["args"]["start_BEloc_epoch"]
        self.use_BE_localiser = False
        self.clip = 2.3
        self.do_validation = True
        self.rnn_present = config["arch"]["args"]["rnn2"]

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """
        if epoch > 40: 
          self.len_epoch = len(self.train_dataloader)

        if epoch > 60 and self.config["arch"]["args"]["rnn2"]==True:
          assert self.dec_weight_loss == 1
          self.dec_weight_loss = 0.1

        if epoch >= self.start_BEloc_epoch:
          self.use_BE_localiser = True

        self.model.train()
        batch_time = AverageMeter("batch_time")
        data_time = AverageMeter("data_time")
        losses_kws = AverageMeter("losses_kws")
        losses_dec = AverageMeter("losses_dec")
        losses_loc = AverageMeter("losses_loc")
        top1 = AverageMeter("top1")
        end = time.time()
  
        pbar = tqdm(total=len(self.train_dataloader))
        for i, lstVwidx in enumerate(self.train_dataloader):
          count = []
          positives = 0
          for k in range(0,len(lstVwidx)):
            for l in lstVwidx[k][1]:
              if l != -1:
                positives +=1
                if l not in count:
                  count.append(l)
          if len(count)>1:   
            input, lens, widx, target, localization_mask,localization_mask_boundaries= transform_batch(lstVwidx, self.train_dataset.get_word_mask(),
                self.num_words, self.config)
            target = torch.from_numpy(target).cuda(async=True)
            input = torch.from_numpy(input).float().cuda(async=True)
            localization_mask = torch.from_numpy(localization_mask).float().cuda(async=True)
            widx = torch.from_numpy(widx).cuda(async=True)
            grapheme = []
            phoneme = []
            p_lens = []
            for w in widx:
              p_lens.append(len(self.train_dataset.get_GP(w)[0]))
              grapheme.append(self.train_dataset.get_GP(w)[0])
              phoneme.append(self.train_dataset.get_GP(w)[1])
            input_var = Variable(input)
            p_lens = np.asarray(p_lens)
            target_var = Variable(target.view(-1,1)).float()
            if self.g2p:
              graphemeTensor = Variable(self.train_dataset.grapheme2tensor_g2p(grapheme)).cuda()
              phonemeTensor = Variable(self.train_dataset.phoneme2tensor_g2p(phoneme)).cuda()
              preds = self.model(vis_feat_lens=lens, p_lengths=p_lens, phonemes=phonemeTensor[:-1].detach(),
                graphemes=graphemeTensor.detach(), vis_feats=input_var, use_BE_localiser
                =self.use_BE_localiser, epoch=epoch, config=self.config)
              tdec = phonemeTensor[1:]
            else:
              graphemeTensor = Variable(self.train_dataset.grapheme2tensor(grapheme)).cuda()
              phonemeTensor = Variable(self.train_dataset.phoneme2tensor(phoneme)).cuda()
              preds = self.model(vis_feat_lens=lens, p_lengths=p_lens, phonemes=phonemeTensor.detach(),
                graphemes=graphemeTensor[:-1].detach(), vis_feats=input_var, use_BE_localiser
                =self.use_BE_localiser, epoch=epoch, config=self.config) #changed vis_feat_lens from lens to p_lens
              tdec = graphemeTensor[1:]
            loss_dec = module_loss.nll_loss(preds["odec"].view(preds["odec"].size(0)*preds["odec"].size(1),-1),
              tdec.view(tdec.size(0)*tdec.size(1)))
            loss_kws = self.BCE_loss(preds["max_logit"], target_var)
            if self.loc_weight_loss:
              localization_mask = localization_mask*-1000000
              o_logits = localization_mask + preds["o_logits"].squeeze(-1)
              max_localised = o_logits.max(1)[0]
              loss_loc = self.BCE_loss(max_localised.unsqueeze(1), target_var)
              loss_total = self.kws_weight_loss*loss_kws + self.dec_weight_loss*loss_dec + self.loc_weight_loss*loss_loc
            else: 
              loss_total = self.kws_weight_loss*loss_kws+ self.dec_weight_loss*loss_dec
              loss_loc = loss_total
            PTrue = preds["keyword_prob"]
            PFalseTrue = torch.cat((PTrue.add(-1).mul(-1),PTrue),1)
            prec1 = module_met.accuracy(PFalseTrue.data, target, topk=(1,))[0]
            losses_kws.update(loss_kws.item(), input.size(0))
            losses_dec.update(loss_dec.item(), input.size(0))
            losses_loc.update(loss_loc.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            self.optimizer.zero_grad()
            loss_total.backward()
            clip_grad_norm(self.model.parameters(), self.clip, 'inf') #this might not work
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

  
          pbar.update(1)
        self.writer.set_step(epoch)
        self.writer.add_scalar("loss_kws", losses_kws.avg)
        self.writer.add_scalar("loss_loc", losses_loc.avg)
        self.writer.add_scalar("loss_dec", losses_dec.avg)
        self.writer.add_scalar("acc", top1.avg)
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t'
              'Loss_kws {loss_kws.val:.4f} ({loss_kws.avg:.4f})\t'
              'Loss_loc{loss_loc.val:.4f} ({loss_loc.avg:.4f})\t'
              'Loss_dec {loss_dec.val:.4f} ({loss_dec.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              epoch, i, len(self.train_dataloader), batch_time=batch_time, data_time= data_time,
              loss_kws=losses_kws, loss_loc=losses_loc, loss_dec=losses_dec, top1=top1))
        self.logger.info('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t'
              'Loss_kws {loss_kws.val:.4f} ({loss_kws.avg:.4f})\t'
              'Loss_loc {loss_loc.val:.4f} ({loss_loc.avg:.4f})\t'
              'Loss_dec {loss_dec.val:.4f} ({loss_dec.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              epoch, i, len(self.train_dataloader), batch_time=batch_time, data_time= data_time,
              loss_kws=losses_kws, loss_loc=losses_loc, loss_dec=losses_dec, top1=top1))

        pbar.close()
         
        if self.do_validation:
          self._valid_epoch(epoch) 

        if self.lr_scheduler is not None:
          self.lr_scheduler.step()
      


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """

        batch_time = AverageMeter("batch_time")
        losses = AverageMeter("losses")
        losses_kws = AverageMeter("losses_kws")
        losses_dec = AverageMeter("losses_dec")
        losses_loc = AverageMeter("losses_loc")
        top1 = AverageMeter("top1")
        myprec = AverageMeter("myprec")
        myrec = AverageMeter("myrec")
        TarRank = []
        NonRank = []
        labels = []
        scores = []

        for k in range(0, self.num_words):
          TarRank.append([])
          NonRank.append([])

        self.model.eval()
        end = time.time()
        pbar = tqdm(total=len(self.val_dataloader))

        for i, lstVwidx in enumerate(self.val_dataloader):
          count = []
          positives = 0
          for k in range(0,len(lstVwidx)):
            for l in lstVwidx[k][1]:
              if l != -1:
                positives +=1
                if l not in count:
                  count.append(l)
          if len(count)>1:   
            input, lens, widx, target, localization_mask,localization_mask_boundaries= transform_batch(lstVwidx, self.val_dataset.get_word_mask(),
                self.num_words, self.config)
            labels = np.concatenate((labels,target), axis=0)
            targetInt = target.astype('int32')
            target = torch.from_numpy(target).cuda(async=True)
            input = torch.from_numpy(input).float().cuda(async=True)
            localization_mask = torch.from_numpy(localization_mask).float().cuda(async=True)
            widx = torch.from_numpy(widx).cuda(async=True)
            input_var = Variable(input)
            target_var = Variable(target.view(-1,1)).float()
            grapheme = []
            phoneme = []
            p_lens = []
            for w in widx:
                p_lens.append(len(self.val_dataset.get_GP(w)[0]))
                grapheme.append(self.val_dataset.get_GP(w)[0])
                phoneme.append(self.val_dataset.get_GP(w)[1])
            p_lens = np.asarray(p_lens) 
            if self.g2p:
              graphemeTensor = Variable(self.val_dataset.grapheme2tensor_g2p(grapheme)).cuda()
              phonemeTensor = Variable(self.val_dataset.phoneme2tensor_g2p(phoneme)).cuda()
              preds = self.model(vis_feat_lens=lens, p_lengths=p_lens, phonemes=phonemeTensor[:-1].detach(),
                graphemes=graphemeTensor.detach(), vis_feats=input_var, use_BE_localiser
                =self.use_BE_localiser, epoch=epoch, config=self.config)
              tdec = phonemeTensor[1:]
            else:
              graphemeTensor = Variable(self.val_dataset.grapheme2tensor(grapheme)).cuda()
              phonemeTensor = Variable(self.val_dataset.phoneme2tensor(phoneme)).cuda()
              preds = self.model(vis_feat_lens=lens, p_lengths=p_lens, phonemes=phonemeTensor.detach(),
                graphemes=graphemeTensor[:-1].detach(), vis_feats=input_var, use_BE_localiser
                =self.use_BE_localiser, epoch=epoch, config=self.config) #changed vis_feat_lens from lens to p_lens
              tdec = graphemeTensor[1:]
            scores = np.concatenate((scores, preds['keyword_prob'].view(1, len(target)).detach().cpu().numpy()[0]), axis=0) 
            loss_dec = module_loss.nll_loss(preds["odec"].view(preds["odec"].size(0)*preds["odec"].size(1),-1), tdec.view(tdec.size(0)*tdec.size(1)))
            loss_kws = self.BCE_loss(preds["max_logit"], target_var )
            if self.loc_weight_loss:
              localization_mask = localization_mask*-1000000
              o_logits = localization_mask + preds["o_logits"].squeeze(-1)
              max_localised = o_logits.max(1)[0]
              loss_loc = self.BCE_loss(max_localised.unsqueeze(1), target_var)
              loss_total = self.kws_weight_loss*loss_kws + self.dec_weight_loss*loss_dec + self.loc_weight_loss*loss_loc
            else:
              loss_total = self.kws_weight_loss*loss_kws + self.dec_weight_loss*loss_dec
              loss_loc = loss_total
            PTrue = preds["keyword_prob"]
            PFalseTrue = torch.cat((PTrue.add(-1).mul(-1),PTrue),1)
            prec1 = module_met.accuracy(PFalseTrue.data, target, topk=(1,))[0]
            PR = module_met.PrecRec(PFalseTrue.data, target, topk=(1,))
            losses.update(loss_total.item(), input.size(0))
            losses_kws.update(loss_kws.item(), input.size(0))
            losses_dec.update(loss_dec.item(), input.size(0))
            losses_loc.update(loss_loc.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            myprec.update(PR[0], (PFalseTrue.data[:,1]>0.5).sum())
            myrec.update(PR[1], target.sum())
          pbar.update(1)
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar("loss_kws", losses_kws.avg)
        self.writer.add_scalar("loss_loc", losses_loc.avg)
        self.writer.add_scalar("loss_dec", losses_dec.avg)
        self.writer.add_scalar("acc", top1.avg)
        batch_time.update(time.time() - end)
        end = time.time()
        
        print("Prec@1 {top1.avg:.3f}, Precision {myprec.avg:.3f}, Recall {myrec.avg:.3f}, Loss_kws {loss_kws.avg:.4f}, Loss_dec {loss_dec.avg:.4f},Loss_loc {loss_loc.avg:.4f}".format(top1=top1, myprec=myprec, myrec=myrec, loss_kws=losses_kws, loss_dec=losses_dec, loss_loc=losses_loc))
        self.logger.info("Prec@1 {top1.avg:.3f}, Precision {myprec.avg:.3f}, Recall {myrec.avg:.3f}, Loss_kws {loss_kws.avg:.4f}, Loss_dec {loss_dec.avg:.4f}".format(top1=top1, myprec=myprec, myrec=myrec, loss_kws=losses_kws, loss_dec=losses_dec))
        pbar.close()


