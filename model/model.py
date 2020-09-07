import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, BaseModule
from model.model_helpers import Beam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math

class KWSModel(BaseModel):
    def __init__(self, hiddenDNNV, dimRnn3, inputDimV, hiddenDimV, birnnV, d_word_emb,
                 outdpV, p_size, g_size, d_embed, d_hidden, embDO, beam_size, num_heads,rnn2, fixed_length_embedding, shortcut, loc_acc,g2p):

        super().__init__()
        self.classifier_init = Classifier_init(
            inputDimV=inputDimV,
            hiddenDimV=hiddenDimV,
            d_word_emb=d_word_emb,
            hiddenDNNV=hiddenDNNV,
            beam_size=beam_size,
            birnnV=birnnV,
            outdpV=outdpV,
            p_size=p_size,
            g_size=g_size,
            d_embed=d_embed,
            d_hidden=d_hidden,
            embDO=embDO,
            num_heads=num_heads,
            rnn2=rnn2,
            fixed_length_embedding=fixed_length_embedding,
            shortcut = shortcut,
            loc_acc = loc_acc,
            g2p=g2p
        )
        self.classifier_BE = Classifier_BE(hiddenDNNV=hiddenDNNV, dimRnn3=dimRnn3)

    def forward(self, epoch, vis_feats, vis_feat_lens, p_lengths, graphemes, phonemes, use_BE_localiser, config):
        if epoch == config["data_loader"]["args"]["start_BEloc_epoch"]:
          for param in self.classifier_init.parameters():
            param.requires_grad = False
        else:
          for param in self.classifier_init.parameters():
            param.requires_grad = True 
        odec, vis_feat_lens, o_init, o_rnn, plotted_mask, o_logits, indices = self.classifier_init(
            x=vis_feats,
            x_lengths=vis_feat_lens,
            phoneme=phonemes,
            grapheme=graphemes,
            p_lengths=p_lengths
        )
        if epoch >= config["data_loader"]["args"]["start_BEloc_epoch"]: 
            o, odec, idx_max, o_logits = self.classifier_BE(o_rnn, odec, vis_feat_lens) 
        else:
            idx_max = Variable(torch.LongTensor(vis_feats.size(0)).fill_(0).cuda())
            o = o_init
        keyword_prob = torch.sigmoid(o)
        return {"max_logit": o, "odec": odec, "idx_max": indices, 
                "keyword_prob": keyword_prob, "plot": plotted_mask, "o_logits": o_logits}

class Classifier_init(nn.Module):
    def __init__(self, inputDimV, hiddenDimV, birnnV, d_word_emb, outdpV, hiddenDNNV,
                 p_size, g_size, d_embed, d_hidden, embDO, beam_size, num_heads,rnn2, fixed_length_embedding, shortcut, loc_acc,g2p):
        super().__init__()
        self.rnn1 = nn.LSTM(
            input_size=inputDimV,
            hidden_size=hiddenDimV,
            batch_first=True,
            bidirectional=birnnV,
            num_layers=1,
            dropout=0,
        ).float()
        self.linrnn = nn.Linear(2 * hiddenDimV, hiddenDimV).float()
        self.rnn2_present = rnn2
        self.rnn2 = nn.LSTM(
            input_size=hiddenDimV *2,
            hidden_size=hiddenDimV,
            bidirectional=birnnV,
            batch_first=True,
            num_layers=1,
            dropout=0,
        )
        self.d_word_emb = d_word_emb
        self.inputDimV = inputDimV
        self.hiddenDimV = hiddenDimV
        self.inBN = nn.BatchNorm1d(inputDimV)
        self.lin_logits = nn.Conv1d(
            in_channels=128,
            out_channels=1,
            kernel_size=1,
            dilation=1,
            padding=0,
            stride=1,
            groups=1,
            bias=1,
        )
        Nh = 2 if birnnV else 1
        self.rnnBN = nn.BatchNorm1d(hiddenDimV)
        self.wBN = nn.BatchNorm1d(d_word_emb)
        self.outProj_L0 = nn.Linear(in_features=Nh * hiddenDimV, out_features=hiddenDNNV)
        self.outDO_L0 = nn.Dropout(p=outdpV)
        self.DNNnonlinear_L0 = nn.LeakyReLU().float()
        self.DNNnonlinear_L1 = nn.LeakyReLU().float()
        self.DNNnonlinear_L2 = nn.LeakyReLU().float()
        self.outProj_L1 = nn.Linear(in_features=hiddenDNNV, out_features=hiddenDNNV // 2)
        self.outDO_L1 = nn.Dropout(p=outdpV)
        self.outProj_L2 = nn.Linear(in_features=hiddenDNNV // 2, out_features=1)
        self.indpV = 0.20
        self.g2p = g2p
        if self.g2p:
            self.enc_dec = G2P(
                p_size=g_size,
                g_size=p_size,
                d_embed=d_embed,
                d_hidden=d_hidden,
                d_word_emb=d_word_emb,
                embDO=embDO,
                beam_size=beam_size,
            )
        else:
            self.enc_dec = G2P(
                p_size=p_size,
                g_size=g_size,
                d_embed=d_embed,
                d_hidden=d_hidden,
                d_word_emb=d_word_emb,
                embDO=embDO,
                beam_size=beam_size,
            )
        self.linear_attn_keys = nn.Linear(256,512)
        self.linear_attn_values = nn.Linear(384,512)
        self.linear_attn_queries = nn.Linear(128,512)
        self.num_heads = num_heads
        self.d_k = inputDimV // self.num_heads
        self.linear_mask = nn.Linear(self.num_heads,512)
        self.final_lin = nn.Linear(512,512)
        self.shortcut = shortcut
        if self.shortcut == True:
          self.conv3 = nn.Conv2d(1,32, kernel_size=5, stride=(1,1), padding=(2,2)) #padding used to (2,2)
          self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=(2,2), padding=(2,2)) #padding used to (2,2)
          self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=(1,2), padding=(2,2)) #padding used to be (2,2)
          self.batch1 = nn.BatchNorm2d(32)
          self.max_pool = nn.MaxPool2d(2, stride=(1,2))
          self.batch2 = nn.BatchNorm2d(128)
          self.batch3 = nn.BatchNorm2d(256)
          self.dropout = nn.Dropout(0.2)
          self.fc1 = nn.Linear(256, 512)
          self.fc2 = nn.Linear(512, 128)
          self.fc3 = nn.Linear(128, 1)
        else:
          self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=(2,2), padding=(2,2)) #padding used to (2,2)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=(1,2), padding=(2,2)) #padding used to be (2,2)
          self.batch1 = nn.BatchNorm2d(32)
          self.max_pool = nn.MaxPool2d(2, stride=(1,2))
          self.batch2 = nn.BatchNorm2d(64)
          self.dropout = nn.Dropout(0.2)
          self.fc1 = nn.Linear(64, 256)
          self.fc2 = nn.Linear(256, 128)
          self.fc3 = nn.Linear(128, 1)

        self.softmax = nn.LogSoftmax(dim=1)
        self.linear_conv= nn.Linear(self.num_heads,1)
        self.fixed_length_embedding = fixed_length_embedding
        self.project_query = nn.Linear(512,32)
        self.linear_baseline = nn.Linear(384,512)        
        self.linear_shortcut = nn.Linear(128, self.num_heads)
        self.loc_acc = loc_acc     
        
    def forward(self, x, x_lengths, p_lengths, grapheme, phoneme):
        batch_size = x.data.size(0)
        if len(x.data.size())==1:
          import pdb; pdb.set_trace()
        T = x.data.size(1)
        if self.g2p:
            emb, all_emb = self.enc_dec.encoder(grapheme)
            odec, _, __ = self.enc_dec.decoder(
                x_seq=phoneme,
                emb=emb,
                projectEmb=True,
            )            
        else:
            emb, all_emb = self.enc_dec.encoder(phoneme)
            odec, _, __ = self.enc_dec.decoder(
                x_seq=grapheme,
                emb=emb,
                projectEmb=True,
            )
        if self.fixed_length_embedding==True:
          word_embedding = self.wBN(emb)
          word_embedding = word_embedding.unsqueeze(1).expand(batch_size, x.data.size(1), self.d_word_emb) # nb videos x number of frames x 128
        else:
          word_embedding = self.wBN(all_emb.transpose(-2,-1)).transpose(-2,-1)

        if self.loc_acc:
          x = self.inBN(x.reshape(-1, self.inputDimV)).view(batch_size, -1, self.inputDimV) 
        else:
         
          x = self.inBN(x.view(-1, self.inputDimV)).view(batch_size, -1, self.inputDimV)
        if self.training:
            mask = torch.FloatTensor(int(x.size(0)/2), 1, self.inputDimV)
            mask = mask.fill_(1).bernoulli_(1 - self.indpV).div(1 - self.indpV)
            mask = mask.expand(int(x.size(0)/2), x.size(1), x.size(2))
            mask = torch.cat((mask, mask), 0)
            mask = Variable(mask.cuda())
            x = x.mul(mask)
 
        pack = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        o, _ = self.rnn1(pack)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = self.linrnn(o.contiguous().view(-1, 2 * self.hiddenDimV))
        o = o.view(batch_size, -1, self.hiddenDimV)  
        o = self.rnnBN(o.view(-1, self.hiddenDimV))
        o = o.view(batch_size, -1, self.hiddenDimV)  
        
        if self.rnn2_present ==False:
          key = self.linear_attn_keys(o) 
          value = [] 
          query = self.linear_attn_queries(word_embedding) 
          batch_size, nb_videos, dimensions = key.size()
          attention_mask = attention(query, key, value, batch_size, nb_videos,self.num_heads, self.d_k)
          plotted_mask=attention_mask
          
        if self.rnn2_present== True:
            plotted_mask= None
            o = torch.cat((o, word_embedding), 2) 
            attention_mask = self.linear_baseline(o)
            pack = torch.nn.utils.rnn.pack_padded_sequence(attention_mask, x_lengths, batch_first=True)
            o, _ = self.rnn2(pack)
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            o = o.contiguous().view(batch_size * T, -1) 
            o = self.outProj_L0(o) #
            o = self.DNNnonlinear_L0(o)
            o_rnn = o.view(batch_size, T, -1)
            o = torch.sum(o_rnn, 1).div(40)  
            o = self.outDO_L1(o)  
            o = self.outProj_L1(o)  
            o = self.DNNnonlinear_L1(o)  
            o = self.outProj_L2(o).view(batch_size, 1)
            logits = None
            indices = 0

        else:
            o_rnn = None
            if self.shortcut:
              o = self.conv3(attention_mask.transpose(-3,-1)) 
              o = self.batch1(o)
              o = F.relu(o)  
              shortcut = self.project_query(query).transpose(-2,-1).unsqueeze(-2).expand(-1,-1,nb_videos,-1) 
              o = torch.cat((o, shortcut), 1) 
              o = self.conv1(o.transpose(-2,-1))        
              o = self.batch2(o) 
              o = F.relu(o)
              o = self.max_pool(o) 
              o = self.conv2(o)
              o = self.batch3(o) 
              o = F.relu(o)
              o = self.dropout(o)
              o = o.mean(-2)                                          
              bs, _, v_dim = o.shape
              o_prelogits = o.permute([0,2,1]).reshape([-1, o.shape[1]])
              o = F.relu(self.fc1(o_prelogits))
              o = self.dropout(F.relu(self.fc2(o)))
              logits = self.fc3(o)
              logits = logits.reshape([bs, -1, logits.shape[-1]])
              o, indices = logits.max(1)
            else:
              o = self.conv1(attention_mask.transpose(-1,-3).transpose(-1,-2))
              o = self.batch1(o) 
              o = F.relu(o)
              o = self.max_pool(o) 
              o = self.conv2(o)
              o = self.batch2(o) 
              o = F.relu(o)
              o = self.dropout(o)
              o = o.mean(-2)
              bs, _, v_dim = o.shape
              o_prelogits = o.permute([0,2,1]).reshape([-1, o.shape[1]])
              o = F.relu(self.fc1(o_prelogits))
              o = self.dropout(F.relu(self.fc2(o)))
              logits = self.fc3(o)
              logits = logits.reshape([bs, -1, logits.shape[-1]])
              o, indices = logits.max(1)

        return odec, x_lengths, o, o_rnn, plotted_mask, logits, indices

def attention(query, key,value, batch_size, nb_v_frames, num_heads, d_k):
    if len(value)==0:
        key = key.view(batch_size,-1, num_heads, d_k) 
        query= query.view(batch_size,-1, num_heads, d_k) 
        key = key.transpose(1,2).unsqueeze(3)
        query = query.transpose(1,2).unsqueeze(2)
        attention_mask = key[:,:,:,None,:]*query[:, :, None, :,:] 
        attention_mask = attention_mask/math.sqrt(d_k)
        attention_mask = attention_mask.sum(-1).squeeze(-2)
        attention_mask = attention_mask.transpose(-3,-1) 

    else: 
        key = key.view(batch_size,-1,num_heads,d_k) 
        value = value.view(batch_size,-1, num_heads, d_k)
        query= query.view(batch_size,-1,num_heads, d_k)
        key = key.transpose(1,2).unsqueeze(3)  
        query = query.transpose(1,2).unsqueeze(2) 
        value = value.transpose(1,2) 
        attention_mask = key[:,:,:,None,:]*query[:, :, None, :,:] 
        attention_mask = attention_mask/math.sqrt(d_k)
        attention_mask = attention_mask.squeeze().sum(-1) 
        attention_mask = F.softmax(attention_mask, dim=-2)
        attention_mask = attention_mask.transpose(-1,-2) 
        b, h, len_p, f = attention_mask.size()
        attention_mask = torch.matmul(attention_mask,value).view(batch_size, len_p, num_heads*d_k) 
       
    return attention_mask


class Classifier_BE(BaseModel):
    def __init__(self, hiddenDNNV, dimRnn3):
        super().__init__()
        self.DNNnonlinear_L1 = nn.LeakyReLU().float()
        self.outProj_L1 = nn.Linear(hiddenDNNV, dimRnn3).float()
        self.BEBN = nn.BatchNorm1d(hiddenDNNV)
        self.dimRnn3 = dimRnn3 
        self.rnn3 = nn.LSTM(
            input_size=dimRnn3,
            hidden_size=dimRnn3,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=0,
        ).float()
        self.linRnn3 = nn.Linear(2*self.dimRnn3, 1)
        self.FBN = nn.BatchNorm1d(2*self.dimRnn3)

    def forward(self, o_rnn, odec, Lens):
        o = o_rnn
        batch_size = o.data.size(0)
        T = o.data.size(1)
        o = self.outProj_L1(o)
        o = self.DNNnonlinear_L1(o)
        o = o.contiguous().view(batch_size, T, self.dimRnn3)
        pack = torch.nn.utils.rnn.pack_padded_sequence(o, Lens, batch_first=True)
        o, _ = self.rnn3(pack)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = o.contiguous().view(-1, 2 * self.dimRnn3)
        o = self.FBN(o)
        o = self.linRnn3(o)
        o_logits = o.view(batch_size, T, 1)
        max_logit = o_logits.max(1)
        o = max_logit[0].view(batch_size, 1)
        return o, odec, max_logit[1], o_logits


class G2P(BaseModel):
    def __init__(self, p_size, g_size, d_embed, d_hidden, d_word_emb, embDO, beam_size):
        super().__init__()
        self.beam_size = beam_size
        self.decoder = Decoder(g_size, d_embed, d_hidden, d_word_emb)
        self.encoder = Encoder(p_size, d_embed, d_hidden, d_word_emb, embDO)
        
    def forward(self, g_seq, p_seq=None):
        emb, all_emb  = self.encoder(g_seq, True)
        context = None
        if p_seq is not None: 
            return self.decoder(p_seq, emb, True, context)
        else:
            assert g_seq.size(1) == 1 
            raise NotImplementedError()

    def _generate(self, emb, context):
        beam = Beam(self.beam_size, cuda=self.config.cuda)
        h = emb[0].expand(beam.size, emb[0].size(1))
        c = emb[1].expand(beam.size, emb[1].size(1))
        if context is not None:
            context = context.expand(beam.size, context.size(1), context.size(2))

        for _ in range(self.config.max_len): 
            x = beam.get_current_state()
            o, h, c = self.decoder(x.unsqueeze(0), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h.data.copy_(h.data.index_select(0, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(0, beam.get_current_origin()))
        tt = torch.cuda if self.config.cuda else torch
        return tt.LongTensor(beam.get_hyp(0))


class Encoder(BaseModule):
    def __init__(self, vocab_size, d_embed, d_hidden, d_word_emb, embDO):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTM(d_embed, d_hidden, batch_first=True, bidirectional=True)
        self.d_hidden = d_hidden
        self.JHCE = nn.Linear(d_hidden * 2, d_word_emb)
        self.pdo = embDO 
        self.linear = nn.Linear(1000, 500)
        self.linear_2 = nn.Linear(1000,500)
    def forward(self, x_seq, cuda=True):
        e_seq = self.embedding(x_seq).permute(1,0,2)
        
        tt = torch.cuda if cuda else torch  
        if self.embedding.training and self.pdo > 0:
            pdo = self.pdo
            vs = e_seq.size(2)
            mask = np.random.binomial(1, 1 - pdo, (vs, )).astype('float') / (1 - pdo)
            xnp = x_seq.data.cpu().numpy()
            mask = torch.from_numpy(mask[xnp]).float()
            mask = mask.cuda() if cuda else mask
            mask = Variable(mask).detach()
            mask = mask.unsqueeze(2).expand(e_seq.size(0), e_seq.size(1), e_seq.size(2))
            e_seq = e_seq.mul(mask)
        output, (hidden_state, cell_state) = self.lstm(e_seq)
        hidden_state = torch.mean(output,1)
        hidden_state = self.linear(hidden_state)
        cell_state = torch.mean(cell_state,0)
        emb = self.JHCE(torch.cat((hidden_state,cell_state),1))
        all_hidden_states = self.linear_2(output) 
        batch_size, nb_phonemes, dimensions = all_hidden_states.size() 
        all_cell_states = cell_state.unsqueeze(-1).transpose(-2,-1).repeat(1,nb_phonemes,1)
        all_emb = self.JHCE(torch.cat((all_hidden_states, all_cell_states), 2))        
        return emb, all_emb


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden, d_word_emb):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.linear = nn.Linear(d_hidden, vocab_size)
        self.JHCE2h = nn.Linear(d_word_emb, d_hidden)
        self.JHCE2c = nn.Linear(d_word_emb, d_hidden)
    

    def forward(self, x_seq, emb, projectEmb):
        hidden_state = []
        e_seq = self.embedding(x_seq)
        if projectEmb:
            h_ = self.JHCE2h(emb)
            c_ = self.JHCE2c(emb)
        else:
            h_ = emb[0]
            c_ = emb[1]
        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h_, c_ = self.lstm(e, (h_, c_))
            hidden_state.append(h_)

        hidden_state = torch.stack(hidden_state, 0)
        o = self.linear(hidden_state.view(-1, h_.size(1)))
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h_, c_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embDO', type=float, default=0,
                        help="dropout prob on G2P encoder")
    parser.add_argument('--birnnV', type=int, default=1, help="use bidirectional lstms")
    parser.add_argument('--outdpV', default=0.2,
                        help="dropout prob. on penultimate linear layer of KWS model")
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument("--dimRnn3", type=int, default=16,
                        help="dimension of hidden state of BiLSTM KWS classifier")
    parser.add_argument("--g_size", type=int, default=34, help="size of grapheme vocab")
    parser.add_argument("--p_size", type=int, default=73, help="size of phoneme vocab")
    parser.add_argument("--d_word_emb", type=int, default=128,
                        help="dimension of `uprojected` hidden state of the G2P model")
    parser.add_argument("--hiddenDimV", type=int, default=256)
    parser.add_argument("--d_embed", type=int, default=64,
                        help="dimension of hidden state of G2P embeddings")
    parser.add_argument('--d_hidden', default=500, type=int,
                        help="dimension of hidden state of G2P LSTMs")
    parser.add_argument("--inputDimV", type=int, default=512,
                        help="dimension of the visual features used by KWS Model")
    parser.add_argument("--hiddenDNNV", type=int, default=128,
                        help=("determines the size of the linear layers at the end of the"
                              "KWS model.  There are three such layers, with shape:"
                              "(x * B, x) -> (x, x / 2) -> (x / 2 , 2), where x is the"
                              "value of `hiddenDNNV` and `B = 2` if a BiLSTM was used for"
                              "the encoding, and `B = 1` otherwise"))

    args = parser.parse_args()

    Nh = 2 if args.birnnV else 1
    model = KWSModel(
        dimRnn3=args.dimRnn3,
        hiddenDNNV=args.hiddenDNNV,
        inputDimV=args.inputDimV,
        hiddenDimV=args.hiddenDimV,
        d_word_emb=args.d_word_emb,
        beam_size=args.beam_size,
        birnnV=args.birnnV,
        outdpV=args.outdpV,
        p_size=args.p_size,
        g_size=args.g_size,
        d_embed=args.d_embed,
        d_hidden=args.d_hidden,
        embDO=args.embDO,
        Nh=Nh,
    )

    outs = model(
        x=input_var,
        x_lengths=x_lengths,
        grapheme=graphemeTensor.detach(),
        phoneme=phonemeTensor[:-1].detach(),
        istrain=False,
        BE_loc=False,
        adaptInit=False,
        adaptEnc=False,
    )
