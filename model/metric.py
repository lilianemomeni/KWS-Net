import math
import torch
import numpy as np
from tqdm import tqdm


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)



def calcEER(Tar,Non,TarKeep='all',NonKeep='all'):
  #  import pdb; pdb.set_trace()
    if TarKeep=='all':
        TarKeep = np.asarray([True for i in range(0,Tar.shape[0])])
    if NonKeep=='all':
        NonKeep = np.asarray([True for i in range(0,Non.shape[0])])
  #  import pdb; pdb.set_trace()
    Tar = Tar[TarKeep]
    Non = Non[NonKeep]
    Tar_2 = [i.item() for i in Tar]
    Non_2 = [i.item() for i in Non]
    Mt = np.mean(Tar_2)
    Mn = np.mean(Non_2)
  #  print('Mean Tar: {}').format(Mt)
  #  print('Mean Non: {}').format(Mn)
    Ns = 500
    E = np.zeros((Ns,2))
    S = np.linspace(Mn,Mt,Ns)
    for s in range(0,Ns):
        E[s,0] = np.sum((Tar_2<S[s]).astype('float'))/Tar.shape[0]
        E[s,1] = np.sum((Non_2>S[s]).astype('float'))/Non.shape[0]
    I = np.argmin(np.abs(E[:,0] - E[:,1]))
    #print('EER: {}').format(np.mean(E[I,:]))
    #print('MDR: {}').format(E[I,0])
    #print('FAR: {}').format(E[I,1])
    #print('thr: {}').format(np.exp(S[I]))
    return np.mean(E[I,:]), E[I,0], E[I,1], np.exp(S[I])

def accuracy_orig(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def PrecRec(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().float()
    target = target.float()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    pred = pred.view(-1)
    Pred = pred.clone()
    target = target.view(-1)
    PR = torch.FloatTensor(2).fill_(0)
    N = torch.sum(Pred.mul_(target))
    sumPr = torch.sum(pred)
    PR[0] = N/sumPr if sumPr>0 else 1.0
    sumTar = torch.sum(target)
    PR[1] = N/sumTar if sumTar>0 else 1.0
    return PR

def calcPrRankMultiInstances(TarRank, NonRank, PrIn):
  #  import pdb; pdb.set_trace()
    N = len(PrIn)
    PrRank = np.zeros((len(TarRank),N))
  #  MnRank = np.zeros((len(TarRank),1))
    t = np.asarray(TarRank)
    n = np.asarray(NonRank)
   # import pdb; pdb.set_trace()
    for k in range(0,t.shape[0]):
        V = np.concatenate((t,n),axis=0)
        I = np.concatenate((np.ones(t.shape[0]),np.zeros(n.shape[0])),axis=0)
       # Is = I 
        Is = I[V.argsort()[::-1]]
        for i in range(0,N):
            PrRank[k][i] = np.any(Is[:PrIn[i]]==1).astype('int32')
        if k==0:
         #remove k = 0
        #where Is is ==1 and take the smallest value 
          MeanRank = np.mean(np.argwhere(Is.astype('int32')).astype('float'))/I.shape[0]
      #  import pdb; pdb.set_trace()
      #  MnRank[k] = np.argwhere(Is.astype('int32')).astype('float')[0]/I.shape[0]
        t = np.delete(t,np.argmax(t),0)
   # import pdb; pdb.set_trace()
   # MeanRank = np.mean(MnRank)
    return PrRank, MeanRank
def calcMeanRank(TarRank, NonRank):  
    MeanRank = np.zeros((len(TarRank),1))
    t = np.asarray(TarRank)
    n = np.asarray(NonRank)
  #  import pdb; pdb.set_trace()
    V = np.concatenate((t,n),axis=0)
    I = np.concatenate((np.ones(t.shape[0]),np.zeros(n.shape[0])),axis=0)
    Is = I[V.argsort()[::-1]]
    m = 0.0
    k = 0
    for c in range(0,Is.shape[0]):        
        if Is[c] == 1:
            MeanRank[k] = m/(n.shape[0]+1)
            k+=1
        else:
            m+=1
        if k>len(TarRank):
            break
    return np.mean(MeanRank),MeanRank   
def calcPrRank(TarRank, NonRank, PrIn):
    N = len(PrIn)
    PrRank = np.zeros((N,))
    t = np.asarray(TarRank)
    n = np.asarray(NonRank)
    V = np.concatenate((t,n),axis=0)
    I = np.concatenate((np.ones(t.shape[0]),np.zeros(n.shape[0])),axis=0)
    Is = I[V.argsort()[::-1]]
    for i in range(0,N):
        PrRank[i] = np.any(Is[:PrIn[i]]==1).astype('int32')
    MeanRank = np.mean(np.argwhere(Is.astype('int32')).astype('float'))/I.shape[0]
    return PrRank, MeanRank



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a sGktandard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class APMeter(Meter):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.

    NOTE: This code is from torchnet.tnt
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())
        self.last_precision = None

    def add(self, output, target, weight=None):
        """Add a new observation
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                (eg: a row [0, 1, 0, 1] indicates that the example is
                associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        if hasattr(torch, "arange"):
            rg = torch.arange(1, self.scores.size(0) + 1).float()
        else:
            rg = torch.range(1, self.scores.size(0)).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)

            # store the precision curve
            self.last_precision = precision
        return ap


class ClassErrorMeter(Meter):
    def __init__(self, topk=[1, 5, 10, 50], accuracy=True):
        super(ClassErrorMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = np.atleast_1d(target.cpu().squeeze().numpy())
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        if np.ndim(output) == 1:
            output = output[np.newaxis]
        else:
            assert np.ndim(output) == 2, \
                'wrong output size (1D or 2D expected)'
            assert np.ndim(target) == 1, \
                'target and output do not match'
        assert target.shape[0] == output.shape[0], \
            'target and output do not match'
        topk = self.topk
        maxk = int(topk[-1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        correct = pred == target[:, np.newaxis].repeat(pred.shape[1], 1)

        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                'invalid k (this k was not provided at construction time)'
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]
