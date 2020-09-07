import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModule(nn.Module):
    """
    Base module - it's primary purpose is to avoid linting errors on the forward
    function argument list :). This will only work with recent (> July 2019) pylint
    versions, since older versions did not handle variadics correctly.
    """
    @abstractmethod
    def forward(self, *inputs, **kwarags):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError


class BaseModel(BaseModule):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs, **kwarags):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
