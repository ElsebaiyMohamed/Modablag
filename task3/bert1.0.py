import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import Tensor
import numpy as np
from numpy import ndarray
from typing import Union, List

from transformer.baseT import Encoder


class Bert(keras.Model):
    def __init__(self, e_config):
        '''@params:
            e_config:              Tuple
                - vocab_size:      int.  Size of the soruce lang vocabulary, i.e. maximum integer index + 1.
                - emp_dim:         int.  Dimension of the dense embedding.
                - max_sent_lenght: int.  Max number of tokens in as sentence that the model will 
                                            deal with it during inference.
                - num_heads:       int.  Number of attention heads. 
                - key_dim:         int.  Size of each attention head for query and key.
                - n_layers:        int.  Number of stacked encoders.
        '''
        super().__init__()
        
        self.enc = Encoder(*e_config)
        
    def forward(self, x, masking, training=False):
        if training:00
        

