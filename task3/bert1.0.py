import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import Tensor
import numpy as np
from numpy import ndarray
from typing import Union, List
from utils.posEncoders import SinuSoidal
from utils.encoder import EncoderLayer


class Bert(keras.Model):
    def __init__(self, vocab_size: int, emp_dim: int, max_sent_lenght: int,
                 key_dim: int, n_heads: int, n_layers: int=1):
        '''
        @params:
                vocab_size:        int.  Size of the vocabulary, i.e. maximum integer index + 1.
                emp_dim:         int.  Dimension of the dense embedding.
                max_sent_lenght: int.  Max number of tokens in as sentence that the model will 
                                          deal with it during inference.
                num_heads:       int.  Number of attention heads. 
                key_dim:         int.  Size of each attention head for query and key.
                n_layers:        int.  Number of stacked encoders.
        '''
    
        super().__init__()
        self.pos_encoder = SinuSoidal(vocab_size, emp_dim, max_sent_lenght)
        self.enc = keras.Sequential([EncoderLayer(key_dim, n_heads, emp_dim) 
                                     for _ in range(n_layers)])
        
        
    def call(self, x, masking, training=False):
        if training:00
        

