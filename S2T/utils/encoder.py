import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from selfAttention import Former
    from feedForward import FForward
except ImportError:
    from .selfAttention import Former
    from .feedForward import FForward
from tensorflow.keras.layers import Layer

from tensorflow import Tensor
from numpy import ndarray
from typing import Union, List



class FEncoderLayer(Layer):
    '''
    the encoder structure from the paper. consist of SelfAttention & FeedForward
    '''
    def __init__(self, key_dim: int, num_heads: int, output_shape: int=None, dropout=0.1):
        '''
        num_heads:     int.    Number of attention heads. 
        key_dim:       int.    Size of each attention head for query and key.
        dropout=0.1:   float.  Dropout probability.
        output_shape:  int.    The expected shape of an output tensor, besides the batch and sequence dims. 
                                   If not specified, projects back to the key feature dim.
        '''        

        super().__init__()
        if output_shape is None:
            output_shape = key_dim
        self.self_attention = Former(num_heads=num_heads,
                                    key_dim=key_dim,
                                    dropout=dropout,
                                    output_shape=output_shape)

        self.ffn = FForward([key_dim+50, output_shape])

    def call(self, x: Union[Tensor, ndarray, List], training: bool=False, **kwargs):
        x = self.self_attention(query=x, key=x, value=x, training=training, **kwargs)
        x = self.ffn(x, training=training)
        return x
