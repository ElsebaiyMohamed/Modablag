import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from selfAttention import Former
from feedForward import FForward
from tensorflow.keras.layers import Layer

from tensorflow import Tensor
from numpy import ndarray
from typing import Union, List

class DecoderLayer(Layer):
    '''
    the decoder structure from the paper. consist of SelfAttention & CrossAttention & FeedForward
    '''
    def __init__(self, key_dim:int , num_heads: int, output_shape: int, dropout=0.1):
        '''
        num_heads:     int.    Number of attention heads. 
        key_dim:       int.    Size of each attention head for query and key.
        dropout=0.1:   float.  Dropout probability.
        output_shape:  int.    The expected shape of an output tensor, besides the batch and sequence dims. 
                                   If not specified, projects back to the key feature dim.
        ''' 
        super().__init__()

        self.self_attention  = Former(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)
        
        self.cross_attention = Former(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)

        self.ffn = FForward((key_dim, output_shape))
        

    def call(self, x: Union[Tensor, ndarray, List], context: Union[Tensor, ndarray, List], 
             training=False, causal_mask=True, **kwargs):

        x = self.self_attention(query=x, key=x, value=x, training=training, causal_mask=causal_mask, **kwargs)
        x = self.cross_attention(query=x, key=context, value=context, training=training, 
                                 causal_mask=causal_mask, **kwargs)
        return self.ffn(x, training=training)

  
  
        
