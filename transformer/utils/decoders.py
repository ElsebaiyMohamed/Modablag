'''
@TODO:
      add type hint to each function 
      add docstring 
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers, Sequential
try:
    from attentions import SelfAttention
    from encoders import FeedForward
except ImportError:
    from .attentions import SelfAttention
    from .encoders import FeedForward
from tensorflow import float32, Tensor, is_tensor, convert_to_tensor
from numpy import ndarray
from typing import Union, List



class DecoderLayer(layers.Layer):
    #TODO -> docstring
    def __init__(self, key_dim, num_heads, output_shape: int, dropout=0.1):
        #TODO -> docstring
        super().__init__()

        self.self_attention  = SelfAttention(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)
        
        self.cross_attention = SelfAttention(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)

        self.ffn = FeedForward([key_dim, output_shape])
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()

    def call(self, x, context, training=False, causal_mask=True, **kwargs):
        #TODO -> docstring
        x = self.self_attention(query=x, key=x, value=x, training=training, causal_mask=causal_mask, **kwargs)
        x = self.cross_attention(query=x, key=context, value=context, training=training, 
                                 causal_mask=causal_mask, **kwargs)
        return self.ffn(x, training=training)





if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices 
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    
    
    dummy = np.random.randn(2, 3, 5)
    context = np.random.randn(2, 3, 10)
    
    f = DecoderLayer(10, 50, 5)
    print(f(dummy, context)) #, causal_mask=True