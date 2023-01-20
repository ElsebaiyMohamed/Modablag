'''
@TODO:
      add type hint to each function 
      add docstring 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import float32, Tensor, is_tensor, convert_to_tensor
from numpy import ndarray
from typing import Union, List

from utils.posEncoders import SinuSoidal
from utils.decoders import DecoderLayer

class Decoder(keras.Model):
    #TODO -> docstring
    def __init__(self, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1) -> None:
        #TODO -> docstring
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.pos_encoder = SinuSoidal(vocab_size, emp_dim, max_sent_lenght)
        self.dec = [DecoderLayer(key_dim, n_heads, emp_dim) for _ in range(n_layers)]
        
    def call(self, x, context, training=False, **kwargs):
        #TODO -> docstring
        x = self.pos_encoder(x, training=training)
        for i in range(self.n_layers):
            x = self.dec[i](x, context, training=training)
        return x
    
    
if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices 
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    
    
    dummy = np.random.randn(2, 3)
    context = np.random.randn(2, 3, 10)
    f = Decoder(10000, 5, 2400, 50, 8, 2)
    print(f(dummy, context)) #, causal_mask=True