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
from utils.encoders import EncoderLayer
from utils.decoders import DecoderLayer


class Encoder(keras.Model):
    #TODO -> docstring
    def __init__(self, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1) -> None:
        #TODO -> docstring
        super(Encoder, self).__init__()
        self.pos_encoder = SinuSoidal(vocab_size, emp_dim, max_sent_lenght)
        self.enc = keras.Sequential([EncoderLayer(key_dim, n_heads, emp_dim) 
                                     for _ in range(n_layers)])
        
    def call(self, x, training=False, **kwargs):
        #TODO -> docstring
        x = self.pos_encoder(x, training=training)
        return self.enc(x, training=training)
    
    
    
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
    

class BTransformer(keras.Model):
    def __init__(self, e_config: tuple, d_config: tuple) -> None:
        super().__init__()
        self.max_len = e_config[2]
        self.enc = Encoder(*e_config)
        self.dec = Decoder(*d_config)
        
        self.final_layer = keras.layers.Dense(d_config[0])
        
    def call(self, src, training=False):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        if training:
            src, target = src

            src = self.enc(src)  # (batch_size, context_len, d_model)

            src = self.dec(target, src)  # (batch_size, target_len, d_model)

            # Final linear layer output.
            return self.final_layer(src)  # (batch_size, target_len, target_vocab_size)
        
        
        # works only with one sentence 
        
        target = self.enc(src)  
        sent = [[1]]
        src = np.array([[1]])
        while src.shape[1] < self.max_len:
            src = self.dec(src, target)
            src = self.final_layer(src)
            src = np.argmax(src, 2)
            print(len(sent[0]))
            sent[0].append(src[0, -1])
            print(sent)
            src = np.array(sent)
            
            if sent[0][-1] == 1:
                return sent[0]
            
        return sent[0]
            
            
if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices 
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    
    
    # dummy = np.random.randn(2, 3)
    # f = Encoder(10000, 5, 2400, 50, 8, 2)
    # print(f(dummy)) #, causal_mask=True
    
    dummy = np.random.randn(1, 12)
    dummy2 = np.random.randn(2, 5)
    context = np.random.randn(2, 12, 50)
    
    #, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1
    #, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1
    config_e = (100, 50, 2000, 25, 8, 3)
    config_d = (200, 50, 2000, 25, 8, 3)
    f = Decoder(*config_d)
    
    
    ff = Encoder(*config_e)
    # print(ff(dummy))
    # print(f(dummy2, ff(dummy)))
    x = BTransformer(config_e, config_d)
    print(x(dummy)) 