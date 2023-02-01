import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import Tensor
import numpy as np
from numpy import ndarray
from typing import Union, List

from utils.posEncoders import SinuSoidal

from utils.encoder import EncoderLayer
from utils.decoder import DecoderLayer


class Encoder(keras.Model):
    '''
    the encoder Module from the paper. consist of many stacked Encoder Layer
    '''
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
        
    def call(self, x: Union[Tensor, ndarray, List], training=False, **kwargs):
        x = self.pos_encoder(x, training=training)
        return self.enc(x, training=training)
    
    
    
class Decoder(keras.Model):
    '''
    the Decoder Module from the paper. consist of many stacked Decoder Layer
    '''
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
        self.n_layers = n_layers
        self.pos_encoder = SinuSoidal(vocab_size, emp_dim, max_sent_lenght)
        self.dec = [DecoderLayer(key_dim, n_heads, emp_dim) for _ in range(n_layers)]
        
    def call(self, x: Union[Tensor, ndarray, List], context: Union[Tensor, ndarray, List], 
             training=False, **kwargs):
        x = self.pos_encoder(x, training=training)
        
        for i in range(self.n_layers):
            x = self.dec[i](x, context, training=training)
        return x
    

class BTransformer(keras.Model):
    '''
    the Transformer Module from the paper. consist of Encoder Module & Decoder Module. This class also Implement
    Greedy and Beam search decoding.
    '''
    def __init__(self, e_config: tuple, d_config: tuple):
        '''
        @params:
            e_config:              Tuple
                - vocab_size:      int.  Size of the soruce lang vocabulary, i.e. maximum integer index + 1.
                - emp_dim:         int.  Dimension of the dense embedding.
                - max_sent_lenght: int.  Max number of tokens in as sentence that the model will 
                                            deal with it during inference.
                - num_heads:       int.  Number of attention heads. 
                - key_dim:         int.  Size of each attention head for query and key.
                - n_layers:        int.  Number of stacked encoders.
                
            d_config:              Tuple
                - vocab_size:        int.  Size of the target lang vocabulary, i.e. maximum integer index + 1.
                - emp_dim:         int.  Dimension of the dense embedding.
                - max_sent_lenght: int.  Max number of tokens in as sentence that the model will 
                                          deal with it during inference.
                - num_heads:       int.  Number of attention heads. 
                - key_dim:         int.  Size of each attention head for query and key.
                - n_layers:        int.  Number of stacked decoders.
        '''
        super().__init__()
        self.max_len = d_config[2]
        self.enc = Encoder(*e_config)
        self.dec = Decoder(*d_config)
        
        self.final_layer = keras.layers.Dense(d_config[0])
        
    def call(self, src, training=False):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        if training:
            src, target = src

            src = self.enc(src)         # (batch_size, context_len, d_model)

            src = self.dec(target, src)  # (batch_size, target_len, d_model)

            # Final linear layer output.
            return self.final_layer(src)  # (batch_size, target_len, target_vocab_size)
        
        
        context = self.enc(src)  
        del src
        
        return self.greedy_decoding(context, mask=1)
        
        
    def greedy_decoding(self, context, mask):
        start = np.zeros((context.shape[0], 1))
        next_word = start
        # print('start decoding')
        for _ in range(self.max_len):
            
            next_word = self.dec(next_word, context)
            next_word = self.final_layer(next_word)
            # print('shape of logits', next_word.shape)
            next_word = np.argmax(next_word, 2)
            # print('shape of argmax', next_word.shape)
            next_word = np.concatenate((start, next_word), axis=1)
            # print('merge shape', next_word.shape)
            next_word[next_word[:, -2] == mask] = mask
            
            
            if next_word[next_word[:, -2] == mask].shape[0] == next_word.shape[0]:
                return next_word
                        
        return next_word
    
    def beam_search_decoding(self, context):
        pass
            
if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    np.random.seed(1)
    # import tensorflow as tf
    # d = np.random.randint(0, 10, (3, 5, 4))
    # d2 = np.random.randint(0, 10, (3, 2))
    # d = tf.convert_to_tensor(d)
    # print(d2)
    # print()
    # print(np.argmax(d, 2))
    # print()
    # print(np.concatenate((d2, np.argmax(d, 2)), 1))
    
    
    # # dummy = np.random.randn(2, 3)
    # # f = Encoder(10000, 5, 2400, 50, 8, 2)
    # # print(f(dummy)) #, causal_mask=True
    
    # dummy = np.random.randn(1, 12)
    # dummy2 = np.random.randn(2, 5)
    # context = np.random.randn(2, 12, 50)
    
    # #, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1
    # #, vocab_size, emp_dim, max_sent_lenght, key_dim, n_heads, n_layers=1
    # config_e = (100, 50, 200, 25, 8, 3)
    # config_d = (200, 50, 200, 25, 8, 3)
    # f = Decoder(*config_d)
    
    
    # ff = Encoder(*config_e)
    # # print(ff(dummy))
    # # print(f(dummy2, ff(dummy)))
    # x = BTransformer(config_e, config_d)
    # print(x(dummy)) 