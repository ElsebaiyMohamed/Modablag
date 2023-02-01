import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import Tensor
import numpy as np
from numpy import ndarray
from typing import Union, List
from dataclasses import dataclass
from utils.posEncoders import SinuSoidal

from utils.encoder import EncoderLayer
from utils.decoder import DecoderLayer

class BTConfig(dataclass):
        
    src_vocublary: int
    src_emp: int
    src_pos_size: int
    enc_key: int 
    enc_heads: int
    enc_size: int = 1
    
    trgt_vocublary: int
    trgt_emp: int
    trgt_pos_size: int
    dec_key: int 
    dec_heads: int
    dec_size: int = 1
    



class BTransformer(keras.Model):
    '''
    the Transformer Module from the paper. consist of Encoder Module & Decoder Module. This class also Implement
    Greedy and Beam search decoding.
    '''
    def __init__(self, config, task='multi'):
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
        self.config: BTConfig = config
        self.task: str = task
        
        self.src_emp = SinuSoidal(self.config.src_vocublary, self.config.src_emp, self.config.src_pos_size)
        self.enc = keras.Sequential([EncoderLayer(self.config.enc_key, self.config.enc_heads, self.config.src_emp) 
                                     for _ in range(self.config.enc_size)])
        
        self.trgt_emp = SinuSoidal(self.config.trgt_vocublary, self.config.trgt_emp, self.config.trgt_pos_size)
        self.dec = [DecoderLayer(self.config.dec_key, self.config.dec_heads, self.config.trgt_emp) for _ in range(self.config.dec_size)]
        
        if self.task == 'multi':
            self.script_layer = keras.layers.Dense(self.config.src_vocublary)
            self.translate_layer = keras.layers.Dense(self.config.trgt_vocublary)
        elif self.task == 'en':      
            self.script_layer = keras.layers.Dense(self.config.src_vocublary)
        else:
            self.translate_layer = keras.layers.Dense(self.config.trgt_vocublary)
        
        
    def call(self, inputs, training=False, teacher=False):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        if teacher:
            src, targt = inputs; del inputs
            src_emp = self.src_emp(src); del src
            src_context = self.enc(src_emp)
            
            if self.task == 'multi':
                return self.teacher_force((src_emp, targt), src_context)
                
            elif self.task == 'script':
                return self.teacher_force(src_emp, src_context)
            else:
                return self.teacher_force(targt, src_context)
        
        src_emp = self.src_emp(src); del src
        src_context = self.enc(src_emp); del src_emp
        return self.greedy_decoding(src_context)

                
        
    def greedy_decoding(self, context, end_token=2):
        start = np.zeros((context.shape[0], 1))
        
        # print('start decoding')
        if self.task == 'multi':
            script_word = start
            translate_word = start
            for _ in range(self.config.src_pos_size):
                script_word = self.src_emp(script_word)
                translate_word = self.trgt_emp(translate_word)
                mixed = self.fussion(script_word, translate_word)
                mixed = self.dec(mixed, context)
                script_word = self.script_layer(mixed)
                translate_word = self.translate_layer(mixed)
                # print('shape of logits', next_word.shape)
                script_word = np.argmax(script_word, 2)
                # print('shape of argmax', next_word.shape)
                script_word = np.concatenate((start, script_word), axis=1)
                # print('merge shape', next_word.shape)
                script_word[script_word[:, -2] == end_token] = end_token
                translate_word = np.argmax(translate_word, 2)
                # print('shape of argmax', next_word.shape)
                translate_word = np.concatenate((start, translate_word), axis=1)
                # print('merge shape', next_word.shape)
                translate_word[translate_word[:, -2] == end_token] = end_token

                
                
                if (script_word[script_word[:, -2] == end_token].shape[0] == script_word.shape[0] and 
                   translate_word[translate_word[:, -2] == end_token].shape[0] == translate_word.shape[0]):
                    return script_word, translate_word
                            
            return script_word, translate_word
            
        elif self.task == 'script':
            script_word = start
            
            for _ in range(self.config.src_pos_size):
                script_word = self.src_emp(script_word)
                script_word = self.dec(script_word, context)
                script_word = self.script_layer(script_word)
                # print('shape of logits', next_word.shape)
                script_word = np.argmax(script_word, 2)
                # print('shape of argmax', next_word.shape)
                script_word = np.concatenate((start, script_word), axis=1)
                # print('merge shape', next_word.shape)
                script_word[script_word[:, -2] == end_token] = end_token
                
                
                if script_word[script_word[:, -2] == end_token].shape[0] == script_word.shape[0]:
                    return script_word
                            
            return script_word
        else:
            translate_word = start
            for _ in range(self.config.trgt_pos_size):
                translate_word = self.trgt_emp(translate_word)
                translate_word = self.dec(translate_word, context)
                translate_word = self.translate_layer(translate_word)
                # print('shape of logits', next_word.shape)
                translate_word = np.argmax(translate_word, 2)
                # print('shape of argmax', next_word.shape)
                translate_word = np.concatenate((start, translate_word), axis=1)
                # print('merge shape', next_word.shape)
                translate_word[translate_word[:, -2] == end_token] = end_token
                
                
                if translate_word[translate_word[:, -2] == end_token].shape[0] == translate_word.shape[0]:
                    return translate_word
                            
            return translate_word
            
        for _ in range(self.max_len):
            
            next_word = self.dec(next_word, context)
            next_word = self.final_layer(next_word)
            # print('shape of logits', next_word.shape)
            next_word = np.argmax(next_word, 2)
            # print('shape of argmax', next_word.shape)
            next_word = np.concatenate((start, next_word), axis=1)
            # print('merge shape', next_word.shape)
            next_word[next_word[:, -2] == end_token] = end_token
            
            
            if next_word[next_word[:, -2] == end_token].shape[0] == next_word.shape[0]:
                return next_word
                        
        return next_word
    
    def beam_search_decoding(self, context):
        pass
    
    def teacher_force(self, inputs, context):
        if self.task == 'multi':
            src, trgt = inputs; del inputs
            trgt = self.trgt_emp(trgt)
            trgt = self.fussion(src, trgt); del src
            trgt = self.dec(trgt)
            return self.script_layer(trgt), self.translate_layer(trgt)
            
        elif self.task == 'script':
            inputs = self.dec(inputs, context)
            return self.script_layer(inputs)
        
        else:
            inputs = self.trgt_emp(inputs)
            inputs = self.dec(inputs, context)
            return self.translate_layer(inputs)
            
            
            
            
            
            
            
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