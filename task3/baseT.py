'''
multi task vanila transformer.
can work on multi task or single task specified on creation of object: task = "multi", "script", "translate" 
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow import Tensor
import numpy as np
from numpy import ndarray
from typing import Union, List
from dataclasses import dataclass

try:
    from utils.posEncoders import SinuSoidal
    from utils.encoder import FEncoderLayer
    from utils.decoder import DecoderLayer
except ImportError:
    from .utils.posEncoders import SinuSoidal
    from .utils.encoder import FEncoderLayer
    from .utils.decoder import DecoderLayer
    
@dataclass
class BTConfig:
    max_seq_size: int
    heads_vocublary: List
    heads_emp: int
    enc_key: int 
    enc_heads: int
    
    dec_key: int 
    dec_heads: int
    
    enc_size: int = 1
    dec_size: int = 1
    


class OutputHead(keras.layers.Layer):
    def __init__(self, projection,  units=[], name=None):
        super().__init__()
        if name is None:
            self.name = 'untitled_Head'
        else:
            self.name = name
            
        layers = [keras.layers.Dense(u) for u in units]
        layers.append(keras.layers.Dense(projection))
        self.head = keras.Sequential(layers)
        
    def call(self, x, training=False):
        return self.head(x)
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

        self.heads_emp:dict = dict()
        for i, emp in enumerate(self.config.heads_emp):
            self.heads_emp[f'emp_{i}'] = keras.layers.Embedding(self.config.heads_vocublary[i][1], emp)

        self.pos = SinuSoidal()
        self.enc = keras.Sequential([FEncoderLayer(self.config.enc_key, self.config.enc_heads) 
                                     for _ in range(self.config.enc_size)])
        
        self.dec = [DecoderLayer(self.config.dec_key, self.config.dec_heads) 
                    for _ in range(self.config.dec_size)]
        self.heads:dict = dict()
        for i, name, vouc in enumerate(self.config.heads_vocublary):
            self.heads[name] = OutputHead(projection=vouc, name=name)
        
        
    def call(self, inputs, training=False):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        if training:
            src, targets = inputs; del inputs
            
            src = self.pos(src)
            
            src_context = self.enc(src, training=training)
            
            if self.task == 'multi':
                target = self.trgt_emp(target, training=training)
                target = self.pos(target)
                
                for dlayer in self.dec:
                    src = dlayer(src, src_context, training=training)
                    
                    target = dlayer(target, src_context, training=training)
                    
                src = self.script_layer(src, training=training)
                target = self.translate_layer(target, training=training)
                return src, target
        
            elif self.task == 'script':
                for dlayer in self.dec:
                    src = dlayer(src, src_context, training=training)
                src = self.script_layer(src, training=training)
                return src
            else:
                target = self.trgt_emp(target, training=training)
                target = self.pos(target)
                for dlayer in self.dec:
                    target = dlayer(target, src_context, training=training)
                target = self.translate_layer(target, training=training)
                return target
        
        inputs = self.src_emp(inputs, training=training)
        inputs = self.pos(inputs)
            
        src_context = self.enc(inputs, training=training); del inputs
        if self.task == 'multi':
            return self.greedy_script(src_context), self.greedy_translation(src_context)
        elif self.task == 'script':
            return self.greedy_script(src_context)
        else:
            return self.greedy_translation(src_context)
            
    
    def greedy_script(self, context, end_token=2, training=False):
        start = np.zeros((context.shape[0], 1))
        next_word = start
        for _ in range(self.config.max_seq_size):
            next_word = self.src_emp(next_word, training=training)
            next_word = self.pos(next_word)
            for dlayer in self.dec:
                next_word = dlayer(next_word, context, training=training)
            next_word = self.script_layer(next_word)
            # print('shape of logits', next_word.shape)
            next_word = np.argmax(next_word, 2)
            # print('shape of argmax', next_word.shape)
            next_word = np.concatenate((start, next_word), axis=1)
            # print('merge shape', next_word.shape)
            next_word[next_word[:, -2] == end_token] = end_token
            
            
            if next_word[next_word[:, -2] == end_token].shape[0] == next_word.shape[0]:
                return next_word
        return next_word
        
    def greedy_translation(self, context, end_token=2, training=False):
        start = np.zeros((context.shape[0], 1))
        next_word = start
        for _ in range(self.config.max_seq_size):
            next_word = self.trgt_emp(next_word, training=training)
            next_word = self.pos(next_word)
            for dlayer in self.dec:
                next_word = dlayer(next_word, context, training=training)
            next_word = self.translate_layer(next_word)
            # print('shape of logits', next_word.shape)
            next_word = np.argmax(next_word, 2)
            # print('shape of argmax', next_word.shape)
            next_word = np.concatenate((start, next_word), axis=1)
            # print('merge shape', next_word.shape)
            next_word[next_word[:, -2] == end_token] = end_token
            
            
            if next_word[next_word[:, -2] == end_token].shape[0] == next_word.shape[0]:
                return next_word
                        
        return next_word
    
    
    
    def beam_search_script(self, context):
        pass
    
    def beam_search_translation(self, context):
        pass
    
            
            
            
            
            
            
            
if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    np.random.seed(1)
    
    import tensorflow as tf
    d = np.random.randint(-10, 10, (3, 5))
    d2 = np.random.randint(-10, 10, (3, 2))
    config = BTConfig(max_seq_size=50, src_vocublary=1000, src_emp=512, enc_key=256, enc_heads=8, enc_size=10,
                      trgt_vocublary=2000, trgt_emp=512, dec_key=256, dec_heads=3, dec_size=5)
    
    x = BTransformer(config, task='script')
    print(x((d, d2), training=True)) 
    