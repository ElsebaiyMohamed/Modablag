import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Layer, LayerNormalization, Add, Dropout, Dense)

from tensorflow import is_tensor, convert_to_tensor, float32, Tensor
from numpy import ndarray
from typing import Union, Tuple


class FForward(Layer):
    '''
    FeedForward part in the encoder from the paper.
    '''
    def __init__(self, units: Tuple, dropout=0.1):
        '''
        @params:
                units:   Tuple.  Number of units at each dense layer, make sure that the number of
                                      units at the last layer same as the inputs to the layer.
                dropout: float. dropout ratio before Add&Normlize layer
        '''
        super().__init__()
        self.ffn = Sequential([Dense(units[0]), Dropout(dropout/1.3), 
                               Dense(units[1]), Dropout(dropout)])
        self.add = Add()
        self.layer_norm = LayerNormalization()

    def call(self, x: Union[Tensor, ndarray], training: bool=False):
        '''
        @params:
                x       : 2D float32 matrix.
                training: bool.               behave in training or inference mode
        @return:
                2D float32 matrix with the same shape as the inputs
        '''
        if not is_tensor(x):
            x = convert_to_tensor(x, dtype=float32)
        x = self.add([x, self.ffn(x)])
        x = self.layer_norm(x, training=training) 
        return x
