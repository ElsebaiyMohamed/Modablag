import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers, Sequential
from attentions import SelfAttention
from tensorflow import float32, Tensor, is_tensor, convert_to_tensor
from numpy import ndarray
from typing import Union



class FeedForward(layers.Layer):
    def __init__(self, units: list, dropout=0.1):
        '''
        @params:
                units:   list.  Number of units at each dense layer, make sure that the number of
                                      units at the last layer same as the inputs to the layer.
                dropout: float. dropout ratio before Add&Normlize layer
        '''
        super(FeedForward, self).__init__()
        self.seq = Sequential([layers.Dense(u) for u in units])
        self.seq.add(layers.Dropout(dropout))
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x: Union[Tensor, ndarray], training: bool=False)->Tensor:
        '''
        @params:
                x       : 2D float32 matrix.
                training: bool.               behave in training or inference mode
        @return:
                2D float32 matrix with the same shape as the inputs
        '''
        if not is_tensor(x):
            x = convert_to_tensor(x, dtype=float32)
        
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x, training=training) 
        return x
    
    
class EncoderLayer(layers.Layer):
    def __init__(self, key_dim, num_heads, output_shape, dropout=0.1):
        super().__init__()

        self.self_attention = SelfAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            dropout=dropout,
                                            output_shape=output_shape)

        self.ffn = FeedForward([key_dim, output_shape])

    def call(self, x, training=False, **kwargs):
        x = self.self_attention(query=x, key=x, value=x, training=training, **kwargs)
        x = self.ffn(x, training=training)
        return x

if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices 
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    
    
    dummy = np.random.randn(2, 3, 5)
    f = EncoderLayer(10, 50, 5)
    print(f(dummy))