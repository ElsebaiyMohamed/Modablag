from tensorflow.keras import layers
from tensorflow import __version__


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        
class SelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
                                query=x,
                                value=x,
                                key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

if __name__ == '__main__':
    print(__version__)