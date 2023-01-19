from tensorflow.keras import layers
from tensorflow import __version__
from selfAttention import BaseAttention


class MaskedSelfAttention(BaseAttention):
    def call(self, x, mask=None):
            
        attn_output = self.mha(
                                query=x,
                                value=x,
                                key=x,
                                attention_mask=mask)#

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
if __name__ == '__main__':
    print(__version__)