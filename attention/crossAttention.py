from tensorflow.keras import layers
from tensorflow import __version__
from selfAttention import BaseAttention


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
                                            query= x,
                                            key= context,
                                            value= context,
                                            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
if __name__ == '__main__':
    print(__version__)