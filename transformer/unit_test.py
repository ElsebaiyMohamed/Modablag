from tqdm import tqdm
from utils import *
from posEncoders import SinuSoidal
import numpy as np

np.random.seed(1)


def create_dummy(voc_size, batch, time_step, feature=None):
    if feature is None:
        return np.random.randint(0, voc_size, (batch, time_step))
    return np.random.randn(batch, time_step, feature)
    
    
def test_SinuSoidal(): #voc_size: int, output_dim: int, max_sent_lenght: int, mask_zero
    
    for v in tqdm(np.arange(1_000, 70_000, 1_000)):          #voc_size: int,
        for o in np.arange(100, 700, 100):  #output_dim: int,
            for m in np.arange(0, 2_000, 200):      #max_sent_lenght: int,
                dummy = create_dummy(v, 128, m)
                # for ms in mask_zero:
                opj = SinuSoidal(v, o, m)
                try:
                    opj(dummy)
                    # opj.compute_mask(dummy)
                except Exception as e:
                    print(f'''
                            @params:
                                    voc_size = {v} 
                                    output_dim = {o}
                                    max_sent_lenght = {m}
                                    dummy_shape     = {dummy.shape}
                            ''')
                    break
                    # print(e)
            

def test_self_attention():
    pass

def test_feed_forward():
    pass

def test_enc_layer():
    pass

def test_dec_layer():
    pass

def test_enc():
    pass

def test_dec():
    pass

def test_model():
    pass




if __name__ == '__main__':
    test_SinuSoidal()