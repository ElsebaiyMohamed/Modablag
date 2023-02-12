import numpy as np
from hepler import *
from dataclasses import dataclass
import gc

@dataclass
class Config:
    en_file: str
    ar_file: str
    yaml_path: str
    # en_token: str
    # ar_token: str 
    wave_folder: str
    save_folder: str
    sr: int
    n_batch: int
    batch_size: int
    min_ar: int
    min_en: int
    min_wave: int
    max_ar: int
    max_en: int
    max_wave: int

def create_batch(c: Config):
    
    # en_token = TokenHandler(c.en_token)
    # ar_token = TokenHandler(c.ar_token)
    os.makedirs(c.save_folder, exist_ok=True)
    
    en_data   = get_en(c.en_file)
    
    ar_data   = get_ar(c.ar_file)
    yaml_data = get_ymal(c.yaml_path)
    f = False
    gc.enable()
    gc.set_threshold(3, 2, 1)
    gc.collect()
    for j in range(c.n_batch):

        i = 0
        en    = np.zeros((c.batch_size), np.string_) 
        ar    = np.zeros((c.batch_size),  np.string_)
         
        waves = np.zeros((c.batch_size, c.max_wave), np.float32) 
        
        
        while i < c.batch_size:
            
            print(end='\r')
            print(i,':', c.batch_size, end='')
            try:
                en_text = next(en_data)
                
                ar_text = next(ar_data)
                
                (duration, offset), wave_path = next(yaml_data)
                wave = get_form_wave(offset, duration, join(c.wave_folder, wave_path), c.sr)
                
                if len(en_text.split()) > c.max_en or len(ar_text.split()) > c.max_ar or \
                        wave.shape[0] > c.max_wave or  wave.shape[0] < c.min_wave:
                    continue
                
                en_text = english_preprocess(en_text)
                # en_text =  en_token.enocde_line(en_text)[0]
                ar_text = arabic_preprocess(ar_text) 
                # ar_text = ar_token.enocde_line(ar_text)[0] 
                
                

                if len(en_text) < c.min_en:
                    en_text = ' '.join(repeat(en_text.split(), c.min_en))

                if len(ar_text) < c.min_ar:
                    ar_text = ' '.join(repeat(ar_text.split(), c.min_ar))                
                
                wave = padd(wave, c.max_wave) 
                en[i] = en_text
                ar[i] = ar_text.encode()
                waves[i] = wave
                i += 1
            except StopIteration as e:
                print(' ',e)
                f = True
                i = c.batch_size
            
        if f:
            print('batch is droped because of its size')
            gc.collect()
            break
            
        
        print('  start save')
        with h5py.File(join(c.save_folder, f'batch_{j}.h5'), 'w') as fp:
            
            fp.create_dataset('en', data=en)
            fp.create_dataset('ar', data=ar)
            fp.create_dataset('waves',data=waves)
            del en, ar, waves
            
            print('done')
            fp.close()
        
        gc.collect(1)




if __name__ == '__main__':
    config = Config(r"D:\Study\GitHub\dev\text\dev.en", r"D:\Study\GitHub\dev\text\dev.ar",
                    r"D:\Study\GitHub\dev\text\dev.yaml.yaml", #r"D:\Study\GitHub\dev\vocublary\en.txt", r"D:\Study\GitHub\dev\vocublary\ar.txt"
                    r"D:\Study\GitHub\dev\wav", r'D:\Study\GitHub\dev\btached',
                    16000, 5, 500, 5, 7, 16000, 25, 25, 16000*20)
    
    create_batch(config)
    
    