import numpy as np
from hepler import *
from dataclasses import dataclass
import gc

@dataclass
class Config:
    en_file: str
    ar_file: str
    yaml_path: str
    en_dict: str
    ar_dict: str 
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
    
    en_dict = load_dict(c.en_dict)
    ar_dict = load_dict(c.ar_dict)
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
        en    = np.zeros((c.batch_size, c.max_en), np.int32) 
        ar    = np.zeros((c.batch_size, c.max_ar), np.int32) 
        waves = np.zeros((c.batch_size, c.max_wave), np.float32) 
        
        
        while i < c.batch_size:
            
            print(end='\r')
            print(i,':', c.batch_size, end='')
            try:
                en_text = next(en_data)
                
                ar_text = next(ar_data)
                
                (duration, offset), wave_path = next(yaml_data)
                wave = get_form_wave(offset, duration, join(c.wave_folder, wave_path), c.sr)
                
                en_text = english_preprocess(en_text)
                en_text = numprize_text(en_text, en_dict)  
                ar_text = arabic_preprocess(ar_text) 
                ar_text = numprize_text(ar_text, ar_dict) 
                
                if en_text.shape[0] > c.max_en or ar_text.shape[0] > c.max_ar or wave.shape[0] > c.max_wave:
                    continue

                if len(en_text) < c.min_en:
                    en_text = repeat(en_text, c.min_en, c.max_en)

                if len(ar_text) < c.min_ar:
                    ar_text = repeat(ar_text, c.min_ar, c.max_ar)

                if wave.shape[0] < c.min_wave:
                    wave = repeat_wave(wave, c.min_wave, c.max_wave)
                
                en_text = padd(en_text, c.max_en)  
                ar_text = padd(ar_text, c.max_ar) 
                wave = padd(wave, c.max_wave) 
                en[i] = en_text
                ar[i] = ar_text
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
            fp.create_dataset('ar', dtype=np.int32, data=ar)
            fp.create_dataset('waves', dtype=np.float32, data=waves)
            del en, ar, waves
            
            print('done')
            fp.close()
        
        gc.collect(1)




if __name__ == '__main__':
    config = Config(r"D:\Study\GitHub\dev\text\dev.en", r"D:\Study\GitHub\dev\text\dev.ar",
                    r"D:\Study\GitHub\dev\text\dev.yaml.yaml", r"D:\Study\GitHub\dev\vocublary\en.txt",
                    r"D:\Study\GitHub\dev\vocublary\ar.txt", r"D:\Study\GitHub\dev\wav", r'D:\Study\GitHub\dev\btached',
                    16000, 5, 500, 5, 7, 16000, 25, 25, 16000*20)
    
    create_batch(config)
    
    