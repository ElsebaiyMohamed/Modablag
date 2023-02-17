import re
import unicodedata
import os
import random
from os.path import join
from tqdm import tqdm
import numpy as np
import librosa
import h5py
from manager import TokenHandler
from dataclasses import dataclass
import gc



from pyarabic.araby import strip_tatweel, strip_lastharaka
from pyarabic.trans import normalize_digits


def digit_convertor(text):
    return normalize_digits(text, source='all', out='east')

def arabic_normlizer(text):
    text = digit_convertor(text)
    text = strip_tatweel(text)
    text = strip_lastharaka(text)
    return text

def unicode_to_ascii(s):
    '''
    Args:
        s : UniCode file
    Returns:
        ASCII file
    
    Converts the unicode file to ascii
    For each character:
                        there are two normal forms: normal form C and normal form D. 
                                                    - Normal form D (NFD) is also known as canonical decomposition,
                                                      and translates each character into its decomposed form. 
                                                      
                                                    - Normal form C (NFC) first applies a canonical decomposition, 
                                                    then composes pre-combined characters again.
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)) # if unicodedata.category(c) != 'Mn'# Mn ==> Mark, Nonspacing (التشكيل )

def arabic_preprocess(s):
    '''
    Args:
        s : A single sentence 
    '''
    s = arabic_normlizer(s)
    s = unicode_to_ascii(s.strip())

#         s = re.sub(r"([?.!,¿؟])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^أ-ۿ١-٩اءئ؟!.':,?_،إ]+", " ", s)
    s = s.rstrip().strip()
#     s = f'<start> {s} <end>'
    return s



def english_preprocess(s):
    
#    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z0-9?.'!:_,¿]+", " ", s)
    
    s = s.rstrip().strip()
#     s = f'<start> {s} <end>'
    return s

def get_en(file_path):
    with open(file_path, encoding='utf-8-sig') as f:
        for line in f:
            yield line.strip().strip('\n').strip()

def get_ar(file_path):
    with open(file_path, encoding='utf-8-sig') as f:
        for line in f:
            yield line.strip().strip('\n').strip()

def get_ymal(file_path):
    
    def fun(text):
        _, value = text.split(':')
        
        value = value.strip()
        return float(value)
    
    
    with open(file_path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip('- {').strip('}\n')
            line = line.split(',')
            times = tuple(map(fun, line[:2]))
            line = line[-1].split(':')[-1].strip()
            yield times, line


def get_form_wave(offset, duration, wave_path, sr):
    audio, sr = librosa.load(wave_path, sr=sr, offset=offset, duration=duration)
    return audio


# --------------------------------------------


def repeat(items, min_len, max_len, p=0.2):
    new_items = []
    while len(items) < min_len:
        for i in items:
            r = random.random()
            if r < p:
                new_items.append(i)
                new_items.append(i)
            else:
                new_items.append(i)
                
        items = new_items[:]
        new_items.clear()
    max_len = max_len if max_len < len(items) else len(items)
    return items[:max_len]
            



def padd(items, max_len, pad=0):
    
    return np.pad(items, (0, np.max([0, max_len-len(items)])), 'constant', constant_values=(0, pad))

@dataclass
class Config:
    en_file: str
    ar_file: str
    yaml_path: str
    en_token: str
    ar_token: str 
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
    
    en_token = TokenHandler(c.en_token)
    ar_token = TokenHandler(c.ar_token)
    os.makedirs(c.save_folder, exist_ok=True)
    
    en_data   = get_en(c.en_file)
    
    ar_data   = get_ar(c.ar_file)
    yaml_data = get_ymal(c.yaml_path)
    f = False
    gc.enable()
    gc.set_threshold(3, 2, 1)
    gc.collect()
    for j in tqdm(range(c.n_batch), desc='Batch Counter', unit='record'):

        i = 0
        en    = [] 
        ar    = []
         
        waves = np.zeros((c.batch_size, c.max_wave), np.float32) 
        
        
        while i < c.batch_size:
            
            # print(end='\r')
            # print(i,':', c.batch_size, end='')
            try:
                en_text = next(en_data)
                
                ar_text = next(ar_data)
                
                (duration, offset), wave_path = next(yaml_data)
                wave = get_form_wave(offset, duration, join(c.wave_folder, wave_path), c.sr)
                
                if len(en_text.split()) > c.max_en or len(ar_text.split()) > c.max_ar or \
                        wave.shape[0] > c.max_wave :
                    continue
                
                en_text = english_preprocess(en_text)
                # en_text =  en_token.enocde_line(en_text)[0]
                ar_text = arabic_preprocess(ar_text) 
                # ar_text = ar_token.enocde_line(ar_text)[0] 
                
                

                if len(en_text.split()) < c.min_en:
                    en_text = ' '.join(repeat(en_text.split(), c.min_en, c.max_en))

                if len(ar_text.split()) < c.min_ar:
                    ar_text = ' '.join(repeat(ar_text.split(), c.min_ar, c.max_ar))                
                
                wave = padd(wave, c.max_wave) 
                en.append(en_text)
                ar.append(ar_text)
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

        en = en_token.encode_batch(en)
        
        en = np.array(en, np.int32)
        ar = ar_token.encode_batch(ar)
        ar = np.array(ar, np.int32)

        with h5py.File(join(c.save_folder, f'batch_{j}.h5'), 'w') as fp:
            
            fp.create_dataset('en', data=en)
            fp.create_dataset('ar', data=ar)
            fp.create_dataset('waves',data=waves)
            del en, ar, waves
            fp.close()
        
        gc.collect(1)




if __name__ == '__main__':
    config = Config(r"D:\Study\GitHub\dev\text\dev.en", 
                    r"D:\Study\GitHub\dev\text\dev.ar",
                    r"D:\Study\GitHub\dev\text\dev.yaml", 
                    r"D:\Study\GitHub\dev\en_tokenizer.json", 
                    r"D:\Study\GitHub\dev\tokens\ar_tokenizer.json",
                    r"D:\Study\GitHub\dev\wav", 
                    r'D:\Study\GitHub\dev\btached',
                    
                    16000, 100, 5000, 10, 10, 16000*3, 50, 45, 16000*15)
    
    create_batch(config)
    