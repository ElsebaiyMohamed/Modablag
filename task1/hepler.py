import unicodedata
import re
import librosa
import numpy as np
import random

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
    '''
    Args:
        s : A single sentence 
    Returns:
        s : Single normalize sentence 
    
    Convert Unicode to ASCII
    Creating a space between a word and the punctuation following it
    eg: "he is a boy." => "he is a boy ." 
    Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    
    Replacing everything with space except (a-z, A-Z, ا-ي ".", "?", "!", ",")
    
    Adding a start and an end token to the sentence
    
    '''
    
#    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z0-9?.'!:_,¿]+", " ", s)
    
    s = s.rstrip().strip()
#     s = f'<start> {s} <end>'
    return s


def sort_freq(pair):
            return pair[1]
    
class GetVoc:
    def __init__(self, unk=1, start=2, end=3, pad=4):
        self.voc = dict()
        self.voc['<unk>'] = unk
        self.voc['<s>'] = start
        self.voc['<\s>'] = end
        self.voc['<pad>'] = pad
    
        self.freq = dict()
        self.freq['<unk>'] = 1
        self.freq['<s>'] = 1
        self.freq['<\s>'] = 1
        self.freq['<pad>'] = 1
        
        self.last_id = 4   
         
    def extend_from_text(self, text, frq_threshold=10):
        text = text.split()
        self._extend_freq(text)
        
        if frq_threshold is None:
            text = set(text)
            for i in text:
                if not self.voc.get(i, 0):
                    self.last_id += 1
                    self.voc[i] = self.last_id
        else:
            for i in sorted(self.freq.items(), key=sort_freq, reverse=True):
                if i[1] >= frq_threshold:
                    if not self.voc.get(i[0], 0):
                        self.last_id += 1
                        self.voc[i[0]] = self.last_id
                
    def extend(self, other): 
        for i in other.keys():
            if not self.voc.get(i, 0):
                self.last_id += 1
                self.voc[i] = self.last_id
    
    def _extend_freq(self, text):
        for word in text:
            if self.freq.get(word, 0):
                self.freq[word] += 1
            else:
                self.freq[word] = 1
       
    def save_voc(self, path, file_name):
        with open(f'{path}/{file_name}.txt', 'w', encoding='utf-8-sig') as f:
            for key, val in self.voc.items():
                f.write(f'{key}=={val}\n')
                
        with open(f'{path}/{file_name}_freq.txt', 'w', encoding='utf-8-sig') as f:
            for key, val in self.freq.items():
                f.write(f'{key}=={val}\n')
                
                
                
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
           
def load_dict(path):
    data = dict()
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            try:
                line = line.decode().strip('\n')
            except:
                line = line.strip('\n')
                
            key, value = line.split('==')
            data[key.strip()] = int(value.strip())
    return data


                
if __name__ == "__main__":
    print(type(padd(librosa.load(r"D:\Study\GitHub\dev\wav\ted_767.wav", duration=3.509999, offset=12.750000)[0], 100)))
    # print(np.max([0, 10000000 - 1000000000000]))