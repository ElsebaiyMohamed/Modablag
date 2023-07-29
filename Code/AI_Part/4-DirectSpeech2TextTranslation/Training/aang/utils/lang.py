from pyarabic.trans import normalize_digits
import re
from tokenizers import Tokenizer


class TokenHandler:
    def __init__(self, json_path: str, lang='en', ):
        self.tok = Tokenizer.from_file(json_path)
        self.tok.enable_padding(pad_id=self.get_id("<PAD>"), pad_token="<PAD>")
        if lang == 'en':
            self.pre = self.english_preprocess
        elif lang == 'ar':
            self.pre = self.arabic_preprocess
        else:
            raise NotImplementedError('This class suports En and Ar language only for now')

    
    def arabic_preprocess(self, s: str):
        '''Remove non arabic characters and unnecessary spaces.
        @input: string
        @return: cleaned string
        '''
        s = normalize_digits(s, source='all', out='west')
        s = re.sub(r'[?]+', "؟", s)
        s = re.sub('[" "]+', ' ', s)
        s = re.sub('[^\sء-ي؟:()!,،.:1-9]+', '', s)
        s = s.rstrip().strip()
        return s
    
    def english_preprocess(self, s: str):
        '''Remove non english characters and unnecessary spaces.
        @input: string
        @return: cleaned string
        '''

        s = re.sub(r'[" "]+', " ", s)
        s = re.sub(r"[^\sa-zA-Z0-9?:()!,،'.:]+", "", s)

        s = s.rstrip().strip()
        return s
        
    
    def __call__(self, text, length=None):
        text = self.pre(text)
        out = self.tok.encode(text)
        if length is not None:
            out.pad(length, pad_id=self.get_id("<PAD>"), pad_token="<PAD>")
            out.truncate(length)            
        return out.ids
        
    def encode(self, text: str):
        '''@input: text --> single string.
        @return:  ids, tokens
        '''
        text = self.pre(text)
        out = self.tok.encode(text)
        return out
    
    def get_id(self, token: int):
        '''@input: token --> single word 
        @return: id --> int
        '''
        return self.tok.token_to_id(token)
    
    def encode_batch(self, data: list):
        '''@input: data --> list of strings.
        @return:  ids, tokens
        '''
        data = tuple(map(self.pre, data))
        output = self.tok.encode_batch(data)
        return output
    
    def decode(self, ids: list):
        '''@input: ids --> list of int
        @return: text --> single string.
        '''
        return self.tok.decode(ids)
    
    def decode_batch(self, ids: list):
        return self.tok.decode_batch(ids)


if __name__ == '__main__':
    