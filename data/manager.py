from tokenizers import Tokenizer
from tokenizers import decoders



class TokenHandler:
    def __init__(self, json_path):
        self.tok = Tokenizer.from_file(json_path)
        self.tok.decoder = decoders.WordPiece()
        self.tok.enable_padding(pad_id=self.get_id("<PAD>"), pad_token="<PAD>")
        
    def enocde_line(self, text):
        out = self.tok.encode(text)
        return out.ids, out.tokens
    
    def get_id(self, token):
        return self.tok.token_to_id(token)
    
    def encode_batch(self, data):
        output = self.tok.encode_batch(data)
        return [o.ids for o in output]
    
    def decode_line(self, ids):
        return self.tok.decode(ids)
    
    def decode_batch(self, ids):
        return self.tok.decode_batch(ids)
    
    
    
if __name__ == '__main__':
    path = r"D:\Study\GitHub\dev\en_tokenizer.json"
    en_token = TokenHandler(path)
    text = ['salah go salah go salah go to hind', 'salah go to hind', 'salah go to hind']
    ids = en_token.encode_batch(text)
    print(ids)
    tok = en_token.decode_batch(ids)
    print(tok)
    
