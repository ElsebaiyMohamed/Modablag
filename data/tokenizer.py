import re
import unicodedata
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders
from tokenizers import trainers


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




def gen(file, preprocess):
    for ff in file:
        with open(ff, encoding='utf-8-sig') as f:
            for line in f:
                try: 
                    line = line.decode()
                except:
                    line = line
                # sys.stdout.buffer.write(line.encode())
                yield preprocess(line)


def get_tokenizer(files, save_dir, file_name, preprocess):

    unk_token = "<UNK>"  # token for unknown words
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<PAD>"]  # special tokens
    tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
    

    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    pre_tokenizer = Whitespace()
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                                  pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                  special_tokens=[("[CLS]", 1), ("[SEP]", 2)])

    tokenizer.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=15000, special_tokens=spl_tokens)

    
    print('Start training')
    tokenizer.train_from_iterator(gen(files, preprocess), trainer, )
    print('finish training')
    tokenizer.save(f"{save_dir}/{file_name}.json")
    
    return tokenizer.from_file(f"{save_dir}/{file_name}.json")



if __name__ == '__main__':
    ## download data if not exist
# data link https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

    import os
    # data_path = r'D:\Study\GitHub\dev\wikitext-103-raw'   # put the path of the unziped downlaoded data 
    save_dir = r'D:\Study\GitHub\dev\tokens'
    tokenizer_name = 'ar_tokenizer'
    os.makedirs(save_dir, exist_ok=True)
    # files = os.listdir(data_path)       # put the pathes of the files that you will train on
    # for i, f in enumerate(files):
    #     files[i] = os.path.join(data_path, f)
    files = [r"E:\Programs_implementation\Anaconda\dataaaaaaaaaaaaaaaa\data\dev\txt\dev.ar",
             r"E:\Programs_implementation\Anaconda\dataaaaaaaaaaaaaaaa\data\train\txt\train.ar"]
    
    tokenizer = get_tokenizer(files, save_dir, tokenizer_name, arabic_preprocess)