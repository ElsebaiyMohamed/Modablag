import os
try:
    from hepler import english_preprocess, arabic_preprocess, GetVoc
except ImportError:
    from .hepler import english_preprocess, arabic_preprocess, GetVoc

def save_data(file, save_dir, lang='en'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/vocublary', exist_ok=True)

    processor = english_preprocess if  lang == 'en' else arabic_preprocess
        
    with open(file, mode="r", encoding='utf-8-sig') as f: 
        voc = GetVoc()
        i = 0
        for text in f:
            print(end='\r')
            print('AT Line',i , end='')
            try:
                text = processor(text.decode())
            except:
                text = processor(text)
            voc.extend_from_text(text)
            i += 1

        voc.save_voc(f'{save_dir}/vocublary', lang)
        print()
        print(f'{lang} Dict size:', len(voc.voc))

    

if __name__ == '__main__':
    save_data(r"D:\Study\GitHub\dev\text\dev.en", r'D:\Study\GitHub\dev', 'en')
    save_data(r"D:\Study\GitHub\dev\text\dev.ar", r'D:\Study\GitHub\dev', 'ar')