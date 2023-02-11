def create_batch(data_dec, en_file, en_dict, ar_file, ar_dict, wave_folder, save_folder, sr=16000, n_batch=20, 
                 batch_size=200, min_ar=1, min_en=1, min_wave=1, max_ar=50, max_en=50, max_wave=16000*120):
    
    en_dict = load_dict(en_dict)
    ar_dict = load_dict(ar_dict)
    os.makedirs(save_folder, exist_ok=True)
    
    en_data   = get_en(en_file)
    
    ar_data   = get_ar(ar_file)
    yaml_data = get_ymal(data_dec)
    f = False
    gc.enable()
    gc.set_threshold(3, 2, 1)
    gc.collect()
    for j in range(n_batch):
        gc.collect()
        i = 0
        en    = [] 
        ar    = []
        waves = []
        
        
        while i < batch_size:
            
            print(end='\r')
            print(i,':', batch_size, end='')
            try:
                en_text = next(en_data)
                ar_text = next(ar_data)
                (duration, offset), wave_path = next(yaml_data)
                wave = get_form_wave(offset, duration, join(wave_folder, wave_path), sr)
                
                en_text = numprize_text(en_text, en_dict)   
                ar_text = numprize_text(ar_text, ar_dict) 
                
                if en_text.shape[0] > max_en or ar_text.shape[0] > max_ar or wave.shape[0] > max_wave:
                    continue

                if len(en_text) < min_en:
                    en_text = repeat(en_text, min_en, max_en)

                if len(ar_text) < min_ar:
                    ar_text = repeat(ar_text, min_ar, max_ar)

                if wave.shape[0] < min_wave:
                    wave = repeat_wave(wave, min_wave, max_wave)
                
                en_text = padd(en_text, max_en)  
                ar_text = padd(ar_text, max_ar) 
                wave = padd(wave, max_wave) 
                en.append(en_text)
                ar.append(ar_text)
                waves.append(wave)
                i += 1
            except Exception as e:
                
                print(' ',e)
                f = True
                i = batch_size
            
        if f:
            print('batch is droped because of its size')
            gc.collect()
            break
            
        ar_type = int
        en_type = int
        wave_type = float
        print('  start save')
        with h5py.File(join(save_folder, f'batch_{j}.h5'), 'w') as fp:
            
            fp.create_dataset('en', data=np.array(en, dtype=en_type))
            fp.create_dataset('ar', dtype=np.int32, data=np.array(ar, dtype=ar_type))
            fp.create_dataset('waves', dtype=np.float32, data=np.array(waves, dtype=wave_type))
            del en, ar, waves
            gc.collect(0)
            print('done')
            fp.close()
        
        gc.collect(1)
