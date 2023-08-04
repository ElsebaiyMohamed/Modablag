# Automatic Video Dubbing system from English to Arabic  

This project presents a comprehensive study on video dubbing techniques and the development of a specialized video dubbing system. The objective is to replace the original voices in foreign language videos with the voices of performers speaking the language of the target audience, while ensuring synchronization between lip movements and the dubbed speech.

## Importance of Automatic Video Dubbing

Video dubbing aims to make video content invariant across worldwide cultures. Automatic video dubbing systems typically involve three sub-tasks:

- Automatic Speech Recognition (ASR), which transcribes the original speech into text in the source language.
- Neural Machine Translation (NMT), which translates the source language text to the target language  
- Text-to-Speech (TTS), which synthesizes the translated text into target speech.  

Video dubbing enhances accessibility, engagement, and global distribution of multilingual content while preserving visual integrity for cross-cultural communication.  

## Challenges

Automatic video dubbing faces several challenges:

- Lip sync accuracy
- Naturalness of dubbed voice
- Cultural adaptation and localization
- Multilingual and multicultural considerations
- Code switching.

## Methodology  

The proposed methodology involves:

1. Separating the audio and video from the source English video
2. Translating the English audio to Arabic speech using a speech translator
3. Preserving the original video frames
4. Merging the translated Arabic speech with the video frames to create an Arabic dubbed video

To improve the results, two additional models are used in speech translator:

- Punctuation model to add punctuation to English subtitles
- Tashkeel model to add diacritical marks to Arabic text

## System Architecture

The system follows a modular architecture consisting of:

- User facing apps (Flutter app)

- Application server (localhost and herouku)

- Database server (firebase)

- Machine learning pipelines for ASR, NMT, TTS (Pytorch, Tensorflow and HuggingFace)

The application server handles user management, video uploads/downloads, and interfacing with the ML pipelines. The database stores user data, video metadata, transcripts etc.

## Speech Recognition

- Experiments compared Wave2Vec2.0 and Google Speech Recognition APIs.

- Wave2Vec2.0 gave lower Word Error Rates by pretraining on large unlabeled speech data followed by finetuning on a small labeled dataset.

- CTC loss function used to train acoustic model to convert speech features into character probabilities.

## Machine Translation

- Google's Neural MT architecture has an encoder-decoder structure.

- Training uses parallel English-Arabic corpora like MuST-C dataset.

- Residual connections in the model improve gradient flow during training.

## Text to Speech

- WaveNet generates audio waveforms directly using dilated causal convolutional layers.
- Causal dilated convolutional layers to model audio sample dependencies
- Residual blocks with skip connections to improve training convergence
- Separate conditioning on linguistic and acoustic features
- Parallel waveform generation with WaveNet vocoders during inference

Advancements like FastSpeech 2 have reduced training time by allowing non-autoregressive generation during inference while maintaining voice quality.

## Results

The system is evaluated using metrics like:

- Perplexity
- Word Error Rate (WER)  
- BLEU score

Promising results are achieved on benchmark evaluation datasets.

## Conclusion

The project demonstrates the potential of video dubbing technology to increase accessibility to multimedia content and bridge language gaps. However, there are opportunities for enhancing dubbing accuracy, voice recognition, audio quality, and feature expansion in future work.
