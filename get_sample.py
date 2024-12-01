# -*- coding: cp949 -*-
from glob import glob
import pandas as pd
import os
import librosa
import soundfile as sf
from torchaudio import transforms as T

SAVED_LIST = []

def cut_and_save(audio, start_t, end_t, file_num, v_num, data_path):
    start_frame = int(int(start_t) * 16000)
    end_frame = int(int(end_t) * 16000)
    
    new_wav = audio[start_frame:end_frame]
    if len(new_wav) > 160000:
        new_wav = new_wav[:160000]
    
    if not os.path.isfile(os.path.join(data_path, str(file_num)+'.wav')):
        sf.write(os.path.join(data_path, str(file_num)+'.wav'), new_wav, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('{} is saved'.format(os.path.join(data_path, str(file_num)+'.wav')))
    else:
        print('{} is passed'.format(os.path.join(data_path, str(file_num)+'.wav')))


def extract_wav(saved_path, video_num, file_num, start_t, end_t, DATA_DIR, new_DATA_DIR):
    prev_video_num = None
    for i in range(len(video_num)):
        # Remove leading zeros by converting to an integer and then back to a string
        v_num = str(int(video_num[i]))

        # Create the directory for the processed file if it doesn't already exist
        new_save_dir = os.path.join(new_DATA_DIR, v_num)
        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir, exist_ok=True)

        # Try to find the file without leading zeros
        audio_file_path = os.path.join(DATA_DIR, f"{v_num}.wav")
        
        # If the file does not exist, try the zero-padded format (e.g., 099.wav)
        if not os.path.exists(audio_file_path):
            audio_file_path = os.path.join(DATA_DIR, f"{v_num.zfill(3)}.wav")
        
        # If the file is found, proceed with reading and processing
        if os.path.exists(audio_file_path):
            y, sr = sf.read(audio_file_path)
            print(f'Loaded {audio_file_path} with shape {y.shape} and sample rate {sr}')
            if sr != 16000:
                print('Sample rate is not 16000, resampling...')
                y, sr = librosa.load(audio_file_path, sr=sr)
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                print(f'Resampled audio shape: {y.shape}')
            else:
                y, sr = librosa.load(audio_file_path, sr=sr)

            # Call the cut_and_save function
            cut_and_save(y, start_t[i], end_t[i], file_num[i], v_num, new_save_dir)
        else:
            print(f"File {audio_file_path} not found.")
        
        prev_video_num = v_num


## For training set
data_dir = './data/MAD_dataset/wav_files/'
train_save_dir = './data/MAD_dataset/training'
train_label_file = './filtered_training.csv' #
train_label = pd.read_csv(train_label_file)

saved_path = train_label['path'].values.tolist()
video_num = train_label['video_num'].values.tolist()
start_t = train_label['Start_time'].values.tolist()
end_t = train_label['End_time'].values.tolist()
file_num = train_label['file_id'].values.tolist() 

extract_wav(saved_path, video_num, file_num, start_t, end_t, data_dir, train_save_dir) # training set

## For test set
test_save_dir = './data/MAD_dataset/test'
test_label_file = './filtered_testing.csv' #
test_label = pd.read_csv(test_label_file)

saved_path = test_label['path'].values.tolist()
video_num = test_label['video_num'].values.tolist()
start_t = test_label['Start_time'].values.tolist()
end_t = test_label['End_time'].values.tolist()
file_num = test_label['file_id'].values.tolist() 

extract_wav(saved_path, video_num, file_num, start_t, end_t, data_dir, test_save_dir) # test set



    
