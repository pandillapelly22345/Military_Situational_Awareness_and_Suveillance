import os
import pandas as pd

# Define the directory containing the wav files
wav_files_dir = './data/MAD_dataset/wav_files/'

# Get the list of available video numbers in the wav_files directory
available_video_nums = {int(f.split('.')[0]) for f in os.listdir(wav_files_dir) if f.endswith('.wav')}

# Load the merged training and testing datasets
merged_training_df = pd.read_csv('merged_training.csv')
merged_testing_df = pd.read_csv('merged_testing.csv')

# Filter rows based on presence of video_num in available_video_nums
filtered_training_df = merged_training_df[merged_training_df['video_num'].isin(available_video_nums)]
filtered_testing_df = merged_testing_df[merged_testing_df['video_num'].isin(available_video_nums)]

# Save the filtered data back to CSV
filtered_training_df.to_csv('filtered_training.csv', index=False)
filtered_testing_df.to_csv('filtered_testing.csv', index=False)
