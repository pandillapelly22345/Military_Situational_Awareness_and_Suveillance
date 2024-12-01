import pandas as pd

# Load the annotation and training data
annotation_df = pd.read_csv('mad_dataset_annotation.csv')
training_df = pd.read_csv('training.csv')

# Extract 'video_num' and 'file_id' from the path column in training_df
training_df['video_num'] = training_df['path'].str.extract(r'/(\d+)/')[0]
training_df['file_id'] = training_df['path'].str.extract(r'/(\d+)\.wav')[0]

# Ensure data types match for merging
training_df['video_num'] = training_df['video_num'].astype(int)
training_df['file_id'] = training_df['file_id'].astype(int)
annotation_df['Video_num'] = annotation_df['Video_num'].astype(int)
annotation_df['File_id'] = annotation_df['File_id'].astype(int)

# Merge based on video_num and file_id
merged_training_df = pd.merge(
    training_df, 
    annotation_df, 
    left_on=['video_num', 'file_id'], 
    right_on=['Video_num', 'File_id'], 
    how='inner'
)

# Drop unnecessary columns like duplicated Video_num and File_id
merged_training_df.drop(columns=['Video_num', 'File_id'], inplace=True)

# Save the merged data as a new CSV
merged_training_df.to_csv('merged_training.csv', index=False)

# Repeat for testing set if needed
testing_df = pd.read_csv('test.csv')
testing_df['video_num'] = testing_df['path'].str.extract(r'/(\d+)/')[0]
testing_df['file_id'] = testing_df['path'].str.extract(r'/(\d+)\.wav')[0]

testing_df['video_num'] = testing_df['video_num'].astype(int)
testing_df['file_id'] = testing_df['file_id'].astype(int)

merged_testing_df = pd.merge(
    testing_df, 
    annotation_df, 
    left_on=['video_num', 'file_id'], 
    right_on=['Video_num', 'File_id'], 
    how='inner'
)

merged_testing_df.drop(columns=['Video_num', 'File_id'], inplace=True)
merged_testing_df.to_csv('merged_testing.csv', index=False)
