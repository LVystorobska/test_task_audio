import pandas as pd
import os
from pydub import AudioSegment

path = 'dataset/audio'
os.chdir(path)
audio_files = os.listdir()

for file in audio_files:
    name, ext = os.path.splitext(file)
    if ext == ".mp3":
       mp3_sound = AudioSegment.from_mp3(file)
       mp3_sound.export("{0}.wav".format(name), format="wav")

labels = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
file_naming = [f'rec{str(ind+1)}.wav' for ind in range(len(labels))]
audio_labels = pd.DataFrame({
    'audio_name': file_naming,
    'class_id': labels
})

audio_labels.iloc[:-5].to_csv('dataset/audio_labels_train.csv', index=False)
audio_labels.iloc[-5:].to_csv('dataset/audio_labels_test.csv', index=False)

files_in_directory = os.listdir(path)
filtered_files = [file for file in files_in_directory if file.endswith('.mp3')]
for file in filtered_files:
	path_to_file = os.path.join(path, file)
	os.remove(path_to_file)


